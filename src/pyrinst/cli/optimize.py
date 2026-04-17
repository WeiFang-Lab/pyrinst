import argparse
import json
import logging
import os
import pickle
import runpy
from functools import partial

import numpy as np

from pyrinst.geometries import GEOMETRY_REGISTRY, HarmRef, Instanton, InstRef, PhaseType, TransitionState
from pyrinst.io.formats import Formats
from pyrinst.io.logging_config import setup_logging
from pyrinst.io.xyz import load
from pyrinst.opt import OPTIMIZER_REGISTRY
from pyrinst.potentials import BUILTIN_POTENTIALS, POTENTIAL_REGISTRY, FixAtom
from pyrinst.thermo import analyze
from pyrinst.utils.coordinates import is_linear
from pyrinst.utils.units import Temperature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Initial guess for the optimization in xyz, txt, or pkl format.")
    parser.add_argument("-o", "--output", default="opt_geom", help="Final optimized geometry.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity level.")
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument("-T", "--Temp", type=float, help="Temperature in K")
    temp_group.add_argument("-b", "--beta", type=float, help="Inverse temperature in whatever unit system you're using")
    parser.add_argument(
        "--mode",
        choices=GEOMETRY_REGISTRY.keys(),
        required=True,
        help="Optimize input to minimum, transition state or instanton.",
    )
    parser.add_argument(
        "--phase", choices=[p.value for p in PhaseType], default=PhaseType.GAS, help="Phase of the system"
    )
    parser.add_argument("-l", "--link", nargs="*", default=[], help="Pass min/TS here when optimizing TS/instanton.")
    parser.add_argument(
        "--cell",
        nargs="+",
        type=float,
        help="Unit cell for systems with periodic boundy condidtions, specified either with 3 numbers (cubic cell), "
        "or 9 numbers (abc vectors). Used only by VASP wrapper currently.",
    )
    parser.add_argument("--fix", help="xyz file containing the atoms whose positions are fixed.")
    parser.add_argument(
        "--dx", nargs=2, default=[-1, 0.01], type=float, help="Finite difference step size for fixed atoms."
    )
    parser.add_argument("-P", "--Potential", type=str.lower, required=True, help="Specify the backend Potential.")
    parser.add_argument("--plugin", help="Custom potential module path.")
    parser.add_argument(
        "-F",
        "--mainInputFile",
        help="""Main input file (for a SCF calculation):
      1. vasp: INCAR;
      2. gaussian/orca: single-point input file without the geometry;
      3. custom: json file with parameters for initializing the PES class.
    This is also used by custom pes as the input in the json format.""",
    )
    parser.add_argument(
        "-A",
        "--additionalFiles",
        nargs="+",
        help="""Additional files needed for the electronic structure calculation.
    These files will be copied to every directory where the electronic structure calculation is running.
    Currently implemented for:
      1. vasp: POTCAR (one can set up the VASP_PP_PATH system variable instead), KPOINTS, vdw_kernel.bindat.""",
    )
    parser.add_argument(
        "--hess-method",
        help="Command for calculating the hessian in the electronic structure code.",
    )
    parser.add_argument(
        "--runcmd",
        help="Bash command for running the electronic structure code. If not specified, "
        "the program will guess this based on the PES. You can specify with system the "
        "environment variable 'RUNCMD' instead.",
    )
    parser.add_argument("--working-dir", default=".", help="Working file directory to preserve the calculations.")
    parser.add_argument("--opt", choices=OPTIMIZER_REGISTRY.keys(), default="EF", help="Optimization algorithm to use.")
    parser.add_argument("-g", "--gtol", default=1e-3, type=float, help="Tolerance in gradient for optimization.")
    parser.add_argument(
        "-p",
        "--project",
        action="store_true",
        help="Project out translational, rotational permutational modes to help optimization.",
    )
    parser.add_argument("--maxstep", default=0.3, type=float, help="Max-step in optimization.")
    parser.add_argument("--maxiter", default=10, type=int, help="Max-iters in optimization.")
    parser.add_argument("--no-update", action="store_true", help="Don't update but recompute Hessian at each step.")
    parser.add_argument(
        "-N", "--beads", type=int, help="Number of ring-polymer beads (default chosen from input file)."
    )
    parser.add_argument("-s", "--spread", type=float, help="Spread of initial guess.")
    args = parser.parse_args()

    if args.plugin:
        runpy.run_path(args.plugin)

    prefix, ext = os.path.splitext(args.output)
    if ext in {".xyz", ".txt", ".pkl"}:
        args.output = prefix
    setup_logging(
        verbose=args.verbose,
        log_file=f"{prefix}.log",
        err_file=f"{prefix}.err",
    )
    log = logging.getLogger(__name__)
    prefix, ext = os.path.splitext(args.input)

    if ext == ".pkl":
        with open(args.input, "rb") as f:
            data = pickle.load(f)
        symbols = data.symbols
    else:
        if ext == ".xyz":
            symbols, x, _ = load(args.input, energy_pattern=False)
        elif ext == ".txt":
            x = np.loadtxt(args.input)
            symbols = None
        else:
            msg: str = f"Unknown file format: {args.input}"
            raise ValueError(msg)
    pot_cls = POTENTIAL_REGISTRY[key := args.Potential.lower()]
    if key in BUILTIN_POTENTIALS:
        kwargs = vars(args)
        kwargs["template_input"] = kwargs["mainInputFile"]
        kwargs["add_files"] = kwargs["additionalFiles"]
        if key == "mace":
            kwargs["model_paths"] = kwargs["mainInputFile"]
        pes = pot_cls(symbols, **kwargs)
    else:
        if args.mainInputFile:
            with open(args.mainInputFile) as f:
                main_input = json.load(f)
            if isinstance(main_input, dict):
                pes = pot_cls(**main_input)
            elif isinstance(main_input, list):
                pes = pot_cls(*main_input)
            else:
                raise ValueError(f"Unknown input file format: {args.mainInputFile}")
        else:
            pes = pot_cls()
    if args.fix:
        symbols_fix, x_fix, _ = load(args.fix, energy_pattern=False)
        args.dx[0] = None if args.dx[0] < 0 else args.dx[0]
        args.dx[1] = None if args.dx[1] < 0 else args.dx[1]
        pes.symbols = np.concat((pes.symbols, symbols_fix))
        pes = FixAtom(pes, x_fix, dx=args.dx)

    if ext != ".pkl":
        match args.phase:
            case PhaseType.SOLID | PhaseType.MODEL:
                n_zero = 0
            case PhaseType.LIQUID:
                n_zero = 3
            case PhaseType.GAS:
                n_zero = 3 if len(x) == 1 else (5 if is_linear(x) else 6)
        if ext == ".txt":
            try:
                m = next(getattr(pes, attr) for attr in ("masses", "mass", "m") if hasattr(pes, attr))
            except StopIteration:
                msg = "Custom PES is missing a mass attribute. Expected one of: 'masses', 'mass', or 'm'."
                raise AttributeError(msg) from None
            m = np.atleast_1d(m)
        else:
            m = None
        if args.mode == "inst" and args.spread is None:
            if m is not None:
                x.shape = (len(x), len(m), -1)
            data = TransitionState(x, symbols, n_zero=n_zero, masses=m)
        else:
            if m is not None:
                x.shape = (len(m), -1)
            data = GEOMETRY_REGISTRY[args.mode](x, symbols, n_zero=n_zero, masses=m)

    if args.Temp is not None:
        temp: float | None = args.Temp
        beta: float | None = Temperature.to_beta(temp)
    elif args.beta is not None:
        beta = args.beta
        temp = Temperature.to_kelvin(beta)
    elif isinstance(data, Instanton):
        beta = data.beta
        temp = Temperature.to_kelvin(beta)
    else:
        beta = temp = None

    if len(args.link):
        data.update_links(*[np.load(file, allow_pickle=True) for file in args.link])

    if args.mode in (Instanton.type_alias, InstRef.type_alias):
        if type(data) in (TransitionState, HarmRef):
            data = data.get_inst_guess(args.beads, beta, args.spread)
        data.set_beta(beta)
        if args.beads and args.beads != data.N:
            data.interpolate(args.beads)

    opt = OPTIMIZER_REGISTRY[args.opt](
        order=data.order, potential=pes, maxstep=args.maxstep, project=args.project, update=not args.no_update
    )
    opt.search(data, gtol=args.gtol, maxiter=args.maxiter, callback=partial(type(data).output, filename=args.output))

    data.final_output(args.output)

    if isinstance(data, InstRef):
        return

    if beta is None:
        log.info("Temperature not specified. Program terminated.")
        return

    log.info("\nComputing rate...")
    fmt: str = Formats.BETA
    log.info(f"T = {temp:{fmt}} K, 1000/T(K) = {1000 / temp:{fmt}}; beta = {beta:{fmt}}")

    analyze(data, beta)


if __name__ == "__main__":
    main()
