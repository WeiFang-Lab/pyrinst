import argparse
import logging
import os
import pickle
from functools import partial

import numpy as np

from pyrinst.cli._common import (
    add_backend_args,
    add_logging_args,
    add_temperature_args,
    build_optimization_executor,
    setup_command_logging,
)
from pyrinst.geometries import (
    GEOMETRY_REGISTRY,
    HarmRef,
    Instanton,
    InstRef,
    PhaseType,
    StationaryPoint,
    TransitionState,
)
from pyrinst.io.formats import Formats
from pyrinst.io.xyz import load
from pyrinst.opt import OPTIMIZER_REGISTRY
from pyrinst.potentials import (
    CachedExecutor,
    FixAtom,
    Level,
    ParallelExecutor,
    SingleExecutor,
)
from pyrinst.thermo import analyze
from pyrinst.utils.coordinates import is_linear
from pyrinst.utils.units import CM_1, KB, Temperature

GEOM_MODES = ("single", *GEOMETRY_REGISTRY.keys())
HBAR: float = 1


def _close_remote_driver(executor) -> None:
    if isinstance(executor, ParallelExecutor):
        executor.close()
    elif isinstance(executor, CachedExecutor):
        _close_remote_driver(executor.executor)


def _potential_owner(executor):
    if isinstance(executor, ParallelExecutor):
        return executor
    if isinstance(executor, CachedExecutor):
        return _potential_owner(executor.executor)
    if isinstance(executor, SingleExecutor):
        return executor.potential
    return executor


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input geometry in xyz, txt, or pkl format.")
    parser.add_argument("-o", "--output", help="Output prefix. Defaults to ref for single mode, opt_geom otherwise.")
    add_logging_args(parser)
    add_temperature_args(parser)
    parser.add_argument(
        "--mode",
        type=str.lower,
        choices=GEOM_MODES,
        required=True,
        help="Geometry operation: single computes a one-geometry reference; others optimize that geometry type.",
    )
    parser.add_argument(
        "--phase", choices=[p.value for p in PhaseType], default=PhaseType.GAS.value, help="Phase of the system"
    )
    parser.add_argument("-l", "--link", nargs="*", default=[], help="Pass min/TS here when optimizing TS/instanton.")
    parser.add_argument("--fix", help="xyz file containing the atoms whose positions are fixed.")
    parser.add_argument(
        "--dx", nargs=2, default=[-1, 0.01], type=float, help="Finite difference step size for fixed atoms."
    )
    add_backend_args(parser, include_parallel=True)
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


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("geom", help="Build geometry objects.")
    configure_parser(parser)
    parser.set_defaults(func=run)
    return parser


def normalize_output_prefix(output: str) -> str:
    prefix, ext = os.path.splitext(output)
    return prefix if ext in {".xyz", ".txt", ".pkl"} else output


def load_input_geometry(filename: str):
    prefix, ext = os.path.splitext(filename)
    if ext == ".pkl":
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data, data.symbols, None, ext
    if ext == ".xyz":
        symbols, x, _ = load(filename, energy_pattern=False)
        return None, symbols, x, ext
    if ext == ".txt":
        return None, None, np.loadtxt(filename), ext
    msg: str = f"Unknown file format: {filename}"
    raise ValueError(msg)


def make_initial_geometry(args: argparse.Namespace, data, symbols, x, ext: str, executor):
    if ext == ".pkl":
        return data

    phase = PhaseType(args.phase)
    match phase:
        case PhaseType.SOLID | PhaseType.MODEL:
            n_zero = 0
        case PhaseType.LIQUID:
            n_zero = 3
        case PhaseType.GAS:
            n_zero = 3 if len(x) == 1 else (5 if is_linear(x) else 6)

    if ext == ".txt":
        source = _potential_owner(executor)
        try:
            m = next(getattr(source, attr) for attr in ("masses", "mass", "m") if hasattr(source, attr))
        except StopIteration:
            msg = "Custom PES is missing a mass attribute. Expected one of: 'masses', 'mass', or 'm'."
            raise AttributeError(msg) from None
        m = np.atleast_1d(m)
    else:
        m = None

    if args.mode == "inst" and args.spread is None:
        if m is not None:
            x.shape = (len(x), len(m), -1)
        return TransitionState(x, symbols, n_zero=n_zero, masses=m)

    if m is not None:
        x.shape = (len(m), -1)
    return GEOMETRY_REGISTRY[args.mode](x, symbols, n_zero=n_zero, masses=m)


def resolve_temperature(args: argparse.Namespace, data):
    if args.temperature is not None:
        temp: float | None = args.temperature
        beta: float | None = Temperature.to_beta(temp)
    elif args.beta is not None:
        beta = args.beta
        temp = Temperature.to_kelvin(beta)
    elif isinstance(data, Instanton):
        beta = data.beta
        temp = Temperature.to_kelvin(beta)
    else:
        beta = temp = None
    return temp, beta


def apply_fixed_atoms(args: argparse.Namespace, executor):
    if not args.fix:
        return executor

    symbols_fix, x_fix, _ = load(args.fix, energy_pattern=False)
    args.dx[0] = None if args.dx[0] < 0 else args.dx[0]
    args.dx[1] = None if args.dx[1] < 0 else args.dx[1]
    calculator = _potential_owner(executor)
    calculator.symbols = np.concat((calculator.symbols, symbols_fix))
    return FixAtom(executor, x_fix, dx=args.dx)


def run_single(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.output = normalize_output_prefix(args.output or "ref")
    data, symbols, x, ext = load_input_geometry(args.input)
    if ext != ".xyz":
        parser.error("--mode single requires an xyz input geometry.")

    executor = build_optimization_executor(symbols, args, parser)
    executor = apply_fixed_atoms(args, executor)
    try:
        reference = HarmRef(
            x,
            getattr(_potential_owner(executor), "symbols_raw", symbols),
            n_zero=(5 if is_linear(x) else 6),
        )
        executor.compute(reference, level=Level.FREQ)
        reference.calc_freq()
        freqs_nonzero = reference.freqs[np.argpartition(abs(reference.freqs), reference.n_zero)[reference.n_zero :]]
        min_freq: float = min(freqs_nonzero)
        if min_freq > 0:
            print("All frequencies are real.")
        else:
            print(f"max imaginary freq: {reference.freqs[0] / CM_1:{Formats.FREQUENCY}} cm^-1")
            print(f"crossover T: {1 / (KB * 2 * np.pi / (HBAR * -min_freq))} K")
        reference.save(args.output)
    finally:
        _close_remote_driver(executor)


def analyze_rate(data, args: argparse.Namespace, parser: argparse.ArgumentParser, log: logging.Logger) -> None:
    if isinstance(data, InstRef):
        parser.error("rate analysis does not support InstRef; use pyrinst fep-eval for FEP references.")
    if not isinstance(data, StationaryPoint):
        parser.error(f"rate analysis requires a stationary-point pkl, got {type(data).__name__}.")

    temp, beta = resolve_temperature(args, data)
    if beta is None:
        parser.error("rate analysis requires -T/--temperature or --beta unless the input stores beta.")

    log.info("\nComputing rate...")
    fmt: str = Formats.BETA
    log.info(f"T = {temp:{fmt}} K, 1000/T(K) = {1000 / temp:{fmt}}; beta = {beta:{fmt}}")
    analyze(data, beta)


def evaluate_current_geometry(data, executor) -> None:
    executor.compute(data, level=Level.FREQ)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    if parser is None:
        parser = argparse.ArgumentParser()
    if args.mode == "single":
        run_single(args, parser)
        return

    args.output = args.output or "opt_geom"
    prefix, ext = os.path.splitext(args.output)
    args.output = normalize_output_prefix(args.output)
    setup_command_logging(
        args,
        log_file=f"{prefix}.log",
        err_file=f"{prefix}.err",
    )
    log = logging.getLogger(__name__)
    data, symbols, x, ext = load_input_geometry(args.input)

    executor = build_optimization_executor(symbols, args, parser)
    executor = apply_fixed_atoms(args, executor)

    try:
        data = make_initial_geometry(args, data, symbols, x, ext, executor)
        temp, beta = resolve_temperature(args, data)

        if len(args.link):
            data.update_links(*[np.load(file, allow_pickle=True) for file in args.link])

        if args.mode in (Instanton.type_alias, InstRef.type_alias):
            if type(data) in (TransitionState, HarmRef):
                data = data.get_inst_guess(args.beads, beta, args.spread)
            data.set_beta(beta)
            if args.beads and args.beads != data.N:
                data.interpolate(args.beads)

        if args.maxiter == 0:
            evaluate_current_geometry(data, executor)
        else:
            opt = OPTIMIZER_REGISTRY[args.opt](
                order=data.order,
                executor=executor,
                maxstep=args.maxstep,
                project=args.project,
                update=not args.no_update,
            )
            opt.search(
                data,
                gtol=args.gtol,
                maxiter=args.maxiter,
                callback=partial(type(data).output, filename=args.output),
            )

        data.final_output(args.output)

        if beta is None or isinstance(data, InstRef):
            log.info("Rate not computed. Run `pyrinst rate` with a temperature to analyze this geometry.")
        else:
            analyze_rate(data, args, parser, log)
    finally:
        _close_remote_driver(executor)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    run(parser.parse_args(argv), parser)


def configure_gen_ref_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Centroid structure in xyz format.")
    parser.add_argument("-o", "--output", default="ref", help="filename of reference pkl.")
    add_backend_args(parser, required=True)
    parser.add_argument("--model-path", dest="input_file", help="path of MACE model", type=str)
    parser.add_argument("--dtype", help="dtype of MACE model", type=str, default="float64")
    parser.add_argument("--device", help="device which model runs on", type=str, default="cuda")
    parser.add_argument("--enable_cueq", action="store_true", help="Enable CUEQ (default: disabled)")


def gen_ref_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pyrinst-gen-ref")
    configure_gen_ref_parser(parser)
    args = parser.parse_args(argv)
    args.mode = "single"
    args.parallel = False
    args.fix = None
    args.dx = [-1, 0.01]
    run_single(args, parser)


def optimize_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pyrinst-optimize")
    configure_parser(parser)
    run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
