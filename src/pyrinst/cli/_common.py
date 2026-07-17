import argparse
import json
import runpy
from typing import Any

from pyrinst.io.logging_config import setup_logging
from pyrinst.potentials import (
    BUILTIN_POTENTIALS,
    POTENTIAL_REGISTRY,
    CachedExecutor,
    ParallelExecutor,
    SingleExecutor,
)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity level.")


def setup_command_logging(args: argparse.Namespace, *, log_file: str, err_file: str) -> None:
    setup_logging(verbose=args.verbose, log_file=log_file, err_file=err_file)


def add_temperature_args(parser: argparse.ArgumentParser) -> None:
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument("-T", "--temperature", type=float, help="Temperature in K")
    temp_group.add_argument("-b", "--beta", type=float, help="Inverse temperature in internal units")


def add_backend_args(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    include_parallel: bool = False,
) -> None:
    group = parser.add_argument_group("backend")
    group.add_argument("-P", "--potential", type=str.lower, required=required, help="Potential backend.")
    group.add_argument("--plugin", help="Custom potential module path.")
    group.add_argument(
        "-F",
        "--input-file",
        help=(
            "Main backend input file: an electronic-structure template, a MACE model, "
            "or a JSON initializer for a custom potential."
        ),
    )
    parser.add_argument("--device", help="device which model runs on", type=str, default="cpu")
    group.add_argument("-A", "--additional-files", nargs="+", help="Additional backend input files.")
    group.add_argument("--hess-method", help="Backend command or method for Hessian calculations.")
    group.add_argument("--runcmd", help="Command for running the backend.")
    group.add_argument(
        "--cell",
        nargs="+",
        type=float,
        help="Unit cell for periodic backends, specified with 3 or 9 numbers.",
    )
    group.add_argument("--working-dir", default=".", help="Working directory for backend calculations.")
    if include_parallel:
        group.add_argument("--parallel", action="store_true", help="Use external pyrinst driver workers.")


def load_plugin(path: str | None) -> None:
    if path:
        runpy.run_path(path)


def get_potential_class(name: str):
    return POTENTIAL_REGISTRY[name.lower()]


def make_builtin_potential_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = vars(args).copy()
    kwargs["template_input"] = args.input_file
    kwargs["add_files"] = args.additional_files
    if args.potential == "mace":
        kwargs["model_paths"] = args.input_file
    return kwargs


def build_potential(symbols, args: argparse.Namespace, *, pass_symbols_to_custom: bool = False):
    load_plugin(args.plugin)
    pot_cls = get_potential_class(args.potential)
    key = args.potential.lower()
    if key in BUILTIN_POTENTIALS:
        return pot_cls(symbols, **make_builtin_potential_kwargs(args))
    if pass_symbols_to_custom:
        return pot_cls(symbols, **make_builtin_potential_kwargs(args))

    if args.input_file:
        with open(args.input_file) as f:
            main_input = json.load(f)
        if isinstance(main_input, dict):
            return pot_cls(**main_input)
        if isinstance(main_input, list):
            return pot_cls(*main_input)
        raise ValueError(f"Unknown input file format: {args.input_file}")
    return pot_cls()


def build_cached_executor(symbols, args: argparse.Namespace) -> CachedExecutor:
    potential = build_potential(symbols, args)
    return CachedExecutor(SingleExecutor(potential, working_dir=args.working_dir))


def build_optimization_executor(symbols, args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.parallel:
        if symbols is None:
            raise ValueError("Parallel on-the-fly drivers require atomic symbols from the input geometry.")
        return ParallelExecutor(symbols)
    if args.potential is None:
        parser.error("-P/--potential is required unless --parallel is used.")
    return build_cached_executor(symbols, args)
