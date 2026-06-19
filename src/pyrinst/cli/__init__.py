"""Command-line interface for PyRInst."""

import argparse

from pyrinst.cli import driver, fep_eval, geom, plot, rate, sampling


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyrinst")
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{geom,rate,sample,fep-eval,plot,driver}",
        required=True,
    )
    geom.add_parser(subparsers)
    rate.add_parser(subparsers)
    sampling.add_parser(subparsers)
    fep_eval.add_parser(subparsers)
    plot.add_parser(subparsers)
    driver.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args, parser)
