import argparse
import logging
import pickle

from pyrinst.cli._common import add_logging_args, add_temperature_args, setup_command_logging
from pyrinst.cli.geom import analyze_rate


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Optimized stationary-point pkl file.")
    add_temperature_args(parser)
    add_logging_args(parser)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("rate", help="Compute rates from an existing geometry pkl.")
    configure_parser(parser)
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    setup_command_logging(args, log_file="pyrinst-rate.log", err_file="pyrinst-rate.err")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    analyze_rate(data, args, parser or argparse.ArgumentParser(), logging.getLogger(__name__))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
