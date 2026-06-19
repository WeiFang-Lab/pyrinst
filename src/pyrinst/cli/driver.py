import argparse
import logging

from pyrinst.cli._common import add_backend_args, add_logging_args, build_potential, setup_command_logging
from pyrinst.potentials.base import OnTheFlyPotential
from pyrinst.potentials.executors import Driver, get_driver_id, read_server_info


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_backend_args(parser, required=True)
    add_logging_args(parser)


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("driver", help="Run an external potential driver worker.")
    configure_parser(parser)
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace, _parser: argparse.ArgumentParser | None = None) -> None:
    setup_command_logging(args, log_file="pyrinst-driver.log", err_file="pyrinst-driver.err")
    log = logging.getLogger(__name__)

    server_info = read_server_info()
    symbols = server_info["symbols"]
    potential = build_potential(symbols, args, pass_symbols_to_custom=True)
    if not isinstance(potential, OnTheFlyPotential):
        raise TypeError("pyrinst driver only supports on-the-fly potential backends.")

    driver = Driver(potential, identity=get_driver_id())
    log.info("starting pyrinst driver worker")
    try:
        driver.run()
    except KeyboardInterrupt:
        log.info("Driver interrupted by user")
    finally:
        driver.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
