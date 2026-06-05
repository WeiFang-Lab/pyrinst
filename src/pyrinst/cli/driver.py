import argparse
import logging

from pyrinst.io.logging_config import setup_logging
from pyrinst.potentials import POTENTIAL_REGISTRY
from pyrinst.potentials.base import OnTheFlyPotential
from pyrinst.potentials.executors import Driver, get_driver_id, read_server_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--Potential", type=str.lower, required=True, help="Specify the on-the-fly backend.")
    parser.add_argument("--plugin", help="Custom potential module path.")
    parser.add_argument("-F", "--mainInputFile", help="Main input template file for the backend.")
    parser.add_argument("-A", "--additionalFiles", nargs="+", help="Additional backend input files.")
    parser.add_argument("--hess-method", help="Command for calculating the hessian in the backend.")
    parser.add_argument("--runcmd", help="Command for running the backend.")
    parser.add_argument(
        "--cell",
        nargs="+",
        type=float,
        help="Unit cell for periodic on-the-fly backends such as VASP.",
    )
    parser.add_argument("--working-dir", default=".", help="Working file directory to preserve calculations.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbosity level.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose, log_file="pyrinst-driver.log", err_file="pyrinst-driver.err")
    log = logging.getLogger(__name__)
    if args.plugin:
        import runpy

        runpy.run_path(args.plugin)

    server_info = read_server_info()
    symbols = server_info["symbols"]
    pot_cls = POTENTIAL_REGISTRY[args.Potential.lower()]
    if not issubclass(pot_cls, OnTheFlyPotential):
        raise TypeError("pyrinst-driver only supports on-the-fly potential backends.")

    kwargs = vars(args).copy()
    kwargs["template_input"] = kwargs["mainInputFile"]
    kwargs["add_files"] = kwargs["additionalFiles"]

    driver = Driver(pot_cls(symbols, **kwargs), identity=get_driver_id())
    log.info("starting pyrinst driver worker")
    try:
        driver.run()
    except KeyboardInterrupt:
        log.info("Driver interrupted by user")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
