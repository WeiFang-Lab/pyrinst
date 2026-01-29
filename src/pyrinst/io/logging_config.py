"""
A module for configuring the application's logging system.

This module defines a custom 'RESULT' logging level and provides a setup
function that configures logging handlers for console output, a main log file,
and a dedicated results file.
"""

import logging
import logging.config
import sys
from pathlib import Path

# Define the custom "RESULT" level and attach it to the Logger class
RESULT_LEVEL_NUM = 25
logging.addLevelName(RESULT_LEVEL_NUM, "RESULT")


def result(self, message, *args, **kws):
    """
    Log a message with severity 'RESULT'.

    To use this, you must first call logging.addLevelName().

    Parameters
    ----------
    message : str
        The message to be logged.
    *args : tuple
        Variable length argument list.
    **kws : dict
        Arbitrary keyword arguments.
    """
    if self.isEnabledFor(RESULT_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(RESULT_LEVEL_NUM, message, args, **kws)


logging.Logger.result = result


# The main setup function using dictConfig
def setup_logging(verbose=False, log_file="run.log", result_file="run.result"):
    """
    Set up the logging configuration for the entire application.

    This function configures three handlers:
    1. A stream handler to print logs to the console (stdout).
    2. A file handler to save all logs to a main log file.
    3. A file handler to save only 'RESULT' level logs to a results file.

    The verbosity of the console and main log file is controlled by the
    `verbose` flag.

    Parameters
    ----------
    verbose : bool, optional
        If True, the logging level for console and main log is set to 'DEBUG'.
        Otherwise, it is set to 'INFO' (default is False).
    log_file : str or pathlib.Path, optional
        Path to the main log file which stores all log records.
        (default is "run.log").
    result_file : str or pathlib.Path, optional
        Path to the results file which stores only 'RESULT' level records.
        (default is "run.result").

    Examples
    --------
    Here is how to use it in your main application entry point:

    >>> import argparse
    >>> from logging_setup import setup_logging
    ...
    >>> if __name__ == "__main__":
    ...     parser = argparse.ArgumentParser()
    ...     parser.add_argument("-v", "--verbose", action="store_true")
    ...     parser.add_argument("--basename", default="run")
    ...     args = parser.parse_args()
    ...
    ...     setup_logging(
    ...         verbose=args.verbose,
    ...         log_file=f"{args.basename}.log",
    ...         result_file=f"{args.basename}.result"
    ...     )
    ...
    ...     log = logging.getLogger(__name__)
    ...     log.info("Logging is set up.")

    """
    # Ensure log directories exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)

    console_level = "DEBUG" if verbose else "INFO"

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console_formatter": {"format": "%(message)s"},
            "file_formatter": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            # Handler for console output
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "console_formatter",
                "stream": sys.stdout,
            },
            # Handler for the main log file (mirrors console)
            "main_log_file": {
                "class": "logging.FileHandler",
                "level": console_level,
                "formatter": "file_formatter",
                "filename": log_file,
                "mode": "w",
                "encoding": "utf-8",
            },
            # Handler for the dedicated results file
            "result_file": {
                "class": "logging.FileHandler",
                "level": "RESULT",  # Only captures RESULT level
                "formatter": "console_formatter",  # Use simple format
                "filename": result_file,
                "mode": "w",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",  # Lowest level to capture all messages
            "handlers": ["console", "main_log_file", "result_file"],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
    