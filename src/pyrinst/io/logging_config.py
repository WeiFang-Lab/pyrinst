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



def log_exception(exc_type, exc_value, exc_traceback):
    """
    Log an unhandled exception using the logging system.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# The main setup function using dictConfig
def setup_logging(
    verbose=False,
    log_file="run.log",
    err_file="run.err",
    console=True,
):
    """
    Set up the logging configuration for the entire application.

    This function configures several handlers:
    1. A stream handler to print logs to the console (stdout).
    2. A file handler to save all logs to a main log file (.log).
    3. A file handler to capture warnings and exceptions to an .err file.

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
    err_file : str or pathlib.Path, optional
        Path to the file capturing warnings and unhandled exceptions.
        (default is "run.err").
    console : bool, optional
        If True, output is also sent to the console (default is True).
    """
    console_level = "DEBUG" if verbose else "INFO"
    active_handlers = []
    warning_handlers = []

    if console:
        active_handlers.append("console")
        warning_handlers.append("console")

    if verbose:
        # Ensure log directories exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(err_file).parent.mkdir(parents=True, exist_ok=True)
        
        active_handlers.extend(["main_log_file", "err_file"])
        warning_handlers.append("err_file")

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
        },
        "loggers": {
            # Specific logger for warnings captured by logging.captureWarnings(True)
            "py.warnings": {
                "handlers": warning_handlers,
                "level": "WARNING",
                "propagate": False,
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": active_handlers,
        },
    }

    if verbose:
        LOGGING_CONFIG["handlers"]["main_log_file"] = {
            "class": "logging.FileHandler",
            "level": console_level,
            "formatter": "file_formatter",
            "filename": log_file,
            "mode": "w",
            "encoding": "utf-8",
        }
        LOGGING_CONFIG["handlers"]["err_file"] = {
            "class": "logging.FileHandler",
            "level": "WARNING",
            "formatter": "file_formatter",
            "filename": err_file,
            "mode": "w",
            "encoding": "utf-8",
        }

    logging.config.dictConfig(LOGGING_CONFIG)

    # Capture standard warnings
    logging.captureWarnings(True)

    # Capture unhandled exceptions
    sys.excepthook = log_exception
    