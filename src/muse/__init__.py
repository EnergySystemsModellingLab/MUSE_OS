"""MUSE model."""

from __future__ import annotations

import os
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

with suppress(PackageNotFoundError):
    __version__ = version("MUSE_OS")


def _create_logger(color: bool = True):
    """Creates the main logger.

    Mainly a convenience function, so logging configuration can happen in one place.
    """
    import logging

    logger = logging.getLogger(name=__name__)
    formatter = "-- %(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"

    if color:
        try:
            import coloredlogs

            coloredlogs.install(logger=logger, fmt=formatter)
        except ImportError:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(formatter, datefmt="%d-%m-%y %H:%M"))
            logger.addHandler(console)
    else:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(formatter, datefmt="%d-%m-%y %H:%M"))
        logger.addHandler(console)

    logger.setLevel(logging.DEBUG)

    return logger


def add_file_logger() -> None:
    """Adds a file logger to the main logger.

    The file logger is split into two files: one for INFO and DEBUG messages, and one
    for WARNING messages and above to avoid cluttering the main log file and highlight
    potential issues.
    """
    import logging
    from pathlib import Path

    from .defaults import DEFAULT_OUTPUT_DIRECTORY

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    DEFAULT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Sets the warning log, for warnings and above
    warning_file = Path(DEFAULT_OUTPUT_DIRECTORY) / "muse_warning.log"
    if warning_file.exists():
        warning_file.unlink()

    warning_file_handler = logging.FileHandler(warning_file)
    warning_file_handler.setLevel(logging.WARNING)
    warning_file_handler.setFormatter(formatter)
    warning_file_handler.filters = [lambda record: record.levelno > logging.INFO]

    logging.getLogger("muse").addHandler(warning_file_handler)

    # Sets the info log, for debug and info only
    info_file = Path(DEFAULT_OUTPUT_DIRECTORY) / "muse_info.log"
    if info_file.exists():
        info_file.unlink()

    info_file_handler = logging.FileHandler(info_file)
    info_file_handler.setLevel(logging.DEBUG)
    info_file_handler.setFormatter(formatter)
    info_file_handler.filters = [lambda record: record.levelno <= logging.INFO]

    logging.getLogger("muse").addHandler(info_file_handler)


logger = _create_logger(os.environ.get("MUSE_COLOR_LOG") != "False")
""" Main logger """

__all__ = [
    "Agent",
    "create_agent",
    "decisions",
    "demand_share",
    "filters",
    "hooks",
    "interactions",
    "investments",
    "objectives",
    "outputs",
    "read_agent_parameters",
    "read_global_commodities",
    "read_initial_capacity",
    "read_io_technodata",
    "read_macro_drivers",
    "read_settings",
    "read_technodictionary",
    "read_technologies",
    "read_timeslice_shares",
    "sectors",
]
