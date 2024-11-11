"""MUSE model."""

import os

VERSION = "1.2.2"


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


def add_file_logger(file_path: str | None) -> None:
    """Adds a file logger to the main logger.

    If the file already exists, it is deleted.

    Args:
        file_path (str): Path to the file where the logs will be written.
    """
    import datetime
    import logging
    from pathlib import Path

    if not file_path:
        file_path = (
            f"muse_{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S-%f')}.log"
        )

    if (path := Path(file_path)).exists():
        path.unlink()

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger(name=__name__).addHandler(file_handler)


logger = _create_logger(os.environ.get("MUSE_COLOR_LOG") != "False")
""" Main logger """

__all__ = [
    "Agent",
    "create_agent",
    "read_global_commodities",
    "read_initial_capacity",
    "read_io_technodata",
    "read_technodictionary",
    "read_technologies",
    "read_timeslice_shares",
    "read_csv_timeslices",
    "read_settings",
    "read_macro_drivers",
    "read_csv_agent_parameters",
    "decisions",
    "demand_share",
    "filters",
    "hooks",
    "interactions",
    "investments",
    "objectives",
    "outputs",
    "sectors",
    "legacy_sectors",
    VERSION,
]
