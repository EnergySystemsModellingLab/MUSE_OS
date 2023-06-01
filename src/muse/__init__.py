"""MUSE model"""
VERSION = "1.0.2"


def _create_logger():
    """Creates the main logger.

    Mainly a convenience function, so logging configuration can happen in one place.
    """
    import logging

    logger = logging.getLogger(name=__name__)
    formatter = "-- %(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"

    try:
        import coloredlogs

        coloredlogs.install(logger=logger, fmt=formatter)
    except ImportError:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(formatter, datefmt="%d-%m-%y %H:%M"))
        logger.addHandler(console)

    logger.setLevel(logging.DEBUG)

    return logger


logger = _create_logger()
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
]
