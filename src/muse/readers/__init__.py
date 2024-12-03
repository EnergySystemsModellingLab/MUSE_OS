"""Aggregates methods to read data from file."""

from muse.defaults import DATA_DIRECTORY
from muse.readers.csv import *  # noqa: F403
from muse.readers.toml import read_settings  # noqa: F401

DEFAULT_SETTINGS_PATH = DATA_DIRECTORY / "default_settings.toml"
"""Default settings path."""


def camel_to_snake(name: str) -> str:
    """Transforms CamelCase to snake_case."""
    from re import sub

    re = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    result = sub("([a-z0-9])([A-Z])", r"\1_\2", re).lower()
    result = result.replace("co2", "CO2")
    result = result.replace("ch4", "CH4")
    result = result.replace("n2_o", "N2O")
    result = result.replace("f-gases", "F-gases")
    return result
