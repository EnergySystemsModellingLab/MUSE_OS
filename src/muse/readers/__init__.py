"""Aggregates methods to read data from file."""

from muse.defaults import DATA_DIRECTORY
from muse.readers.csv import *  # noqa: F403
from muse.readers.toml import read_settings, read_timeslices  # noqa: F401

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


def kebab_to_camel(string):
    return "".join(x.capitalize() for x in string.split("-"))


def snake_to_kebab(string: str) -> str:
    from re import sub

    result = sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r"-\1", string)
    return result.lower()
