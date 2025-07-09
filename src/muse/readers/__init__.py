"""Aggregates methods to read data from file."""

from muse.defaults import DATA_DIRECTORY
from muse.readers.csv import *  # noqa: F403
from muse.readers.toml import read_settings  # noqa: F401

DEFAULT_SETTINGS_PATH = DATA_DIRECTORY / "default_settings.toml"
"""Default settings path."""
