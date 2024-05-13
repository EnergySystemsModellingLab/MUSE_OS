"""Default global values used in keyword arguments."""

from pathlib import Path

try:
    import SGIModelData

    DEFAULT_SECTORS_DIRECTORY = SGIModelData.PATH
except ImportError:
    DEFAULT_SECTORS_DIRECTORY = Path().cwd() / "data"

DATA_DIRECTORY = Path(__file__).parent / "data"
""" Standard data directory."""

""" Default root directory with sector data """
DEFAULT_OUTPUT_DIRECTORY = Path("Results")
""" Default root directory with sector data """
DEFAULT_SETTINGS_DIRECTORY = Path.cwd()
""" Directory where to look for `settings.toml`."""
