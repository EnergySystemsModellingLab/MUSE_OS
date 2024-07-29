"""Ensemble of functions to read MUSE data."""

__all__ = ["read_settings"]

import importlib.util as implib
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import (
    IO,
    Any,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr

from muse.decorators import SETTINGS_CHECKS, register_settings_check
from muse.defaults import DATA_DIRECTORY, DEFAULT_SECTORS_DIRECTORY

DEFAULT_SETTINGS_PATH = DATA_DIRECTORY / "default_settings.toml"
"""Default settings path."""


class InputError(Exception):
    """Root for TOML input errors."""


class MissingSettings(InputError):
    """Error when an input is missing."""


class IncorrectSettings(InputError):
    """Error when an input exists but is incorrect."""


def convert(dictionary):
    """Converts a dictionary (with nested ones) to a nametuple."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert(value)
    return namedtuple("MUSEOptions", dictionary.keys())(**dictionary)


def undo_damage(nt):
    """Unconvert nested nametuple."""
    if not hasattr(nt, "_asdict"):
        return nt
    result = nt._asdict()
    for key, value in result.items():
        result[key] = undo_damage(value)
    return result


class FormatDict(dict):
    """Allows partial formatting of a string."""

    def __missing__(self, key):
        return FormatDict.FormatPlaceholder(key)

    class FormatPlaceholder:
        def __init__(self, key):
            self.key = key

        def __format__(self, spec):
            result = f"{self.key}:{spec}" if spec else self.key
            return f"{{{result}}}"


def format_path(
    filepath: str,
    replacements: Optional[Mapping] = None,
    path: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    muse_sectors: Optional[str] = None,
):
    """Replaces known patterns in a path.

    Unknown patterns are left alone. This allows downstream object factories to format
    the paths according to their own specifications.
    """
    from string import Formatter

    patterns = FormatDict(
        {
            **{
                "cwd": Path("" if cwd is None else cwd).absolute(),
                "muse_sectors": Path(
                    DEFAULT_SECTORS_DIRECTORY if muse_sectors is None else muse_sectors
                ).absolute(),
                "path": Path("" if path is None else path).absolute(),
            },
            **({} if replacements is None else replacements),
        }
    )
    formatter = Formatter()
    return str(Path(formatter.vformat(str(filepath), (), patterns)).absolute())


def format_paths(
    settings: Mapping,
    replacements: Optional[Mapping] = None,
    path: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    muse_sectors: Optional[str] = None,
    suffixes: Sequence[str] = (".csv", ".nc", ".xls", ".xlsx", ".py", ".toml"),
):
    """Format paths passed to settings.

    A setting is recognized as a path if it's name ends in `_path`, `_file`, or `_dir`,
    or if the associated value is text object and ends with `.csv, as well as settings
    called `path`.

    Paths are first formatted using the input replacement keywords. These replacements
    will include "cwd" and "sectors" by default. For simplicity, any item called `path`
    is considered first in any dictionary, and then used within that dictionary and
    nested dictionaries.

    Examples:
        Starting from a simple example, we see `data_path` has been modified to point to
        the current working directory:

        >>> from pathlib import Path
        >>> from muse.readers.toml import format_paths
        >>> a = format_paths({"a_path": "{cwd}/a/b/c"})
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_path"]
        True

        Or it can be modified to point to the default locations for sectorial data:

        >>> from muse.defaults import DEFAULT_SECTORS_DIRECTORY
        >>> a = format_paths({"a_path": "{muse_sectors}/a/b/c"})
        >>> str(DEFAULT_SECTORS_DIRECTORY.absolute() / "a" / "b" / "c") == a["a_path"]
        True

        Similarly, if not given, `path` defaults to the current working directory:

        >>> a = format_paths({"a_path": "{path}/a/b/c"})
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_path"]
        True

        However, it can be made to point to anything of interest:

        >>> a = format_paths({"path": "{cwd}/a/b", "a_path": "{path}/c"})
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_path"]
        True

        Any property ending in `_path`, `_dir`, `_file`, or with a value that can be
        interpreted as a path with suffix `.csv`, `.nc`, `.xls`, `.xlsx`, `.py` or
        `.toml` is considered a path and transformed:

        >>> a = format_paths({"path": "{cwd}/a/b", "a_dir": "{path}/c"})
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_dir"]
        True
        >>> a = format_paths({"path": "{cwd}/a/b", "a_file": "{path}/c"})
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_file"]
        True
        >>> a = format_paths({"path": "{cwd}/a/b", "a": "{path}/c.csv"})
        >>> str(Path().absolute() / "a" / "b" / "c.csv") == a["a"]
        True
        >>> a = format_paths({"path": "{cwd}/a/b", "a": "{path}/c.toml"})
        >>> str(Path().absolute() / "a" / "b" / "c.toml") == a["a"]
        True

        Finally, paths in nested directories are also processed:

        >>> a = format_paths(
        ...     {
        ...         "path": "{cwd}/a/b",
        ...         "nested": { "a_path": "{path}/c" }
        ...     }
        ... )
        >>> str(Path().absolute() / "a" / "b" / "c") == a["nested"]["a_path"]
        True

        Note that `path` points to the latest one:

        >>> a = format_paths(
        ...     {
        ...         "path": "{cwd}/a/b",
        ...         "a_path": "{path}/c",
        ...         "nested": {
        ...             "path": "{cwd}/toot/suite",
        ...             "b_path": "{path}/c"
        ...         }
        ...     }
        ... )
        >>> str(Path().absolute() / "a" / "b" / "c") == a["a_path"]
        True
        >>> str(Path().absolute() / "toot" / "suite" / "c") == a["nested"]["b_path"]
        True
    """
    import re
    from pathlib import Path

    patterns = {
        **{
            "cwd": Path("" if cwd is None else cwd).absolute(),
            "muse_sectors": Path(
                DEFAULT_SECTORS_DIRECTORY if muse_sectors is None else muse_sectors
            ).absolute(),
            "path": Path("" if path is None else path).absolute(),
        },
        **({} if replacements is None else replacements),
    }

    def format(path: str) -> str:
        if path.lower() in ("optional", "required"):
            return path
        return format_path(path, **patterns)  # type: ignore

    path_names = (
        re.compile(r"_path$"),
        re.compile("_dir$"),
        re.compile("_file$"),
        re.compile("filename"),
    )

    def is_a_path(key, value):
        return any(re.search(x, key) is not None for x in path_names) or (
            isinstance(value, str) and Path(value).suffix in suffixes
        )

    path = format(settings.get("path", str(patterns["path"])))
    patterns["path"] = path  # type: ignore

    result = dict(**settings)
    if "path" in settings:
        result["path"] = path
    for key, value in result.items():
        if is_a_path(key, value):
            result[key] = format(value)
        elif isinstance(value, Mapping):
            result[key] = format_paths(value, patterns, path)
        elif isinstance(value, list):
            result[key] = [
                format_paths(item, patterns, path)
                if isinstance(item, Mapping)
                else format_path(item, patterns, path)
                if is_a_path("", item)
                else item
                for item in result[key]
            ]

    return result


def read_split_toml(
    tomlfile: Union[str, Path, IO[str], Mapping],
    path: Optional[Union[str, Path]] = None,
) -> MutableMapping:
    """Reads and consolidate TOML files.

    Our TOML accepts as input sections that are farmed off to other files:

        [some_section]
            include_path = "path/to/included.toml"

        [another_section]
            option_a = "a"

    The section `some_section` should contain only one item, `include_path`, giving the
    path to the toml file to include. This file is then spliced into the original toml.
    It **must** repeat the section that it splices. Hence if `included.toml` looks like:

        [some_section]
            some_option = "b"

            [some_section.inner_section]
                other_option = "c"

    Then the spliced toml would look like:

        [some_section]
            some_option = "b"

            [some_section.inner_section]
            other_option = "c"

        [another_section]
            option_a = "a"

    `included.toml` must contain a single section (possibly with inner options).
    Anything else will result in an error:

    This is an error:

        outer_option = "b"

        [some_section]
            some_option = "b"

    This is also an error:

        [some_section]
            some_option = "b"

        [some_other_section]
            some_other_option = "c"

    Arguments:
        tomlfile: path to the toml file. Can be any input to `toml.load`.
        path: Root path when formatting path options. See `format_paths`.
    """
    from toml import load

    def splice_section(settings: Mapping):
        settings = dict(**settings)

        for key, section in settings.items():
            if not isinstance(section, Mapping):
                continue

            if "include_path" in section and len(section) > 1:
                raise IncorrectSettings(
                    "Sections with an `include_path` option "
                    "should contain only that option."
                )
            elif "include_path" in section:
                inner = read_split_toml(section["include_path"], path=path)
                if key not in inner:
                    raise MissingSettings(
                        f"Could not find section {key} in {section['include_path']}"
                    )
                if len(inner) != 1:
                    raise IncorrectSettings(
                        "More than one section found in included"
                        f"file {section['include_path']}"
                    )
                settings[key] = inner[key]
            else:
                settings[key] = splice_section(section)

        return settings

    toml = tomlfile if isinstance(tomlfile, Mapping) else load(tomlfile)
    settings = format_paths(toml, path=path)  # type: ignore
    return splice_section(settings)


def read_settings(
    settings_file: Union[str, Path, IO[str], Mapping],
    path: Optional[Union[str, Path]] = None,
) -> Any:
    """Loads the input settings for any MUSE simulation.

    Loads a MUSE settings file. This must be a TOML formatted file. Missing settings are
    loaded from the DEFAULT_SETTINGS. Custom Python modules, if present, are loaded
    and checks are run to validate the settings and ensure that they are compatible with
    a MUSE simulation.

    Arguments:
        settings_file: A string or a Path to the settings file
        path: A string or path to the settings folder

    Returns:
        A dictionary with the settings
    """
    getLogger(__name__).info("Reading MUSE settings")

    # The user data
    if path is None and not isinstance(settings_file, (Mapping, IO)):
        path = Path(settings_file).parent
    elif path is None:
        path = Path()
    user_settings = read_split_toml(settings_file, path=path)

    # User defined default settings
    default_path = Path(user_settings.get("default_settings", DEFAULT_SETTINGS_PATH))

    if not default_path.is_absolute():
        default_path = path / default_path

    default_settings = read_split_toml(default_path, path=path)

    # Check that there is at least 1 sector.
    msg = "ERROR - There must be at least 1 sector."
    assert len(user_settings["sectors"]) >= 1, msg

    # timeslice information cannot be merged. Accept only information from one.
    if "timeslices" in user_settings:
        default_settings.pop("timeslices", None)

    # We update the default information with the user provided data
    settings = add_known_parameters(default_settings, user_settings)
    settings = add_unknown_parameters(settings, user_settings)

    # Finally, we run some checks to make sure all makes sense and files exist.
    validate_settings(settings)

    return convert(settings)


def read_ts_multiindex(
    settings: Optional[Union[Mapping, str]] = None,
    timeslice: Optional[xr.DataArray] = None,
    transforms: Optional[dict[tuple, np.ndarray]] = None,
) -> pd.MultiIndex:
    '''Read multiindex for a timeslice from TOML.

    Example:
        The timeslices are read from ``timeslice_levels``. The levels (keyword) and
        slice (list of values) correspond to the level, slices and slice aggregates
        defined  in the the ``timeslices`` section.

        >>> toml = """
        ...     ["timeslices"]
        ...     winter.weekday.day = 5
        ...     winter.weekday.night = 5
        ...     winter.weekend.day = 2
        ...     winter.weekend.night = 2
        ...     winter.weekend.dusk = 1
        ...     summer.weekday.day = 5
        ...     summer.weekday.night = 5
        ...     summer.weekend.day = 2
        ...     summer.weekend.night = 2
        ...     summer.weekend.dusk = 1
        ...     level_names = ["semester", "week", "day"]
        ...     aggregates.allday = ["day", "night"]
        ...     [timeslice_levels]
        ...     day = ["dusk", "allday"]
        ... """
        >>> from muse.timeslices import (
        ...     reference_timeslice,  aggregate_transforms
        ... )
        >>> from muse.readers.toml import read_ts_multiindex
        >>> ref = reference_timeslice(toml)
        >>> transforms = aggregate_transforms(toml, ref)
        >>> read_ts_multiindex(toml, ref, transforms)
        MultiIndex([('summer', 'weekday', 'allday'),
                    ('summer', 'weekend',   'dusk'),
                    ('summer', 'weekend', 'allday'),
                    ('winter', 'weekday', 'allday'),
                    ('winter', 'weekend',   'dusk'),
                    ('winter', 'weekend', 'allday')],
                   names=['semester', 'week', 'day'])

        It is an error to refer to a level or a slice that does not exist:

        >>> read_ts_multiindex(dict(days=["dusk", "allday"]), ref, transforms)
        Traceback (most recent call last):
        ...
        muse.readers.toml.IncorrectSettings: Unexpected level name(s): ...
        >>> read_ts_multiindex(dict(day=["usk", "allday"]), ref, transforms)
        Traceback (most recent call last):
        ...
        muse.readers.toml.IncorrectSettings: Unexpected slice(s): ...
    '''
    from itertools import product

    from toml import loads

    from muse.timeslices import TIMESLICE, TRANSFORMS

    indices = (TIMESLICE if timeslice is None else timeslice).get_index("timeslice")
    if transforms is None:
        transforms = TRANSFORMS
    if isinstance(settings, str):
        settings = loads(settings)
    elif settings is None:
        return indices
    elif not isinstance(settings, Mapping):
        settings = undo_damage(settings)
    settings = settings.get("timeslice_levels", settings)
    assert isinstance(settings, Mapping)
    if not set(settings).issubset(indices.names):
        msg = "Unexpected level name(s): " + ", ".join(
            set(settings).difference(indices.names)
        )
        raise IncorrectSettings(msg)
    levels = [
        settings.get(name, level) for name, level in zip(indices.names, indices.levels)
    ]
    levels = [[level] if isinstance(level, str) else level for level in levels]
    for i, level in enumerate(levels):
        known = [index[i] for index in transforms if len(index) > i]
        unexpected = set(level).difference(known)
        if unexpected:
            raise IncorrectSettings("Unexpected slice(s): " + ", ".join(unexpected))
    return pd.MultiIndex.from_tuples(
        [index for index in product(*levels) if index in transforms],
        names=indices.names,
    )


def read_timeslices(
    settings: Optional[Union[str, Mapping]] = None,
    timeslice: Optional[xr.DataArray] = None,
    transforms: Optional[dict[tuple, np.ndarray]] = None,
) -> xr.Dataset:
    '''Reads timeslice levels and create resulting timeslice coordinate.

    Args:
        settings: TOML dictionary. It should contain a ``timeslice_levels`` section.
            Otherwise, the timeslices will default to the global (finest) timeslices.
        timeslice: Finest timeslices. Defaults to the global in
            :py:mod:`~muse.timeslices`. If using the default, then this function
            should be called *after* the timeslice module has been setup with a call to
            :py:func:`~muse.timeslice.setup_module`.
        transforms: Transforms from desired timeslices to the finest timeslice. Defaults
            to the global in :py:mod:`~muse.timeslices`. If using the default,
            then this function should be called *after* the timeslice module has been
            setup with a call to :py:func:`~muse.timeslice.setup_module`.

    Returns:
        A xr.Dataset with the timeslice coordinates.

    Example:
        >>> toml = """
        ...     ["timeslices"]
        ...     winter.weekday.day = 5
        ...     winter.weekday.night = 5
        ...     winter.weekend.day = 2
        ...     winter.weekend.night = 2
        ...     winter.weekend.dusk = 1
        ...     summer.weekday.day = 5
        ...     summer.weekday.night = 5
        ...     summer.weekend.day = 2
        ...     summer.weekend.night = 2
        ...     summer.weekend.dusk = 1
        ...     level_names = ["semester", "week", "day"]
        ...     aggregates.allday = ["day", "night"]
        ...     [timeslice_levels]
        ...     day = ["dusk", "allday"]
        ... """
        >>> from muse.timeslices import (
        ...     reference_timeslice,  aggregate_transforms
        ... )
        >>> from muse.readers.toml import read_timeslices
        >>> ref = reference_timeslice(toml)
        >>> transforms = aggregate_transforms(toml, ref)
        >>> ts = read_timeslices(toml, ref, transforms)
        >>> assert "semester" in ts.coords
        >>> assert "week" in ts.coords
        >>> assert "day" in ts.coords
        >>> assert "represent_hours" in ts.coords
        >>> assert set(ts.coords["day"].data) == {"dusk", "allday"}
        >>> assert set(ts.coords["week"].data) == {"weekday", "weekend"}
        >>> assert set(ts.coords["semester"].data) == {"summer", "winter"}
    '''
    from muse.timeslices import TIMESLICE, timeslice_projector

    if timeslice is None:
        timeslice = TIMESLICE
    if settings is None:
        return xr.Dataset({"represent_hours": timeslice}).set_coords("represent_hours")
    indices = read_ts_multiindex(settings, timeslice=timeslice, transforms=transforms)
    units = xr.DataArray(
        np.ones(len(indices)), coords={"timeslice": indices}, dims="timeslice"
    )
    proj = timeslice_projector(units, finest=timeslice, transforms=transforms)
    proj *= xr.DataArray(
        timeslice.values,
        coords={"finest_timeslice": proj.finest_timeslice},
        dims="finest_timeslice",
    )

    return xr.Dataset({"represent_hours": proj.sum("finest_timeslice")}).set_coords(
        "represent_hours"
    )


def add_known_parameters(dd, u, parent=None):
    """Function for updating the settings dictionary recursively.

    Those variables that take default values are logged.
    """
    defaults_used = []
    missing = []
    d = deepcopy(dd)

    for k in dd:
        # Known parameters with user-defined values
        if k in u:
            v = u[k]
            if isinstance(v, Mapping):
                new_parent = k
                if parent is not None:
                    new_parent = f"{parent}.{k}"
                d[k] = add_known_parameters(d.get(k, {}), v, new_parent)
            else:
                d[k] = v
        # Required parameters
        elif isinstance(d[k], str) and d[k].lower() == "required":
            missing.append(k)
        # Optional parameters with default values
        elif isinstance(d[k], str) and d[k].lower() == "optional":
            d.pop(k)
        elif parent is not None:
            defaults_used.append(f"{parent}.{k}")
        else:
            defaults_used.append(k)

    msg = f"ERROR - Required parameters missing in input file: {missing}."
    if len(missing) > 0:
        raise MissingSettings(msg)

    msg = ", ".join(defaults_used)
    msg = " Default input values used: " + msg

    if len(defaults_used) > 0:
        getLogger(__name__).info(msg)

    return d


def add_unknown_parameters(dd, u):
    """Function for adding new parameters not known in the defaults file."""
    d = deepcopy(dd)
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = add_unknown_parameters(d.get(k, {}), v)
        else:
            d[k] = v

    return d


def validate_settings(settings: dict) -> None:
    """Run the checks on the settings file."""
    msg = " Validating input settings..."
    getLogger(__name__).info(msg)

    check_plugins(settings)

    for check in SETTINGS_CHECKS:
        SETTINGS_CHECKS[check](settings)


def check_plugins(settings: dict) -> None:
    """Checks that the user custom defined python files exist.

    Checks that the user custom defined python files exist. If flagged to use, they are
    also loaded.

    While this is a settings check, it is run separately to ensure that custom defined
    settings checks are all loaded before validating the settings.
    """
    plugins = settings.get("plugins", [])

    if isinstance(plugins, (dict, Mapping)):
        plugins = plugins.get("plugins")

    if isinstance(plugins, (Path, str)):
        plugins = [plugins]

    if not plugins:
        return

    for path in map(lambda x: Path(format_path(x)), plugins):
        if not path.exists():
            msg = f"ERROR plugin does not exist: {path}"
            getLogger(__name__).critical(msg)
            raise IncorrectSettings(msg)

        # The module is loaded, registering anything inside that is decorated
        spec = implib.spec_from_file_location(path.stem, path)
        mod = implib.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        getLogger(__name__).info(f"Loaded plugin {path.stem} from {path}")


@register_settings_check(vary_name=False)
def check_log_level(settings: dict) -> None:
    """Check the log level required in the simulation."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    msg = "ERROR - Valid log levels are {}.".format(", ".join(valid_levels))
    assert settings["log_level"].upper() in valid_levels, msg

    settings["log_level"] = settings["log_level"].upper()


@register_settings_check(vary_name=False)
def check_interpolation_mode(settings: dict) -> None:
    """Just updates the interpolation mode to a bool.

    There's no check, actually.
    """
    settings["interpolation_mode"] = settings["interpolation_mode"].lower()

    valid_modes = ["off", "false", "linear", "active", "cubic"]
    msg = 'ERROR - Valid interpolation modes are "off", "linear" and "cubic"'
    assert settings["interpolation_mode"] in valid_modes, msg

    # And we normalize the interpolation mode
    # If there's no interpolation, we get the nearest value
    if settings["interpolation_mode"] in ["off", "false"]:
        settings["interpolation_mode"] = "nearest"
    elif settings["interpolation_mode"] in ["linear", "active"]:
        settings["interpolation_mode"] = "linear"


@register_settings_check(vary_name=False)
def check_budget_parameters(settings: dict) -> None:
    """Check the parameters that are required if carbon_budget > 0."""
    length = len(settings["carbon_budget_control"]["budget"])
    if length > 0:
        msg = "ERROR - budget_check must have the same length that time_framework"
        if isinstance(settings["time_framework"], list):
            assert length == len(settings["time_framework"]), msg
            coords = settings["time_framework"]
        else:
            assert length + 1 == len(settings["time_framework"]), msg
            coords = settings["time_framework"][:-1]

        # If Ok, we transform the list into an xr.DataArray
        settings["carbon_budget_control"]["budget"] = xr.DataArray(
            np.array(settings["carbon_budget_control"]["budget"]),
            dims="year",
            coords={"year": coords},
        )
    else:
        settings["carbon_budget_control"]["budget"] = xr.DataArray([])


@register_settings_check(vary_name=False)
def check_foresight(settings: dict) -> None:
    """Check that foresight is a multiple of the smaller time_framework difference.

    If so, we update the time framework adding the foresight year to the list and
    transforming it into an array
    """
    tfmin = np.diff(settings["time_framework"]).min()
    msg = "ERROR - foresight is not a multiple of the smaller time_framework difference"
    assert settings["foresight"] % tfmin == 0, msg

    settings["time_framework"].sort()

    # This adds to the years list a new year separated from the last one a “foresight”
    # number of years.
    settings["time_framework"].append(
        settings["time_framework"][-1] + settings["foresight"]
    )
    settings["time_framework"] = np.array(settings["time_framework"], dtype=int)


@register_settings_check(vary_name=False)
def check_iteration_control(settings: dict) -> None:
    """Checks the variables related to the control of the iterations.

    This includes whether equilibrium must be reached, the maximum number of iterations
    or the tolerance to consider convergence.
    """
    # Anything that is not "off" or False, means that equilibrium should be reached.
    if str(settings["equilibrium"]).lower() in ("false", "off"):
        settings["equilibrium"] = False

    else:
        settings["equilibrium"] = True

        msg = "ERROR - The number of iterations must be a positive number."
        assert settings["maximum_iterations"] > 0, msg
        settings["maximum_iterations"] = int(settings["maximum_iterations"])

        msg = "ERROR - The convergence tolerance must be a positive number."
        assert settings["tolerance"] > 0, msg


@register_settings_check(vary_name=False)
def check_time_slices(settings: dict) -> None:
    """Check the time slices.

    If there is no error, they are transformed into a xr.DataArray
    """
    from muse.timeslices import setup_module

    setup_module(settings)
    settings["timeslices"] = read_timeslices(
        settings.get("mca", settings).get("timeslice_levels", None)
    ).timeslice


@register_settings_check(vary_name=False)
def check_global_data_files(settings: dict) -> None:
    """Checks that the global user files exist."""
    user_data = settings["global_input_files"]

    if Path(user_data["path"]).is_absolute():
        basedir = Path(user_data["path"])
    else:
        basedir = settings["root"] / Path(user_data["path"])

    msg = f"ERROR Directory of global user files does not exist: {basedir}."
    assert basedir.exists(), msg

    # Update the path to the base directory
    user_data["path"] = basedir

    files = list(user_data.keys())
    files.remove("path")
    for m in files:
        if user_data[m] == "":
            user_data.pop(m)
            continue
        if Path(user_data[m]).is_absolute():
            f = Path(user_data[m])
        else:
            f = basedir / user_data[m]
        assert f.exists(), f"{m.title()} file does not exist ({f})"

        # The path is updated so it can be readily used
        user_data[m] = f


@register_settings_check(vary_name=False)
def check_sectors_files(settings: dict) -> None:
    """Checks that the sector files exist."""
    sectors = settings["sectors"]
    priorities = {
        "preset": 0,
        "presets": 0,
        "demand": 10,
        "conversion": 20,
        "supply": 30,
        "last": 100,
    }

    if "list" in sectors:
        sectors = {k: sectors[k] for k in sectors["list"]}

    for name, sector in sectors.items():
        # Finally the priority of the sectors is used to set the order of execution
        sector["priority"] = sector.get("priority", priorities["last"])
        sector["priority"] = int(
            priorities.get(str(sector["priority"]).lower().strip(), sector["priority"])
        )

    sectors["list"] = sorted(
        settings["sectors"].keys(), key=lambda x: settings["sectors"][x]["priority"]
    )
    settings["sectors"] = sectors


def read_technodata(
    settings: Any,
    sector_name: Optional[str] = None,
    time_framework: Optional[Sequence[int]] = None,
    commodities: Optional[Union[str, Path]] = None,
    regions: Optional[Sequence[str]] = None,
    **kwargs,
) -> xr.Dataset:
    """Helper function to create technodata for a given sector."""
    from muse.readers.csv import read_technologies, read_trade

    if time_framework is None:
        time_framework = getattr(settings, "time_framework", [2010, 2050])

    if commodities is None:
        commodities = settings.global_input_files.global_commodities

    if regions is None:
        regions = settings.regions

    if sector_name is not None:
        settings = getattr(settings.sectors, sector_name)

    technodata_timeslices = getattr(settings, "technodata_timeslices", None)
    # normalizes case where technodata is not in own subsection
    if not hasattr(settings, "technodata") and sector_name is not None:
        raise MissingSettings(f"Missing technodata section in {sector_name}")
    elif not hasattr(settings, "technodata"):
        raise MissingSettings("Missing technodata section")
    technosettings = undo_damage(settings.technodata)

    if isinstance(technosettings, str):
        technosettings = dict(
            technodata=technosettings,
            technodata_timeslices=technodata_timeslices,
            commodities_in=settings.commodities_in,
            commodities_out=settings.commodities_out,
        )
    else:
        for comm in ("in", "out"):
            name = f"commodities_{comm}"
            if hasattr(settings, comm) and comm in technosettings:
                raise IncorrectSettings(f"{name} specified twice")
            elif hasattr(settings, comm):
                technosettings[name] = getattr(settings, name)

    for name in ("technodata", "commodities_in", "commodities_out"):
        if name not in technosettings:
            raise MissingSettings(f"Missing required technodata input {name}")
        filename = technosettings[name]
        if not Path(filename).exists():
            raise IncorrectSettings(f"File {filename} does not exist.")
        if not Path(filename).is_file():
            raise IncorrectSettings(f"File {filename} is not a file.")

    technologies = read_technologies(
        technodata_path_or_sector=technosettings.pop("technodata"),
        technodata_timeslices_path=technosettings.pop("technodata_timeslices", None),
        comm_out_path=technosettings.pop("commodities_out"),
        comm_in_path=technosettings.pop("commodities_in"),
        commodities=commodities,
    ).sel(region=regions)

    ins = (technologies.fixed_inputs > 0).any(("year", "region", "technology"))
    outs = (technologies.fixed_outputs > 0).any(("year", "region", "technology"))
    techcomms = technologies.commodity[ins | outs]
    technologies = technologies.sel(commodity=techcomms)
    for name, value in technosettings.items():
        if isinstance(name, (str, Path)):
            data = read_trade(value, drop="Unit")
            if "region" in data.dims:
                data = data.sel(region=regions)
            if "dst_region" in data.dims:
                data = data.sel(dst_region=regions)
                if data.dst_region.size == 1:
                    data = data.squeeze("dst_region", drop=True)

        else:
            data = value
        if isinstance(data, xr.Dataset):
            technologies = technologies.merge(data)
        else:
            technologies[name] = data

    # make sure technologies includes the requisite years
    maxyear = getattr(settings, "forecast", 5) + max(time_framework)
    if technologies.year.max() < maxyear:
        msg = "Forward-filling technodata to fit simulation timeframe"
        getLogger(__name__).info(msg)
        years = [*technologies.year.data.tolist(), maxyear]
        technologies = technologies.sel(year=years, method="ffill")
        technologies["year"] = "year", years
    minyear = min(time_framework)
    if technologies.year.min() > minyear:
        msg = "Back-filling technodata to fit simulation timeframe"
        getLogger(__name__).info(msg)
        years = [minyear, *technologies.year.data.tolist()]
        technologies = technologies.sel(year=years, method="bfill")
        technologies["year"] = "year", years

    year = sorted(set(time_framework).union(technologies.year.data.tolist()))
    technologies = technologies.interp(year=year, **kwargs)
    return technologies
