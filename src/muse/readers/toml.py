"""Ensemble of functions to read MUSE data."""

from __future__ import annotations

__all__ = ["read_settings"]

import importlib.util as implib
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from muse.defaults import DATA_DIRECTORY

DEFAULT_SETTINGS_PATH = DATA_DIRECTORY / "default_settings.toml"
"""Default settings path."""

SETTINGS_HOOKS_SIGNATURE = Callable[[dict], None]
"""settings checks signature."""

SETTINGS_HOOKS: list[tuple[int, str, SETTINGS_HOOKS_SIGNATURE]] = []
"""Dictionary of settings checks."""


class InputError(Exception):
    """Root for TOML input errors."""


class MissingSettings(InputError):
    """Error when an input is missing."""


class IncorrectSettings(InputError):
    """Error when an input exists but is incorrect."""


def convert(dictionary: dict) -> namedtuple:
    """Converts a dictionary (with nested ones) to a nametuple."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert(value)
    return namedtuple("MUSEOptions", dictionary.keys())(**dictionary)


def undo_damage(nt) -> Any:
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
    replacements: Mapping | None = None,
    path: str | Path | None = None,
    cwd: str | Path | None = None,
) -> Path:
    """Replaces known patterns in a path.

    Unknown patterns are left alone. This allows downstream object factories to format
    the paths according to their own specifications.
    """
    from string import Formatter

    patterns = FormatDict(
        {
            **{
                "cwd": Path("" if cwd is None else cwd).absolute(),
                "path": Path("" if path is None else path).absolute(),
            },
            **({} if replacements is None else replacements),
        }
    )
    formatter = Formatter()
    return Path(formatter.vformat(str(filepath), (), patterns)).absolute()


def format_paths(
    settings: Mapping,
    path: Path,
    cwd: Path,
    suffixes: Sequence[str] = (".csv", ".nc", ".xls", ".xlsx", ".py", ".toml"),
) -> dict:
    """Format paths passed to settings.

    This function is used to format paths in the settings file. It is used to replace
    the {path} and {cwd} placeholders with the actual path and current working
    directory.

    Args:
        settings: The settings dictionary to format
        path: The path to the settings file
        cwd: The current working directory
        suffixes: Suffixes used to identify strings as paths
    """

    def is_a_path(key, value):
        return (
            isinstance(value, (str, Path)) and Path(value).suffix in suffixes
        ) or key == "filename"

    # Recursively format paths
    result = dict(**settings)
    for key, value in result.items():
        if is_a_path(key, value):
            result[key] = format_path(value, path=path, cwd=cwd)
        elif isinstance(value, Mapping):
            result[key] = format_paths(settings=value, path=path, cwd=cwd)
        elif isinstance(value, list):
            result[key] = [
                format_paths(settings=item, path=path, cwd=cwd)
                if isinstance(item, Mapping)
                else format_path(item, path=path, cwd=cwd)
                if is_a_path(key, item)
                else item
                for item in result[key]
            ]

    return result


def read_toml(tomlfile: str | Path, path: str | Path | None = None) -> MutableMapping:
    """Reads a TOML file and formats the paths.

    Args:
        tomlfile: Path to the TOML file (string or Path object)
        path: Optional path to use for formatting relative paths (string or Path object)

    Returns:
        MutableMapping containing the formatted TOML data
    """
    from toml import load

    tomlfile = Path(tomlfile)
    toml = load(tomlfile)
    if path is None:
        path = tomlfile.parent
    else:
        path = Path(path)
    settings = format_paths(toml, path=path, cwd=Path())
    return settings


def read_settings(settings_file: str | Path) -> namedtuple:
    """Loads the input settings for any MUSE simulation.

    Loads a MUSE settings file. This must be a TOML formatted file. Missing settings are
    loaded from the DEFAULT_SETTINGS. Custom Python modules, if present, are loaded
    and hooks are run to process and validate the settings and ensure that they are
    compatible with a MUSE simulation.

    Arguments:
        settings_file: A string or a Path to the settings file

    Returns:
        A dictionary with the settings
    """
    getLogger(__name__).info("Reading MUSE settings")
    settings_file = Path(settings_file)

    # The user data
    user_settings = read_toml(settings_file)

    # Get default settings
    default_path = Path(user_settings.get("default_settings", DEFAULT_SETTINGS_PATH))
    default_settings = read_toml(default_path, path=settings_file.parent)

    # Timeslice information cannot be merged. Accept only information from one.
    if "timeslices" in user_settings:
        default_settings.pop("timeslices", None)

    # We update the default information with the user provided data
    settings = add_known_parameters(default_settings, user_settings)
    settings = add_unknown_parameters(settings, user_settings)

    # Finally, we run some hooks to make sure all makes sense and files exist.
    process_settings(settings)
    return convert(settings)


def add_known_parameters(default_dict, user_dict, parent=None) -> dict:
    """Recursively merge user settings with default settings.

    Validates required parameters and handles optional ones.

    Args:
        default_dict: Dictionary containing default settings
        user_dict: Dictionary containing user-provided settings
        parent: Parent key for nested dictionaries (used for logging)

    Returns:
        Merged dictionary with validated settings
    """
    from logging import getLogger

    merged = deepcopy(default_dict)
    defaults_used = []
    missing = []

    for key in default_dict:
        if key in user_dict:
            value = user_dict[key]
            if isinstance(value, Mapping):
                new_parent = f"{parent}.{key}" if parent else key
                merged[key] = add_known_parameters(
                    merged.get(key, {}), value, new_parent
                )
            else:
                merged[key] = value
        elif isinstance(merged[key], str):
            if merged[key].lower() == "required":
                missing.append(key)
            elif merged[key].lower() == "optional":
                merged.pop(key)
        else:
            defaults_used.append(f"{parent}.{key}" if parent else key)

    if missing:
        raise MissingSettings(f"Required parameters missing in input file: {missing}")

    if defaults_used:
        getLogger(__name__).info(
            f"Default input values used: {', '.join(defaults_used)}"
        )

    return merged


def add_unknown_parameters(default_dict, user_dict) -> dict:
    """Recursively merge user settings with default settings.

    Preserves unknown parameters from user settings.

    Args:
        default_dict: Dictionary containing default settings
        user_dict: Dictionary containing user-provided settings

    Returns:
        Merged dictionary containing both default and user settings
    """
    merged = deepcopy(default_dict)

    for key, value in user_dict.items():
        if isinstance(value, Mapping):
            merged[key] = add_unknown_parameters(merged.get(key, {}), value)
        else:
            merged[key] = value

    return merged


def process_settings(settings: dict) -> None:
    """Run the hooks on the settings file."""
    msg = " Processing input settings..."
    getLogger(__name__).info(msg)

    # Load extra hooks from plugins
    check_plugins(settings)
    # This must be run before the other hooks to ensure that custom defined settings
    # hooks are all loaded before validating the settings.

    # Run hooks in order of priority
    for _, _, hook in sorted(SETTINGS_HOOKS, key=lambda x: x[0]):
        hook(settings)


def check_plugins(settings: dict) -> None:
    """Check and load user-defined Python plugin files if they exist."""
    plugins = settings.get("plugins", [])

    # Handle plugins as dict, str, or Path
    if isinstance(plugins, (dict, Mapping)):
        plugins = plugins.get("plugins", [])
    if isinstance(plugins, (Path, str)):
        plugins = [plugins]
    if not plugins:
        return

    for plugin in plugins:
        plugin_path = Path(format_path(plugin))
        if not plugin_path.exists():
            msg = f"ERROR plugin does not exist: {plugin_path}"
            getLogger(__name__).critical(msg)
            raise IncorrectSettings(msg)

        # Load the plugin module
        spec = implib.spec_from_file_location(plugin_path.stem, plugin_path)
        mod = implib.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        getLogger(__name__).info(f"Loaded plugin {plugin_path.stem} from {plugin_path}")


def register_settings_hook(
    func: SETTINGS_HOOKS_SIGNATURE | None = None, *, priority: int = 100
) -> Callable:
    """Register a function to be called during settings validation.

    The function will be called with the settings dictionary as its only argument.
    The function can modify the settings dictionary in place.

    Args:
        func: The function to register
        priority: The priority of the function. Lower numbers are called first.

    Returns:
        The decorated function
    """

    def decorated(f: SETTINGS_HOOKS_SIGNATURE) -> SETTINGS_HOOKS_SIGNATURE:
        """Register the function and return it unchanged."""
        getLogger(__name__).debug(
            f"Registering settings hook {f.__name__} with priority {priority}"
        )
        SETTINGS_HOOKS.append((priority, f.__name__, f))
        return f

    if func is None:
        return decorated
    return decorated(func)


@register_settings_hook(priority=1)
def check_sectors(settings: dict) -> None:
    """Check that there is at least 1 sector."""
    assert len(settings["sectors"]) >= 1, "ERROR - There must be at least 1 sector."


@register_settings_hook(priority=1)
def setup_timeslices(settings: dict) -> None:
    """Set up the timeslices."""
    from muse.timeslices import setup_module

    setup_module(settings)
    settings.pop("timeslices", None)


@register_settings_hook(priority=1)
def setup_time_framework(settings: dict) -> None:
    """Converts the time framework to a sorted array."""
    settings["time_framework"] = np.array(sorted(settings["time_framework"]), dtype=int)


@register_settings_hook(priority=1)
def standardise_case(settings: dict) -> None:
    """Standardise certain fields to snake_case."""
    from muse.readers import camel_to_snake

    fields_to_standardise = ["excluded_commodities"]
    for field in fields_to_standardise:
        if field in settings:
            settings[field] = [camel_to_snake(x) for x in settings[field]]


@register_settings_hook
def check_log_level(settings: dict) -> None:
    """Check the log level required in the simulation."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    msg = "ERROR - Valid log levels are {}.".format(", ".join(valid_levels))
    assert settings["log_level"].upper() in valid_levels, msg

    settings["log_level"] = settings["log_level"].upper()


@register_settings_hook
def check_interpolation_mode(settings: dict) -> None:
    """Checks that the interpolation mode is valid."""
    settings["interpolation_mode"] = settings["interpolation_mode"].lower()

    valid_modes = [
        "linear",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
        "active",  # legacy: see below
    ]
    msg = (
        'ERROR - Valid interpolation modes are "linear", "nearest", "zero", '
        '"slinear", "quadratic", "cubic", "previous", "next"'
    )
    assert settings["interpolation_mode"] in valid_modes, msg

    # Legacy: "Active" was previous default - switch to "linear" (#642)
    if settings["interpolation_mode"] == "active":
        msg = "'Active' interpolation mode is deprecated. Defaulting to 'linear'."
        getLogger(__name__).warning(msg)
        settings["interpolation_mode"] = "linear"


@register_settings_hook
def check_budget_parameters(settings: dict) -> None:
    """Check the parameters that are required if carbon_budget > 0."""
    budget = settings["carbon_budget_control"]["budget"]
    time_framework = settings["time_framework"]
    if not budget:
        settings["carbon_budget_control"]["budget"] = xr.DataArray([])
        return

    msg = "ERROR - budget_check must have the same length that time_framework"
    if len(budget) != len(time_framework):
        raise AssertionError(msg)

    coords = time_framework
    settings["carbon_budget_control"]["budget"] = xr.DataArray(
        np.array(budget), dims="year", coords={"year": coords}
    )


@register_settings_hook
def check_iteration_control(settings: dict) -> None:
    """Check and set iteration control parameters for equilibrium and convergence."""
    equilibrium = str(settings["equilibrium"]).lower()
    if equilibrium in ("false", "off"):
        settings["equilibrium"] = False
        return

    settings["equilibrium"] = True
    if settings["maximum_iterations"] <= 0:
        raise ValueError("ERROR - The number of iterations must be a positive number.")
    settings["maximum_iterations"] = int(settings["maximum_iterations"])
    if settings["tolerance"] <= 0:
        raise ValueError("ERROR - The convergence tolerance must be a positive number.")


@register_settings_hook
def sort_sectors(settings: dict) -> None:
    """Set the priorities of the sectors."""
    sectors = settings["sectors"]
    priorities = {
        "preset": 0,
        "presets": 0,
        "demand": 10,
        "conversion": 20,
        "supply": 30,
        "last": 100,
    }

    # If sectors has a 'list' key, flatten it
    if "list" in sectors:
        sectors = {k: sectors[k] for k in sectors["list"]}

    for sector in sectors.values():
        # Assign priority, using default if not present or not recognized
        prio = sector.get("priority", priorities["last"])
        sector["priority"] = int(priorities.get(str(prio).lower().strip(), prio))

    # Sort sector names by priority
    sectors["list"] = sorted(sectors.keys(), key=lambda x: sectors[x]["priority"])
    settings["sectors"] = sectors


@register_settings_hook
def check_deprecated_params(settings: dict) -> None:
    """Check for and warn about deprecated parameters."""
    deprecated_params = ["foresight", "interest_rate"]
    for param in deprecated_params:
        if param in settings:
            msg = (
                f"The `{param}` parameter has been deprecated. "
                "Please remove it from your settings file."
            )
            getLogger(__name__).warning(msg)
            settings.pop(param)


@register_settings_hook(priority=10)
def check_subsector_settings(settings: dict) -> None:
    """Check for invalid or deprecated subsector settings.

    Validates:
    - Renamed asset_threshhold parameter (PR #447)
    - Missing lpsolver parameter (PR #587)
    - Deprecated forecast parameter (PR #645)
    """
    from logging import getLogger

    # Check each sector's subsectors
    for sector_name, sector in settings["sectors"].items():
        if "subsectors" not in sector:
            continue

        for subsector_name, subsector in sector["subsectors"].items():
            # Check for renamed asset_threshhold parameter
            if "asset_threshhold" in subsector:
                msg = (
                    "Invalid parameter asset_threshhold. Did you mean asset_threshold?"
                )
                raise ValueError(msg)

            # Check for missing lpsolver
            if "lpsolver" not in subsector:
                msg = (
                    f"lpsolver not specified for subsector '{subsector_name}' "
                    "in sector '{sector_name}'. Defaulting to 'scipy'"
                )
                getLogger(__name__).warning(msg)

            # Check for deprecated forecast parameter
            if "forecast" in subsector:
                msg = (
                    "The 'forecast' parameter has been deprecated. "
                    "Please remove from your settings file."
                )
                getLogger(__name__).warning(msg)


def read_technodata(
    settings: Any,
    sector_name: str | None = None,
    time_framework: Sequence[int] | None = None,
    commodities: str | Path | None = None,
    regions: Sequence[str] | None = None,
    interpolation_mode: str = "linear",
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

    if isinstance(technosettings, (str, Path)):
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
    maxyear = max(time_framework)
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
    technologies = technologies.interp(year=year, method=interpolation_mode)
    technologies = technologies.set_index(commodity="commodity")  # See PR #638
    return technologies


def read_presets_sector(settings: Any, sector_name: str) -> xr.Dataset:
    """Read data for a preset sector."""
    from muse.commodities import CommodityUsage
    from muse.readers import (
        read_attribute_table,
        read_macro_drivers,
        read_presets,
        read_regression_parameters,
        read_timeslice_shares,
    )
    from muse.regressions import endogenous_demand
    from muse.timeslices import (
        TIMESLICE,
        broadcast_timeslice,
        distribute_timeslice,
        drop_timeslice,
    )

    sector_conf = getattr(settings.sectors, sector_name)
    presets = xr.Dataset()

    timeslice = TIMESLICE.timeslice
    if getattr(sector_conf, "consumption_path", None) is not None:
        consumption = read_presets(sector_conf.consumption_path)
        presets["consumption"] = consumption.assign_coords(timeslice=timeslice)
    elif getattr(sector_conf, "demand_path", None) is not None:
        presets["consumption"] = read_attribute_table(sector_conf.demand_path)
    elif (
        getattr(sector_conf, "macrodrivers_path", None) is not None
        and getattr(sector_conf, "regression_path", None) is not None
    ):
        macro_drivers = read_macro_drivers(
            getattr(sector_conf, "macrodrivers_path", None)
        )
        regression_parameters = read_regression_parameters(
            getattr(sector_conf, "regression_path", None)
        )
        forecast = getattr(sector_conf, "forecast", 0)
        if isinstance(forecast, Sequence):
            forecast = xr.DataArray(
                forecast, coords={"forecast": forecast}, dims="forecast"
            )
        consumption = endogenous_demand(
            drivers=macro_drivers,
            regression_parameters=regression_parameters,
            forecast=forecast,
        )
        if hasattr(sector_conf, "filters"):
            consumption = consumption.sel(sector_conf.filters._asdict())
        if "sector" in consumption.dims:
            consumption = consumption.sum("sector")

        if getattr(sector_conf, "timeslice_shares_path", None) is not None:
            assert isinstance(timeslice, xr.DataArray)
            shares = read_timeslice_shares(sector_conf.timeslice_shares_path)
            shares = shares.assign_coords(timeslice=timeslice)
            assert consumption.commodity.isin(shares.commodity).all()
            assert consumption.region.isin(shares.region).all()
            consumption = broadcast_timeslice(consumption) * shares.sel(
                region=consumption.region, commodity=consumption.commodity
            )
        presets["consumption"] = consumption

    if getattr(sector_conf, "supply_path", None) is not None:
        supply = read_presets(sector_conf.supply_path)
        supply.coords["timeslice"] = presets.timeslice
        presets["supply"] = supply

    if getattr(sector_conf, "costs_path", None) is not None:
        presets["costs"] = read_attribute_table(sector_conf.costs_path)
    elif getattr(sector_conf, "lcoe_path", None) is not None and "supply" in presets:
        costs = (
            read_presets(
                sector_conf.lcoe_path,
                indices=("RegionName",),
                columns="timeslices",
            )
            * presets["supply"]
        )
        presets["costs"] = costs

    if len(presets.data_vars) == 0:
        raise OSError("None of supply, consumption, costs given")

    # add missing data as zeros: we only need one of consumption, costs, supply
    components = {"supply", "consumption", "costs"}
    for component in components:
        others = components.intersection(presets.data_vars).difference({component})
        if component not in presets and len(others) > 0:
            presets[component] = drop_timeslice(xr.zeros_like(presets[others.pop()]))

    # add timeslice, if missing
    for component in {"supply", "consumption"}:
        if "timeslice" not in presets[component].dims:
            presets[component] = distribute_timeslice(presets[component])

    comm_usage = (presets.costs > 0).any(set(presets.costs.dims) - {"commodity"})
    presets["comm_usage"] = (
        "commodity",
        [CommodityUsage.PRODUCT if u else CommodityUsage.OTHER for u in comm_usage],
    )
    presets = presets.set_coords("comm_usage")
    return presets
