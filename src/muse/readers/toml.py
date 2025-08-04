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


@register_settings_hook(priority=0)
def standardise_case(settings: dict) -> None:
    """Standardise certain fields to snake_case."""
    from muse.utilities import camel_to_snake

    fields_to_standardise = ["excluded_commodities", "regions"]
    for field in fields_to_standardise:
        if field in settings:
            settings[field] = [camel_to_snake(x) for x in settings[field]]

    # Handle timeslice level_names if present
    if "level_names" in settings["timeslices"]:
        settings["timeslices"]["level_names"] = [
            camel_to_snake(x) for x in settings["timeslices"]["level_names"]
        ]


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
def setup_commodities(settings: dict) -> None:
    """Set up the commodities."""
    from muse.commodities import setup_module

    setup_module(settings["global_input_files"]["global_commodities"])


@register_settings_hook(priority=1)
def setup_time_framework(settings: dict) -> None:
    """Converts the time framework to a sorted array."""
    settings["time_framework"] = np.array(sorted(settings["time_framework"]), dtype=int)


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
def check_currency(settings: dict) -> None:
    """Raise a warning if no currency is specified."""
    if not settings.get("currency", None):
        msg = (
            "No currency specified. Please specify a currency in the settings file "
            "using the 'currency' parameter."
        )
        getLogger(__name__).warning(msg)
        return


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
                    f"in sector '{sector_name}'. Defaulting to 'scipy'"
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
    sector_name: str,
    interpolation_mode: str = "linear",
) -> xr.Dataset:
    """Read and process technodata for a given sector.

    This function reads technology data from CSV files and processes it for use in MUSE
    simulations. It handles technology specifications, trade data, and interpolates
    the data to match the simulation timeframe.

    Args:
        settings: MUSE settings object containing configuration parameters
        sector_name: Name of the sector to read technodata for
        interpolation_mode: Method for interpolating data between years.
            Defaults to "linear"

    Returns:
        xr.Dataset: Processed technodata containing technology specifications,
            inputs/outputs, and trade information
    """
    from muse.readers.csv import read_technologies, read_trade_technodata

    regions = settings.regions
    time_framework = settings.time_framework
    settings = getattr(settings.sectors, sector_name)

    # Legacy: technodata settings could be in a "technodata" section
    if isinstance(undo_damage(settings.technodata), Mapping):
        settings = settings.technodata

    # Read technodata
    technologies = read_technologies(
        technodata_path=Path(settings.technodata),
        technodata_timeslices_path=getattr(settings, "technodata_timeslices", None),
        comm_out_path=Path(settings.commodities_out),
        comm_in_path=Path(settings.commodities_in),
        time_framework=time_framework,
        interpolation_mode=interpolation_mode,
    ).sel(region=regions)

    # Only keep commodities that are used as inputs or outputs
    dims = ("year", "region", "technology")
    fixed_ins = (technologies.fixed_inputs > 0).any(
        [d for d in dims if d in technologies.fixed_inputs.dims]
    )
    flex_ins = (technologies.flexible_inputs > 0).any(
        [d for d in dims if d in technologies.flexible_inputs.dims]
    )
    outs = (technologies.fixed_outputs > 0).any(
        [d for d in dims if d in technologies.fixed_outputs.dims]
    )
    techcomms = technologies.commodity[fixed_ins | flex_ins | outs]
    technologies = technologies.sel(commodity=techcomms)

    # Read trade technodata
    if hasattr(settings, "trade"):
        trade_data = read_trade_technodata(settings.trade)
        if "region" in trade_data.dims:
            trade_data = trade_data.sel(region=regions)
        if "dst_region" in trade_data.dims:
            trade_data = trade_data.sel(dst_region=regions)
            if trade_data.dst_region.size == 1:
                trade_data = trade_data.squeeze("dst_region", drop=True)

        # Drop duplicate data vars before merging
        common_vars = set(technologies.data_vars) & set(trade_data.data_vars)
        technologies = technologies.drop_vars(common_vars)
        technologies = technologies.merge(trade_data)

    technologies = technologies.set_index(commodity="commodity")  # See PR #638
    return technologies


def read_presets_sector(settings: Any, sector_name: str) -> xr.Dataset:
    """Read data for a preset sector.

    This function reads consumption and supply data for a preset sector from various
    data sources. It supports multiple input formats including direct consumption data,
    demand tables, or correlation-based consumption calculated from macro drivers and
    regression parameters.

    Args:
        settings: MUSE settings object containing configuration parameters
        sector_name: Name of the preset sector to read data for

    Returns:
        xr.Dataset: Dataset containing consumption and supply data for the sector.
            Costs are initialized to zero.
    """
    from muse.readers import read_attribute_table, read_presets
    from muse.timeslices import distribute_timeslice, drop_timeslice

    sector_conf = getattr(settings.sectors, sector_name)

    # Read consumption data
    if getattr(sector_conf, "consumption_path", None) is not None:
        consumption = read_presets(sector_conf.consumption_path)
    elif getattr(sector_conf, "demand_path", None) is not None:
        consumption = read_attribute_table(sector_conf.demand_path)
        if "timeslice" not in consumption.dims:
            consumption = distribute_timeslice(consumption)
    elif (
        getattr(sector_conf, "macrodrivers_path", None) is not None
        and getattr(sector_conf, "regression_path", None) is not None
    ):
        consumption = read_correlation_consumption(sector_conf)
    else:
        raise MissingSettings(f"Missing consumption data for sector {sector_name}")

    # Create presets dataset
    presets = xr.Dataset(
        {
            "consumption": consumption,
            "supply": read_presets(sector_conf.supply_path)
            if getattr(sector_conf, "supply_path", None) is not None
            else drop_timeslice(xr.zeros_like(consumption)),
            "costs": drop_timeslice(xr.zeros_like(consumption)),
        }
    )

    return presets


def read_correlation_consumption(sector_conf: Any) -> xr.Dataset:
    """Read consumption data for a sector based on correlation files.

    This function calculates endogenous demand for a sector using macro drivers and
    regression parameters. It applies optional filters, handles sector aggregation,
    and distributes the consumption across timeslices if timeslice shares are provided.

    Args:
        sector_conf: Sector configuration object containing paths to macro drivers,
            regression parameters, and timeslice shares files

    Returns:
        xr.Dataset: Consumption data distributed across timeslices and regions
    """
    from muse.readers import (
        read_macro_drivers,
        read_regression_parameters,
        read_timeslice_shares,
    )
    from muse.regressions import endogenous_demand
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    macro_drivers = read_macro_drivers(sector_conf.macrodrivers_path)
    regression_parameters = read_regression_parameters(sector_conf.regression_path)
    consumption = endogenous_demand(
        drivers=macro_drivers,
        regression_parameters=regression_parameters,
        forecast=0,
    )

    # Legacy: apply filters
    if hasattr(sector_conf, "filters"):
        consumption = consumption.sel(sector_conf.filters._asdict())

    # Legacy: we permit regression parameters to split by sector, so have to sum
    if "sector" in consumption.dims:
        consumption = consumption.sum("sector")

    # Split by timeslice
    if sector_conf.timeslice_shares_path is not None:
        shares = read_timeslice_shares(sector_conf.timeslice_shares_path)
        consumption = broadcast_timeslice(consumption) * shares
    else:
        consumption = distribute_timeslice(consumption)

    return consumption
