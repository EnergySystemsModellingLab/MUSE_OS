"""Example models and datasets.

Helps create and run small standard models from the command-line or directly from
python.

To run from the command-line:

.. code-block:: Bash

    python -m muse --model default


Other models may be available. Check the command-line help:

.. code-block:: Bash

    python -m muse --help

The same models can be instantiated in a python script as follows:

.. code-block:: Python

    from muse import example
    model = example.model("default")
    model.run()
"""

from __future__ import annotations

from logging import getLogger
from os import mkdir
from pathlib import Path
from shutil import copyfile, copytree, rmtree
from typing import cast

import numpy as np
import xarray as xr

from muse.mca import MCA
from muse.sectors import AbstractSector
from muse.timeslices import drop_timeslice

__all__ = ["model", "technodata"]

AVAILABLE_EXAMPLES = [
    "default",
    "default_retro",
    "default_adhoc",
    "default_timeslice",
    "default_correlation",
    "medium",
    "multiple_agents",
    "minimum_service",
    "trade",
]


def example_data_dir() -> Path:
    """Gets the examples folder."""
    import muse

    return Path(muse.__file__).parent / "data" / "example"


def model(name: str = "default", test: bool = False) -> MCA:
    """Fully constructs a given example model.

    File logging is added if ``test`` is False.

    Args:
        name: Name of the model to load.
        test: If True, the logging to file is not added.

    Returns:
        The MCA model.
    """
    from tempfile import TemporaryDirectory

    from muse import add_file_logger
    from muse.readers.toml import read_settings

    # we could modify the settings directly, but instead we use the copy_model function.
    # That way, there is only one function to get a model.
    with TemporaryDirectory() as tmpdir:
        path = copy_model(name, tmpdir)
        settings = read_settings(path / "settings.toml")
        getLogger("muse").setLevel(settings.log_level)
        if not test:
            add_file_logger()
        return MCA.factory(settings)


def copy_model(
    name: str = "default",
    path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy model files to given path.

    The model ends up in a "model" subfolder of the given path, or of the current
    working directory if no path is given. The subfolder must not exist, unless
    permission to ``overwrite`` is explicitly given. If the directory does exist and
    permission to ``overwrite`` is given, then all files inside the directory are
    deleted.
    """
    if name.lower() not in AVAILABLE_EXAMPLES:
        raise ValueError(f"Unknown model {name}")

    path = Path() if path is None else Path(path)

    if path.exists() and not path.is_dir():
        raise OSError(f"{path} exists and is not a directory")

    path /= "model"
    if path.exists():
        if not path.is_dir():
            raise OSError(f"{path} exists and is not a directory")
        elif not overwrite:
            raise OSError(f"{path} exists and ``overwrite`` is not allowed")
        rmtree(path)

    if name.lower() == "default":
        _copy_default(path)
    elif name.lower() == "default_retro":
        _copy_default_retro(path)
    elif name.lower() == "default_adhoc":
        _copy_default_adhoc(path)
    elif name.lower() == "default_timeslice":
        _copy_default_timeslice(path)
    elif name.lower() == "default_correlation":
        _copy_default_correlation(path)
    elif name.lower() == "medium":
        _copy_medium(path)
    elif name.lower() == "multiple_agents":
        _copy_multiple_agents(path)
    elif name.lower() == "minimum_service":
        _copy_minimum_service(path)
    elif name.lower() == "trade":
        _copy_trade(path)
    return path


def technodata(sector: str, model: str = "default") -> xr.Dataset:
    """Technology for a sector of a given example model."""
    from tempfile import TemporaryDirectory

    from muse.readers.toml import read_settings, read_technodata

    sector = sector.lower()
    allowed = {"residential", "power", "gas", "preset"}
    if sector == "preset":
        raise RuntimeError("The preset sector has no technodata.")
    if sector not in allowed:
        raise RuntimeError(f"This model only knows about sectors {allowed}.")
    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml")
        return read_technodata(settings, sector)


def search_space(sector: str, model: str = "default") -> xr.DataArray:
    """Determines which technology is considered for which asset.

    Used in constraints or during investment.
    """
    if model == "trade" and sector != "residential":
        return _trade_search_space(sector, model)
    return _nontrade_search_space(sector, model)


def sector(sector: str, model: str = "default") -> AbstractSector:
    """Loads a given sector from a given example model."""
    from tempfile import TemporaryDirectory

    from muse.readers.toml import read_settings
    from muse.sectors import SECTORS_REGISTERED

    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml")
        kind = getattr(settings.sectors, sector).type
        return SECTORS_REGISTERED[kind](sector, settings)


def available_sectors(model: str = "default") -> list[str]:
    """Sectors in this particular model."""
    from tempfile import TemporaryDirectory

    from muse.readers.toml import read_settings, undo_damage

    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml").sectors
        return [u for u in undo_damage(settings).keys() if u != "list"]


def mca_market(model: str = "default") -> xr.Dataset:
    """Initial market as seen by the MCA."""
    from tempfile import TemporaryDirectory

    from xarray import zeros_like

    from muse.readers.csv import read_initial_market
    from muse.readers.toml import read_settings

    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml")

        market = (
            read_initial_market(
                projections_path=settings.global_input_files.projections,
                base_year_export_path=getattr(
                    settings.global_input_files, "base_year_export", None
                ),
                base_year_import_path=getattr(
                    settings.global_input_files, "base_year_import", None
                ),
            )
            .sel(region=settings.regions)
            .interp(year=settings.time_framework, method=settings.interpolation_mode)
        )
        market["supply"] = drop_timeslice(zeros_like(market.exports))
        market["consumption"] = drop_timeslice(zeros_like(market.exports))

        return cast(xr.Dataset, market)


def residential_market(model: str = "default") -> xr.Dataset:
    """Initial market as seen by the residential sector."""
    from muse.mca import single_year_iteration

    market = mca_market(model)
    sectors = [sector("residential_presets", model=model)]
    return single_year_iteration(market, sectors)[0][
        ["prices", "supply", "consumption"]
    ]


def random_agent_assets(rng: np.random.Generator):
    """Creates random set of assets for testing and debugging."""
    nassets = rng.integers(low=1, high=6)
    nyears = rng.integers(low=2, high=5)
    years = rng.choice(list(range(2030, 2051)), size=nyears, replace=False)
    installed = rng.choice([2030, 2030, 2025, 2010], size=nassets)
    technologies = rng.choice(["stove", "thermomix", "oven"], size=nassets)
    capacity = rng.integers(101, size=(nassets, nyears))
    result = xr.Dataset()
    result["capacity"] = xr.DataArray(
        capacity.astype("int64"),
        coords=dict(
            installed=("asset", installed.astype("int64")),
            technology=("asset", technologies),
            region=rng.choice(["USA", "EU18", "Brexitham"]),
            year=sorted(years.astype("int64")),
        ),
        dims=("asset", "year"),
    )
    return result


def matching_market(sector: str, model: str = "default") -> xr.Dataset:
    """Market with a demand matching the maximum production from a sector."""
    from muse.examples import sector as load_sector
    from muse.quantities import consumption, maximum_production
    from muse.sectors import Sector
    from muse.utilities import agent_concatenation, broadcast_over_assets

    loaded_sector = cast(Sector, load_sector(sector, model))
    assets = agent_concatenation({u.uuid: u.assets for u in list(loaded_sector.agents)})

    market = xr.Dataset()
    techs = broadcast_over_assets(loaded_sector.technologies, assets.capacity)
    production = maximum_production(techs, assets.capacity)
    market["supply"] = production.sum("asset")
    if "dst_region" in market.dims:
        market = market.rename(dst_region="region")
    if market.region.dims:
        consump = consumption(techs, production)
        market["consumption"] = drop_timeslice(
            consump.groupby("region").sum(
                {"asset", "dst_region"}.intersection(consump.dims)
            )
            + market.supply
        )
    else:
        market["consumption"] = (
            consumption(techs, production).sum(
                {"asset", "dst_region"}.intersection(market.dims)
            )
            + market.supply
        )
    market["prices"] = market.supply.dims, np.random.random(market.supply.shape)
    return market


def _copy_default(path: Path):
    copytree(example_data_dir() / "default", path)


def _copy_default_retro(path: Path):
    copytree(example_data_dir() / "default_retro", path)


def _copy_default_adhoc(path: Path):
    from muse.wizard import modify_toml

    copytree(example_data_dir() / "default", path)

    def update_lpsolver(data):
        for sector in data.get("sectors", {}):
            subsectors = data["sectors"][sector].get("subsectors", {})
            for sub in subsectors:
                subsectors[sub]["lpsolver"] = "adhoc"

    modify_toml(path / "settings.toml", update_lpsolver)


def _copy_default_timeslice(path: Path):
    copytree(example_data_dir() / "default_timeslice", path)


def _copy_default_correlation(path: Path):
    from muse.wizard import modify_toml

    # Start with default model
    copytree(example_data_dir() / "default", path)

    # Delete presets
    rmtree(path / "residential_presets")

    # Copy regression files
    mkdir(path / "residential_presets")
    for file in [
        "Macrodrivers.csv",
        "regressionparameters.csv",
        "TimesliceSharepreset.csv",
    ]:
        copyfile(
            example_data_dir() / "default_correlation" / file,
            path / "residential_presets" / file,
        )

    # Modify toml file to point to new presets
    settings_file = path / "settings.toml"
    path_prefix = "{path}/residential_presets/"
    modify_toml(
        settings_file,
        lambda settings: (
            settings["sectors"]["residential_presets"].update(
                {
                    "timeslice_shares_path": path_prefix + "TimesliceSharepreset.csv",
                    "macrodrivers_path": path_prefix + "Macrodrivers.csv",
                    "regression_path": path_prefix + "regressionparameters.csv",
                }
            ),
            settings["sectors"]["residential_presets"].pop("consumption_path"),
        ),
    )


def _copy_multiple_agents(path: Path):
    from toml import dump, load

    # Start with default model
    copytree(example_data_dir() / "default", path)

    # Copy new agents file and modified technodata file
    copyfile(
        example_data_dir() / "multiple_agents" / "Agents.csv",
        path / "residential" / "Agents.csv",
    )
    copyfile(
        example_data_dir() / "multiple_agents" / "residential" / "Technodata.csv",
        path / "residential" / "Technodata.csv",
    )

    # Link to new agents file in settings.toml
    toml = load(path / "settings.toml")
    toml["sectors"]["residential"]["subsectors"]["all"]["agents"] = (
        "{path}/residential/Agents.csv"
    )
    with (path / "settings.toml").open("w") as fileobj:
        dump(toml, fileobj)


def _copy_medium(path: Path):
    copytree(example_data_dir() / "medium", path)
    copytree(example_data_dir() / "default" / "power", path / "power")
    copytree(example_data_dir() / "default" / "gas", path / "gas")
    copyfile(example_data_dir() / "default" / "Agents.csv", path / "Agents.csv")
    copyfile(example_data_dir() / "default" / "settings.toml", path / "settings.toml")


def _copy_minimum_service(path: Path):
    copytree(example_data_dir() / "minimum_service", path)


def _copy_trade(path: Path):
    copytree(example_data_dir() / "trade", path)


def _trade_search_space(sector: str, model: str = "default") -> xr.DataArray:
    from muse.agents import Agent
    from muse.examples import sector as load_sector
    from muse.sectors import Sector
    from muse.utilities import agent_concatenation

    loaded_sector = cast(Sector, load_sector(sector, model))

    market = matching_market(sector, model)
    return cast(
        xr.DataArray,
        agent_concatenation(
            {
                a.uuid: cast(Agent, a).search_rules(
                    agent=a,
                    demand=market.consumption.isel(year=0, drop=True),
                    technologies=loaded_sector.technologies.isel(year=0, drop=True),
                    market=market,
                )
                for a in loaded_sector.agents
            },
            dim="agent",
        ),
    )


def _nontrade_search_space(sector: str, model: str = "default") -> xr.DataArray:
    from numpy import ones

    technology = technodata(sector, model).technology
    return xr.DataArray(
        ones((len(technology), len(technology)), dtype=bool),
        coords=dict(asset=technology.values, replacement=technology.values),
        dims=("asset", "replacement"),
    )
