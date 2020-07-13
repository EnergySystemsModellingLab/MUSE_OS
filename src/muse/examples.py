"""Example models and datasets.

Helps create and run small standard models from the command-line or directly from
python.

To run from the command-line:

.. code-block:: Bash

    python -m muse --model default


Other models may be available. Check the command-line help:

.. code-block:: Bash

    python -m muse --help

The same models can be instanciated in a python script as follows:

.. code-block:: Python

    from muse import example
    model = example.model("default")
    model.run()
"""
from pathlib import Path
from typing import List, Optional, Text, Union, cast

import numpy as np
import xarray as xr

from muse.mca import MCA
from muse.sectors import AbstractSector

__all__ = ["model", "technodata"]


def example_data_dir() -> Path:
    import muse

    return Path(muse.__file__).parent / "data" / "example"


def model(name: Text = "default") -> MCA:
    """Simple model with Residential, Power, and Gas sectors."""
    from tempfile import TemporaryDirectory
    from muse.readers.toml import read_settings

    # we could modify the settings directly, but instead we use the copy_model function.
    # That way, there is only one function to get a model.
    with TemporaryDirectory() as tmpdir:

        path = copy_model(name, tmpdir)
        return MCA.factory(read_settings(path / "settings.toml"))


def copy_model(
    name: Text = "default",
    path: Optional[Union[Text, Path]] = None,
    overwrite: bool = False,
) -> Path:
    """Copy model files to given path.

    The model ends up in a "model" subfolder of the given path, or of the current
    working directory if no path is given. The subfolder must not exist, unless
    permission to ``overwrite`` is explicitly given. If the directory does exist and
    permission to ``overwrite`` is given, then all files inside the directory are
    deleted.
    """
    from shutil import rmtree

    if name.lower() not in {"default", "multiple-agents", "medium", "minimum-service"}:
        raise ValueError(f"Unknown model {name}")

    path = Path() if path is None else Path(path)

    if path.exists() and not path.is_dir():
        raise IOError(f"{path} exists and is not a directory")

    path /= "model"
    if path.exists():
        if not path.is_dir():
            raise IOError(f"{path} exists and is not a directory")
        elif not overwrite:
            raise IOError(f"{path} exists and ``overwrite`` is not allowed")
        rmtree(path)

    if name.lower() == "default":
        _copy_default(path)
    elif name.lower() == "medium":
        _copy_medium(path)
    elif name.lower() == "multiple-agents":
        _copy_multiple_agents(path)
    elif name.lower() == "minimum-service":
        _copy_minimum_service(path)
    return path


def technodata(sector: Text, model: Text = "default") -> xr.Dataset:
    """Technology for a sector of a given example model."""
    from tempfile import TemporaryDirectory
    from muse.readers.csv import read_technologies

    sector = sector.lower()
    allowed = {"residential", "power", "gas", "preset"}
    if sector == "preset":
        raise RuntimeError("The preset sector has no technodata.")
    if sector not in allowed:
        raise RuntimeError(f"This model only knows about sectors {allowed}.")
    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        return read_technologies(
            path / "technodata" / sector.lower() / "Technodata.csv",
            path / "technodata" / sector.lower() / "CommOut.csv",
            path / "technodata" / sector.lower() / "CommIn.csv",
            path / "input" / "GlobalCommodities.csv",
        )


def search_space(sector: Text, model: Text = "default") -> xr.DataArray:
    """Determines which technology is considered for which asset.

    Used in constraints or during investment.
    """
    from numpy import ones

    technology = technodata(sector, model).technology
    return xr.DataArray(
        ones((len(technology), len(technology)), dtype=bool),
        coords=dict(asset=technology.values, replacement=technology.values),
        dims=("asset", "replacement"),
    )


def sector(sector: Text, model: Text = "default") -> AbstractSector:
    from tempfile import TemporaryDirectory
    from muse.readers.toml import read_settings
    from muse.sectors import SECTORS_REGISTERED

    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml")
        kind = getattr(settings.sectors, sector).type
        return SECTORS_REGISTERED[kind](sector, settings)


def available_sectors(*sectors: Text, model: Text = "default") -> List[Text]:
    from tempfile import TemporaryDirectory
    from muse.readers.toml import read_settings, undo_damage

    with TemporaryDirectory() as tmpdir:
        path = copy_model(model, tmpdir)
        settings = read_settings(path / "settings.toml").sectors
        return [u for u in undo_damage(settings).keys() if u != "list"]


def mca_market(model: Text = "default") -> xr.Dataset:
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
                settings.global_input_files.projections,
                base_year_export=getattr(
                    settings.global_input_files, "base_year_export", None
                ),
                base_year_import=getattr(
                    settings.global_input_files, "base_year_import", None
                ),
                timeslices=settings.timeslices,
            )
            .sel(region=settings.regions)
            .interp(year=settings.time_framework, method=settings.interpolation_mode)
        )
        market["supply"] = zeros_like(market.exports)
        market["consumption"] = zeros_like(market.exports)

        return cast(xr.Dataset, market)


def residential_market(model: Text = "default") -> xr.Dataset:
    """Initial market as seen by the residential sector."""
    from muse.mca import single_year_iteration

    market = mca_market(model)
    sectors = [sector("residential_presets", model=model)]
    return cast(
        xr.Dataset,
        single_year_iteration(market, sectors)[0][
            ["prices", "supply", "consumption"]
        ].drop_vars("units_prices"),
    )


def _copy_default(path: Path):
    from shutil import copytree, copyfile

    copytree(example_data_dir() / "default" / "input", path / "input")
    copytree(example_data_dir() / "default" / "technodata", path / "technodata")
    copyfile(example_data_dir() / "default" / "settings.toml", path / "settings.toml")


def _copy_multiple_agents(path: Path):
    from shutil import copytree, copyfile
    from toml import load, dump

    copytree(example_data_dir() / "default" / "input", path / "input")
    copytree(example_data_dir() / "default" / "technodata", path / "technodata")
    toml = load(example_data_dir() / "default" / "settings.toml")
    toml["sectors"]["residential"]["subsectors"]["retro_and_new"][
        "agents"
    ] = "{path}/technodata/residential/Agents.csv"
    with (path / "settings.toml").open("w") as fileobj:
        dump(toml, fileobj)
    copyfile(
        example_data_dir() / "multiple_agents" / "Agents.csv",
        path / "technodata" / "residential" / "Agents.csv",
    )
    copyfile(
        example_data_dir() / "multiple_agents" / "residential" / "Technodata.csv",
        path / "technodata" / "residential" / "Technodata.csv",
    )


def _copy_medium(path: Path):
    from shutil import copytree, copyfile

    copytree(example_data_dir() / "medium" / "input", path / "input")
    copytree(example_data_dir() / "medium" / "technodata", path / "technodata")
    copytree(
        example_data_dir() / "default" / "technodata" / "power",
        path / "technodata" / "power",
    )
    copytree(
        example_data_dir() / "default" / "technodata" / "gas",
        path / "technodata" / "gas",
    )
    copyfile(
        example_data_dir() / "default" / "technodata" / "Agents.csv",
        path / "technodata" / "Agents.csv",
    )
    copyfile(example_data_dir() / "default" / "settings.toml", path / "settings.toml")


def _copy_minimum_service(path: Path):
    from shutil import copytree, copyfile

    copytree(example_data_dir() / "minimum_service" / "input", path / "input")
    copytree(example_data_dir() / "minimum_service" / "technodata", path / "technodata")
    copyfile(
        example_data_dir() / "minimum_service" / "settings.toml", path / "settings.toml"
    )


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
