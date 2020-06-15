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
from typing import Optional, Text, Union

import numpy as np
import xarray as xr

from muse.mca import MCA


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
    from shutil import copytree, copyfile, rmtree
    from toml import load, dump

    if name.lower() not in {"default", "multiple-agents"}:
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

    copytree(example_data_dir() / "input", path / "input")
    copytree(example_data_dir() / "technodata", path / "technodata")
    if name.lower() == "default":
        copyfile(example_data_dir() / "settings.toml", path / "settings.toml")
    if name.lower() == "multiple-agents":
        toml = load(example_data_dir() / "settings.toml")
        toml["sectors"]["residential"][
            "agents"
        ] = "{path}/technodata/residential/Agents.csv"
        with (path / "settings.toml").open("w") as fileobj:
            dump(toml, fileobj)
        copyfile(
            example_data_dir() / "multiple_agents" / "Agents.csv",
            path / "technodata" / "residential" / "Agents.csv",
        )
        copyfile(
            example_data_dir() / "multiple_agents" / "residential" / "technodata.csv",
            path / "technodata" / "residential" / "Technodata.csv",
        )
    return path


def technodata(sector: Text) -> xr.Dataset:
    """Technology for a sector of the default example model."""
    from muse.readers.csv import read_technologies

    sector = sector.lower()
    allowed = {"residential", "power", "gas", "preset"}
    if sector == "preset":
        raise RuntimeError("The preset sector has no technodata.")
    if sector not in allowed:
        raise RuntimeError(f"This model only knows about sectors {allowed}.")
    return read_technologies(
        example_data_dir() / "technodata" / sector.title() / "Technodata.csv",
        example_data_dir() / "technodata" / sector.title() / "CommOut.csv",
        example_data_dir() / "technodata" / sector.title() / "CommIn.csv",
        example_data_dir() / "input" / "GlobalCommodities.csv",
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
