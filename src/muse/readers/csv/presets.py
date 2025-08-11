"""Reads and processes preset data from multiple CSV files.

This runs once per sector, reading in csv files and outputting an xarray Dataset.
"""

from logging import getLogger
from pathlib import Path

import pandas as pd
import xarray as xr

from .helpers import (
    check_commodities,
    create_multiindex,
    create_xarray_dataset,
    read_csv,
)


def read_presets(presets_paths: Path) -> xr.Dataset:
    """Reads and processes preset data from multiple CSV files.

    Accepts a path pattern for presets files, e.g. `Path("path/to/*Consumption.csv")`.
    The file name of each file must contain a year (e.g. "2020Consumption.csv").
    """
    from glob import glob
    from re import match

    # Find all files matching the path pattern
    allfiles = [Path(p) for p in glob(str(presets_paths))]
    if len(allfiles) == 0:
        raise OSError(f"No files found with paths {presets_paths}")

    # Read all files
    datas: dict[int, pd.DataFrame] = {}
    for path in allfiles:
        # Extract year from filename
        reyear = match(r"\S*.(\d{4})\S*\.csv", path.name)
        if reyear is None:
            raise OSError(f"Unexpected filename {path.name}")
        year = int(reyear.group(1))
        if year in datas:
            raise OSError(f"Year f{year} was found twice")

        # Read data
        data = read_presets_csv(path)
        data["year"] = year
        datas[year] = data

    # Process data
    datas = process_presets(datas)
    return datas


def read_presets_csv(path: Path) -> pd.DataFrame:
    data = read_csv(
        path,
        required_columns=["region", "timeslice"],
        msg=f"Reading presets from {path}.",
    )

    # Legacy: drop technology column and sum data (PR #448)
    if "technology" in data.columns:
        getLogger(__name__).warning(
            f"The technology (or ProcessName) column in file {path} is "
            "deprecated. Data has been summed across technologies, and this column "
            "has been dropped."
        )
        data = (
            data.drop(columns=["technology"])
            .groupby(["region", "timeslice"])
            .sum()
            .reset_index()
        )

    return data


def process_presets(datas: dict[int, pd.DataFrame]) -> xr.Dataset:
    """Processes preset DataFrames into an xarray Dataset."""
    from muse.commodities import COMMODITIES
    from muse.timeslices import TIMESLICE

    # Combine into a single DataFrame
    data = pd.concat(datas.values())

    # Extract commodity columns
    commodities = [c for c in data.columns if c in COMMODITIES.commodity.values]

    # Convert commodity columns to long format (i.e. single "commodity" column)
    data = data.melt(
        id_vars=["region", "year", "timeslice"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Create multiindex for region, year, timeslice and commodity
    data = create_multiindex(
        data,
        index_columns=["region", "year", "timeslice", "commodity"],
        index_names=["region", "year", "timeslice", "commodity"],
        drop_columns=True,
    )

    # Create DataArray
    result = create_xarray_dataset(data).value.astype(float)

    # Assign timeslices
    result = result.assign_coords(timeslice=TIMESLICE.timeslice)

    # Check commodities
    result = check_commodities(result, fill_missing=True, fill_value=0)
    return result
