"""Reads and processes existing capacity data from a CSV file.

This runs once per subsector, reading in a csv file and outputting an xarray DataArray.
"""

from pathlib import Path

import pandas as pd
import xarray as xr

from .helpers import create_assets, create_multiindex, create_xarray_dataset, read_csv


def read_assets(path: Path) -> xr.DataArray:
    """Reads and processes existing capacity data from a CSV file."""
    df = read_existing_capacity_csv(path)
    return process_existing_capacity(df)


def read_existing_capacity_csv(path: Path) -> pd.DataFrame:
    """Reads and formats data about initial capacity into a DataFrame."""
    required_columns = {
        "region",
        "technology",
    }
    return read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading initial capacity from {path}.",
    )


def process_existing_capacity(data: pd.DataFrame) -> xr.DataArray:
    """Processes initial capacity DataFrame into an xarray DataArray."""
    # Drop unit column if present
    if "unit" in data.columns:
        data = data.drop(columns=["unit"])

    # Select year columns
    year_columns = [col for col in data.columns if col.isdigit()]

    # Convert year columns to long format (i.e. single "year" column)
    data = data.melt(
        id_vars=["technology", "region"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )

    # Create multiindex for region, technology, and year
    data = create_multiindex(
        data,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    # Create Dataarray
    result = create_xarray_dataset(data).value.astype(float)

    # Create assets
    result = create_assets(result)
    return result
