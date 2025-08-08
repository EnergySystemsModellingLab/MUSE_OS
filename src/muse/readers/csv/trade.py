from pathlib import Path

import pandas as pd
import xarray as xr

from .helpers import (
    create_assets,
    create_multiindex,
    create_xarray_dataset,
    read_csv,
)


def read_trade_technodata(path: Path) -> xr.Dataset:
    """Reads and processes trade technodata from a CSV file."""
    df = read_trade_technodata_csv(path)
    return process_trade_technodata(df)


def read_trade_technodata_csv(path: Path) -> pd.DataFrame:
    required_columns = {"technology", "region", "parameter"}
    return read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading trade technodata from {path}.",
    )


def process_trade_technodata(data: pd.DataFrame) -> xr.Dataset:
    # Drop unit column if present
    if "unit" in data.columns:
        data = data.drop(columns=["unit"])

    # Select region columns
    # TODO: this is a bit unsafe as user could supply other columns
    regions = [
        col for col in data.columns if col not in ["technology", "region", "parameter"]
    ]

    # Melt data over regions
    data = data.melt(
        id_vars=["technology", "region", "parameter"],
        value_vars=regions,
        var_name="dst_region",
        value_name="value",
    )

    # Pivot data over parameters
    data = data.pivot(
        index=["technology", "region", "dst_region"],
        columns="parameter",
        values="value",
    )

    # Create DataSet
    return create_xarray_dataset(data)


def read_existing_trade(path: Path) -> xr.DataArray:
    """Reads and processes existing trade data from a CSV file."""
    df = read_existing_trade_csv(path)
    return process_existing_trade(df)


def read_existing_trade_csv(path: Path) -> pd.DataFrame:
    required_columns = {
        "region",
        "technology",
        "year",
    }
    return read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading existing trade from {path}.",
    )


def process_existing_trade(data: pd.DataFrame) -> xr.DataArray:
    # Select region columns
    # TODO: this is a bit unsafe as user could supply other columns
    regions = [
        col for col in data.columns if col not in ["technology", "region", "year"]
    ]

    # Melt data over regions
    data = data.melt(
        id_vars=["technology", "region", "year"],
        value_vars=regions,
        var_name="dst_region",
        value_name="value",
    )

    # Create multiindex for region, dst_region, technology and year
    data = create_multiindex(
        data,
        index_columns=["region", "dst_region", "technology", "year"],
        index_names=["region", "dst_region", "technology", "year"],
        drop_columns=True,
    )

    # Create DataArray
    result = create_xarray_dataset(data).value.astype(float)

    # Create assets from technologies
    result = create_assets(result)
    return result
