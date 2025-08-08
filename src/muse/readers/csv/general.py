from pathlib import Path

import pandas as pd
import xarray as xr

from .helpers import create_xarray_dataset, read_csv


def read_attribute_table(path: Path) -> xr.Dataset:
    """Reads and processes attribute table data from a CSV file."""
    df = read_attribute_table_csv(path)
    return process_attribute_table(df)


def read_attribute_table_csv(path: Path) -> pd.DataFrame:
    """Read a standard MUSE csv file for price projections into a DataFrame."""
    table = read_csv(
        path,
        required_columns=["region", "attribute", "year"],
        msg=f"Reading attribute table from {path}.",
    )
    return table


def process_attribute_table(data: pd.DataFrame) -> xr.Dataset:
    """Process attribute table DataFrame into an xarray Dataset."""
    # Extract commodity columns
    commodities = [
        col for col in data.columns if col not in ["region", "year", "attribute"]
    ]

    # Convert commodity columns to long format (i.e. single "commodity" column)
    data = data.melt(
        id_vars=["region", "year", "attribute"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Pivot data over attributes
    data = data.pivot(
        index=["region", "year", "commodity"],
        columns="attribute",
        values="value",
    )

    # Create DataSet
    result = create_xarray_dataset(data)
    return result
