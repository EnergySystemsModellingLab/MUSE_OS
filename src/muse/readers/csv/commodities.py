from logging import getLogger
from pathlib import Path

import pandas as pd
import xarray as xr

from .helpers import camel_to_snake, create_xarray_dataset, standardize_dataframe


def read_global_commodities(path: Path) -> xr.Dataset:
    """Reads and processes global commodities data from a CSV file."""
    df = read_global_commodities_csv(path)
    return process_global_commodities(df)


def read_global_commodities_csv(path: Path) -> pd.DataFrame:
    """Reads commodities information from input into a DataFrame."""
    # Due to legacy reasons, users can supply both Commodity and CommodityName columns
    # In this case, we need to remove the Commodity column to avoid conflicts
    # This is fine because Commodity just contains a long description that isn't needed
    getLogger(__name__).info(f"Reading global commodities from {path}.")
    df = pd.read_csv(path)
    df = df.rename(columns=camel_to_snake)
    if "commodity" in df.columns and "commodity_name" in df.columns:
        df = df.drop(columns=["commodity"])

    required_columns = {
        "commodity",
        "commodity_type",
    }
    data = standardize_dataframe(
        df,
        required_columns=required_columns,
    )

    # Raise warning if units are not defined
    if "unit" not in data.columns:
        msg = (
            "No units defined for commodities. Please define units for all commodities "
            "in the global commodities file."
        )
        getLogger(__name__).warning(msg)

    return data


def process_global_commodities(data: pd.DataFrame) -> xr.Dataset:
    """Processes global commodities DataFrame into an xarray Dataset."""
    # Drop description column if present. It's useful to include in the file, but we
    # don't need it for the simulation.
    if "description" in data.columns:
        data = data.drop(columns=["description"])

    data.index = [u for u in data.commodity]
    data = data.drop("commodity", axis=1)
    data.index.name = "commodity"
    return create_xarray_dataset(data)
