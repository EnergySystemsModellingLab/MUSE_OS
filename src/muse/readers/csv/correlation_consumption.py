"""Reads and processes correlation consumption data from CSV files.

This will only run for preset sectors that have macro drivers and regression files.
Otherwise, read_presets will be used instead.
"""

from __future__ import annotations

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


def read_correlation_consumption(
    macro_drivers_path: Path,
    regression_path: Path,
    timeslice_shares_path: Path | None = None,
) -> xr.Dataset:
    """Read consumption data for a sector based on correlation files.

    This function calculates endogenous demand for a sector using macro drivers and
    regression parameters. It applies optional filters, handles sector aggregation,
    and distributes the consumption across timeslices if timeslice shares are provided.

    Args:
        macro_drivers_path: Path to macro drivers file
        regression_path: Path to regression parameters file
        timeslice_shares_path: Path to timeslice shares file (optional)

    Returns:
        xr.Dataset: Consumption data distributed across timeslices and regions
    """
    from muse.regressions import endogenous_demand
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    macro_drivers = read_macro_drivers(macro_drivers_path)
    regression_parameters = read_regression_parameters(regression_path)
    consumption = endogenous_demand(
        drivers=macro_drivers,
        regression_parameters=regression_parameters,
        forecast=0,
    )

    # Legacy: we permit regression parameters to split by sector, so have to sum
    if "sector" in consumption.dims:
        consumption = consumption.sum("sector")

    # Split by timeslice
    if timeslice_shares_path is not None:
        shares = read_timeslice_shares(timeslice_shares_path)
        consumption = broadcast_timeslice(consumption) * shares
    else:
        consumption = distribute_timeslice(consumption)

    return consumption


def read_timeslice_shares(path: Path) -> xr.DataArray:
    """Reads and processes timeslice shares data from a CSV file."""
    df = read_timeslice_shares_csv(path)
    return process_timeslice_shares(df)


def read_timeslice_shares_csv(path: Path) -> pd.DataFrame:
    """Reads sliceshare information into a DataFrame."""
    data = read_csv(
        path,
        required_columns=["region", "timeslice"],
        msg=f"Reading timeslice shares from {path}.",
    )

    return data


def process_timeslice_shares(data: pd.DataFrame) -> xr.DataArray:
    """Processes timeslice shares DataFrame into an xarray DataArray."""
    from muse.commodities import COMMODITIES
    from muse.timeslices import TIMESLICE

    # Extract commodity columns
    commodities = [c for c in data.columns if c in COMMODITIES.commodity.values]

    # Convert commodity columns to long format (i.e. single "commodity" column)
    data = data.melt(
        id_vars=["region", "timeslice"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Create multiindex for region and timeslice
    data = create_multiindex(
        data,
        index_columns=["region", "timeslice", "commodity"],
        index_names=["region", "timeslice", "commodity"],
        drop_columns=True,
    )

    # Create DataArray
    result = create_xarray_dataset(data).value.astype(float)

    # Assign timeslices
    result = result.assign_coords(timeslice=TIMESLICE.timeslice)

    # Check commodities
    result = check_commodities(result, fill_missing=True, fill_value=0)
    return result


def read_macro_drivers(path: Path) -> pd.DataFrame:
    """Reads and processes macro drivers data from a CSV file."""
    df = read_macro_drivers_csv(path)
    return process_macro_drivers(df)


def read_macro_drivers_csv(path: Path) -> pd.DataFrame:
    """Reads a standard MUSE csv file for macro drivers into a DataFrame."""
    table = read_csv(
        path,
        required_columns=["region", "variable"],
        msg=f"Reading macro drivers from {path}.",
    )

    # Validate required variables
    required_variables = ["Population", "GDP|PPP"]
    missing_variables = [
        var for var in required_variables if var not in table.variable.unique()
    ]
    if missing_variables:
        raise ValueError(f"Missing required variables in {path}: {missing_variables}")

    return table


def process_macro_drivers(data: pd.DataFrame) -> xr.Dataset:
    """Processes macro drivers DataFrame into an xarray Dataset."""
    # Drop unit column if present
    if "unit" in data.columns:
        data = data.drop(columns=["unit"])

    # Select year columns
    year_columns = [col for col in data.columns if col.isdigit()]

    # Convert year columns to long format (i.e. single "year" column)
    data = data.melt(
        id_vars=["variable", "region"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )

    # Pivot data to create Population and GDP|PPP columns
    data = data.pivot(
        index=["region", "year"],
        columns="variable",
        values="value",
    )

    # Legacy: rename Population to population and GDP|PPP to gdp
    if "Population" in data.columns:
        data = data.rename(columns={"Population": "population"})
    if "GDP|PPP" in data.columns:
        data = data.rename(columns={"GDP|PPP": "gdp"})

    # Create DataSet
    result = create_xarray_dataset(data)
    return result


def read_regression_parameters(path: Path) -> xr.Dataset:
    """Reads and processes regression parameters from a CSV file."""
    df = read_regression_parameters_csv(path)
    return process_regression_parameters(df)


def read_regression_parameters_csv(path: Path) -> pd.DataFrame:
    """Reads the regression parameters from a MUSE csv file into a DataFrame."""
    table = read_csv(
        path,
        required_columns=["region", "function_type", "coeff"],
        msg=f"Reading regression parameters from {path}.",
    )

    # Legacy: warn about "sector" column
    if "sector" in table.columns:
        getLogger(__name__).warning(
            f"The sector column (in file {path}) is deprecated. Please remove."
        )

    return table


def process_regression_parameters(data: pd.DataFrame) -> xr.Dataset:
    """Processes regression parameters DataFrame into an xarray Dataset."""
    from muse.commodities import COMMODITIES

    # Extract commodity columns
    commodities = [c for c in data.columns if c in COMMODITIES.commodity.values]

    # Melt to long format
    melted = data.melt(
        id_vars=["sector", "region", "function_type", "coeff"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Extract sector -> function_type mapping
    sector_to_ftype = melted.drop_duplicates(["sector", "function_type"])[
        ["sector", "function_type"]
    ].set_index("sector")["function_type"]

    # Pivot to create coefficient variables
    pivoted = melted.pivot_table(
        index=["sector", "region", "commodity"], columns="coeff", values="value"
    )

    # Create dataset and add function_type
    result = create_xarray_dataset(pivoted)
    result["function_type"] = xr.DataArray(
        sector_to_ftype[result.sector.values].astype(object),
        dims=["sector"],
        name="function_type",
    )

    # Check commodities
    result = check_commodities(result, fill_missing=True, fill_value=0)
    return result
