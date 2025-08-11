from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from .general import process_attribute_table
from .helpers import check_commodities, read_csv


def read_initial_market(
    projections_path: Path,
    base_year_import_path: Path | None = None,
    base_year_export_path: Path | None = None,
    currency: str | None = None,
) -> xr.Dataset:
    """Reads and processes initial market data.

    Args:
        projections_path: path to the projections file
        base_year_import_path: path to the base year import file (optional)
        base_year_export_path: path to the base year export file (optional)
        currency: currency string (e.g. "USD")

    Returns:
        xr.Dataset: Dataset containing initial market data.
    """
    # Read projections
    projections_df = read_projections_csv(projections_path)

    # Read base year export (optional)
    if base_year_export_path:
        export_df = read_csv(
            base_year_export_path,
            msg=f"Reading base year export from {base_year_export_path}.",
        )
    else:
        export_df = None

    # Read base year import (optional)
    if base_year_import_path:
        import_df = read_csv(
            base_year_import_path,
            msg=f"Reading base year import from {base_year_import_path}.",
        )
    else:
        import_df = None

    # Assemble into xarray Dataset
    result = process_initial_market(projections_df, import_df, export_df, currency)
    return result


def read_projections_csv(path: Path) -> pd.DataFrame:
    """Reads projections data from a CSV file."""
    required_columns = {
        "region",
        "attribute",
        "year",
    }
    projections_df = read_csv(
        path, required_columns=required_columns, msg=f"Reading projections from {path}."
    )
    return projections_df


def process_initial_market(
    projections_df: pd.DataFrame,
    import_df: pd.DataFrame | None,
    export_df: pd.DataFrame | None,
    currency: str | None = None,
) -> xr.Dataset:
    """Process market data DataFrames into an xarray Dataset.

    Args:
        projections_df: DataFrame containing projections data
        import_df: Optional DataFrame containing import data
        export_df: Optional DataFrame containing export data
        currency: Currency string (e.g. "USD")
    """
    from muse.commodities import COMMODITIES
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    # Process projections
    projections = process_attribute_table(projections_df).commodity_price.astype(
        "float64"
    )

    # Process optional trade data
    if export_df is not None:
        base_year_export = process_attribute_table(export_df).exports.astype("float64")
    else:
        base_year_export = xr.zeros_like(projections)

    if import_df is not None:
        base_year_import = process_attribute_table(import_df).imports.astype("float64")
    else:
        base_year_import = xr.zeros_like(projections)

    # Distribute data over timeslices
    projections = broadcast_timeslice(projections, level=None)
    base_year_export = distribute_timeslice(base_year_export, level=None)
    base_year_import = distribute_timeslice(base_year_import, level=None)

    # Assemble into xarray
    result = xr.Dataset(
        {
            "prices": projections,
            "exports": base_year_export,
            "imports": base_year_import,
            "static_trade": base_year_import - base_year_export,
        }
    )

    # Check commodities
    result = check_commodities(result, fill_missing=True, fill_value=0)

    # Add units_prices coordinate
    # Only added if the currency is specified and commodity units are defined
    if currency and "unit" in COMMODITIES.data_vars:
        units_prices = [
            f"{currency}/{COMMODITIES.sel(commodity=c).unit.item()}"
            for c in result.commodity.values
        ]
        result = result.assign_coords(units_prices=("commodity", units_prices))

    return result
