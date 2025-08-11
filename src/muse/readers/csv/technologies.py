"""Reads and processes technology data from multiple CSV files.

This runs once per sector, reading in csv files and outputting an xarray Dataset.

Several csv files are read in:
- technodictionary: contains technology parameters
- comm_out: contains output commodity data
- comm_in: contains input commodity data
- technodata_timeslices (optional): allows some parameters to be defined per timeslice
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


def read_technologies(
    technodata_path: Path,
    comm_out_path: Path,
    comm_in_path: Path,
    time_framework: list[int],
    interpolation_mode: str = "linear",
    technodata_timeslices_path: Path | None = None,
) -> xr.Dataset:
    """Reads and processes technology data from multiple CSV files.

    Will also interpolate data to the time framework if provided.

    Args:
        technodata_path: path to the technodata file
        comm_out_path: path to the comm_out file
        comm_in_path: path to the comm_in file
        time_framework: list of years to interpolate data to
        interpolation_mode: Interpolation mode to use
        technodata_timeslices_path: path to the technodata_timeslices file

    Returns:
        xr.Dataset: Dataset containing the processed technology data. Any fields
        that differ by year will contain a "year" dimension interpolated to the
        time framework. Other fields will not have a "year" dimension.
    """
    # Read all data
    technodata = read_technodictionary(technodata_path)
    comm_out = read_io_technodata(comm_out_path)
    comm_in = read_io_technodata(comm_in_path)
    technodata_timeslices = (
        read_technodata_timeslices(technodata_timeslices_path)
        if technodata_timeslices_path
        else None
    )

    # Assemble xarray Dataset
    return process_technologies(
        technodata,
        comm_out,
        comm_in,
        time_framework,
        interpolation_mode,
        technodata_timeslices,
    )


def read_technodictionary(path: Path) -> xr.Dataset:
    """Reads and processes technodictionary data from a CSV file."""
    df = read_technodictionary_csv(path)
    return process_technodictionary(df)


def read_technodictionary_csv(path: Path) -> pd.DataFrame:
    """Reads and formats technodata into a DataFrame."""
    required_columns = {
        "cap_exp",
        "region",
        "var_par",
        "fix_exp",
        "interest_rate",
        "utilization_factor",
        "minimum_service_factor",
        "year",
        "cap_par",
        "var_exp",
        "technology",
        "technical_life",
        "fix_par",
    }
    data = read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading technodictionary from {path}.",
    )

    # Check for deprecated columns
    if "fuel" in data.columns:
        msg = (
            f"The 'fuel' column in {path} has been deprecated. "
            "This information is now determined from CommIn files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)
    if "end_use" in data.columns:
        msg = (
            f"The 'end_use' column in {path} has been deprecated. "
            "This information is now determined from CommOut files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)
    if "scaling_size" in data.columns:
        msg = (
            f"The 'scaling_size' column in {path} has been deprecated. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)

    return data


def process_technodictionary(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodictionary DataFrame into an xarray Dataset."""
    # Create multiindex for technology and region
    data = create_multiindex(
        data,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    # Create dataset
    result = create_xarray_dataset(data)

    # Handle tech_type if present
    if "type" in result.variables:
        result["tech_type"] = result.type.isel(region=0, year=0)

    return result


def read_technodata_timeslices(path: Path) -> xr.Dataset:
    """Reads and processes technodata timeslices from a CSV file."""
    df = read_technodata_timeslices_csv(path)
    return process_technodata_timeslices(df)


def read_technodata_timeslices_csv(path: Path) -> pd.DataFrame:
    """Reads and formats technodata timeslices into a DataFrame."""
    from muse.timeslices import TIMESLICE

    timeslice_columns = set(TIMESLICE.coords["timeslice"].indexes["timeslice"].names)
    required_columns = {
        "utilization_factor",
        "technology",
        "minimum_service_factor",
        "region",
        "year",
    } | timeslice_columns
    return read_csv(
        path,
        required_columns=required_columns,
        exclude_extra_columns=True,
        msg=f"Reading technodata timeslices from {path}.",
    )


def process_technodata_timeslices(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodata timeslices DataFrame into an xarray Dataset."""
    from muse.timeslices import TIMESLICE, sort_timeslices

    # Create multiindex for all columns except factor columns
    factor_columns = ["utilization_factor", "minimum_service_factor", "obj_sort"]
    index_columns = [col for col in data.columns if col not in factor_columns]
    data = create_multiindex(
        data,
        index_columns=index_columns,
        index_names=index_columns,
        drop_columns=True,
    )

    # Create dataset
    result = create_xarray_dataset(data)

    # Stack timeslice levels (month, day, hour) into a single timeslice dimension
    timeslice_levels = TIMESLICE.coords["timeslice"].indexes["timeslice"].names
    if all(level in result.dims for level in timeslice_levels):
        result = result.stack(timeslice=timeslice_levels)
    return sort_timeslices(result)


def read_io_technodata(path: Path) -> xr.Dataset:
    """Reads and processes input/output technodata from a CSV file."""
    df = read_io_technodata_csv(path)
    return process_io_technodata(df)


def read_io_technodata_csv(path: Path) -> pd.DataFrame:
    """Reads process inputs or outputs into a DataFrame."""
    data = read_csv(
        path,
        required_columns=["technology", "region", "year"],
        msg=f"Reading IO technodata from {path}.",
    )

    # Unspecified Level values default to "fixed"
    if "level" in data.columns:
        data["level"] = data["level"].fillna("fixed")
    else:
        # Particularly relevant to outputs files where the Level column is omitted by
        # default, as only "fixed" outputs are allowed.
        data["level"] = "fixed"

    return data


def process_io_technodata(data: pd.DataFrame) -> xr.Dataset:
    """Processes IO technodata DataFrame into an xarray Dataset."""
    from muse.commodities import COMMODITIES

    # Extract commodity columns
    commodities = [c for c in data.columns if c in COMMODITIES.commodity.values]

    # Convert commodity columns to long format (i.e. single "commodity" column)
    data = data.melt(
        id_vars=["technology", "region", "year", "level"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Pivot data to create fixed and flexible columns
    data = data.pivot(
        index=["technology", "region", "year", "commodity"],
        columns="level",
        values="value",
    )

    # Create xarray dataset
    result = create_xarray_dataset(data)

    # Fill in flexible data
    if "flexible" in result.data_vars:
        result["flexible"] = result.flexible.fillna(0)
    else:
        result["flexible"] = xr.zeros_like(result.fixed).rename("flexible")

    # Check commodities
    result = check_commodities(result, fill_missing=True, fill_value=0)
    return result


def process_technologies(
    technodata: xr.Dataset,
    comm_out: xr.Dataset,
    comm_in: xr.Dataset,
    time_framework: list[int],
    interpolation_mode: str = "linear",
    technodata_timeslices: xr.Dataset | None = None,
) -> xr.Dataset:
    """Processes technology data DataFrames into an xarray Dataset."""
    from muse.commodities import COMMODITIES, CommodityUsage
    from muse.timeslices import drop_timeslice
    from muse.utilities import interpolate_technodata

    # Process inputs/outputs
    ins = comm_in.rename(flexible="flexible_inputs", fixed="fixed_inputs")
    outs = comm_out.rename(flexible="flexible_outputs", fixed="fixed_outputs")

    # Legacy: Remove flexible outputs
    if not (outs["flexible_outputs"] == 0).all():
        raise ValueError(
            "'flexible' outputs are not permitted. All outputs must be 'fixed'"
        )
    outs = outs.drop_vars("flexible_outputs")

    # Collect all years from the time framework and data files
    time_framework = list(
        set(time_framework).union(
            technodata.year.values.tolist(),
            ins.year.values.tolist(),
            outs.year.values.tolist(),
            technodata_timeslices.year.values.tolist() if technodata_timeslices else [],
        )
    )

    # Interpolate data to match the time framework
    technodata = interpolate_technodata(technodata, time_framework, interpolation_mode)
    outs = interpolate_technodata(outs, time_framework, interpolation_mode)
    ins = interpolate_technodata(ins, time_framework, interpolation_mode)
    if technodata_timeslices:
        technodata_timeslices = interpolate_technodata(
            technodata_timeslices, time_framework, interpolation_mode
        )

    # Merge inputs/outputs with technodata
    technodata = technodata.merge(outs).merge(ins)

    # Merge technodata_timeslices if provided. This will prioritise values defined in
    # technodata_timeslices, and fallback to the non-timesliced technodata for any
    # values that are not defined in technodata_timeslices.
    if technodata_timeslices:
        technodata["utilization_factor"] = (
            technodata_timeslices.utilization_factor.combine_first(
                technodata.utilization_factor
            )
        )
        technodata["minimum_service_factor"] = drop_timeslice(
            technodata_timeslices.minimum_service_factor.combine_first(
                technodata.minimum_service_factor
            )
        )

    # Check commodities
    technodata = check_commodities(technodata, fill_missing=False)

    # Add info about commodities
    technodata = technodata.merge(COMMODITIES.sel(commodity=technodata.commodity))

    # Add commodity usage flags
    technodata["comm_usage"] = (
        "commodity",
        CommodityUsage.from_technologies(technodata).values,
    )
    technodata = technodata.drop_vars("commodity_type")

    # Check utilization and minimum service factors
    check_utilization_and_minimum_service_factors(technodata)

    return technodata


def check_utilization_and_minimum_service_factors(data: xr.Dataset) -> None:
    """Check utilization and minimum service factors in an xarray dataset.

    Args:
        data: xarray Dataset containing utilization_factor and minimum_service_factor
    """
    if "utilization_factor" not in data.data_vars:
        raise ValueError(
            "A technology needs to have a utilization factor defined for every "
            "timeslice."
        )

    # Check UF not all zero (sum across timeslice dimension if it exists)
    if "timeslice" in data.dims:
        utilization_sum = data.utilization_factor.sum(dim="timeslice")
    else:
        utilization_sum = data.utilization_factor

    if (utilization_sum == 0).any():
        raise ValueError(
            "A technology can not have a utilization factor of 0 for every timeslice."
        )

    # Check UF in range
    utilization = data.utilization_factor
    if not ((utilization >= 0) & (utilization <= 1)).all():
        raise ValueError(
            "Utilization factor values must all be between 0 and 1 inclusive."
        )

    # Check MSF in range
    min_service_factor = data.minimum_service_factor
    if not ((min_service_factor >= 0) & (min_service_factor <= 1)).all():
        raise ValueError(
            "Minimum service factor values must all be between 0 and 1 inclusive."
        )

    # Check UF not below MSF
    if (data.utilization_factor < data.minimum_service_factor).any():
        raise ValueError(
            "Utilization factors must all be greater than or equal "
            "to their corresponding minimum service factors."
        )
