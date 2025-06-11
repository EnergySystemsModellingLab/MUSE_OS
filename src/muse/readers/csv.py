"""Ensemble of functions to read MUSE data."""

from __future__ import annotations

__all__ = [
    "read_agent_parameters",
    "read_global_commodities",
    "read_initial_assets",
    "read_initial_market",
    "read_macro_drivers",
    "read_presets",
    "read_regression_parameters",
    "read_technodictionary",
    "read_technologies",
    "read_timeslice_shares",
]

from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from muse.errors import UnitsConflictInCommodities

# Global mapping of column names to their standardized versions
COLUMN_RENAMES = {
    "process_name": "technology",
    "sector_name": "sector",
    "region_name": "region",
    "time": "year",
    "commodity_name": "commodity",
    "commodity_type": "comm_type",
    "commodity_emission_factor_co2": "emmission_factor",
    "commodity_price": "prices",
    "units_commodity_price": "units_prices",
    "enduse": "end_use",
}

# Columns who's values should be converted from camelCase to snake_case
CAMEL_TO_SNAKE_COLUMNS = [
    "tech_type",
    "commodity",
    "comm_type",
    "agent_share",
    "attribute",
    "sector",
]

# Global mapping of column names to their expected types
COLUMN_TYPES = {
    "year": int,
    "region": str,
    "technology": str,
    "commodity": str,
    "sector": str,
    "attribute": str,
    "variable": str,
    "timeslice": str,
    "name": str,
    "comm_type": str,
    "tech_type": str,
    "type": str,
    "function_type": str,
    "level": str,
    "search_rule": str,
    "decision_method": str,
    "quantity": float,
    "share": float,
    "coeff": float,
    "value": float,
    "utilization_factor": float,
    "minimum_service_factor": float,
    "maturity_threshold": float,
    "spend_limit": float,
    "prices": float,
    "emmission_factor": float,
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names in a DataFrame.

    This function:
    1. Converts column names to snake_case
    2. Applies the global COLUMN_RENAMES mapping
    3. Preserves any columns not in the mapping

    Args:
        df: DataFrame to standardize

    Returns:
        DataFrame with standardized column names
    """
    # First convert to snake_case
    df = df.rename(columns=camel_to_snake)

    # Drop any columns that start with "Unname"
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

    # Then apply global mapping
    df = df.rename(columns=COLUMN_RENAMES)

    return df


def create_multiindex(
    data: pd.DataFrame,
    index_columns: list[str],
    index_names: list[str],
    drop_columns: bool = True,
) -> pd.DataFrame:
    """Creates a MultiIndex from specified columns.

    Args:
        data: DataFrame to create index from
        index_columns: List of column names to use for index
        index_names: List of names for the index levels
        drop_columns: Whether to drop the original columns

    Returns:
        DataFrame with new MultiIndex
    """
    index = pd.MultiIndex.from_arrays(
        [data[col] for col in index_columns], names=index_names
    )
    result = data.copy()
    result.index = index
    if drop_columns:
        result = result.drop(columns=index_columns)
    return result


def create_xarray_dataset(
    data: pd.DataFrame,
    name: str | None = None,
    coords: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Creates an xarray Dataset from a DataFrame with standardized options.

    Args:
        data: DataFrame to convert
        name: Optional name for the Dataset
        coords: Optional coordinates to add
        attrs: Optional attributes to add

    Returns:
        xarray Dataset
    """
    result = xr.Dataset.from_dataframe(data)
    if name:
        result.name = name
    if coords:
        for key, value in coords.items():
            result.coords[key] = value
    if attrs:
        for key, value in attrs.items():
            result.attrs[key] = value
    return result


def create_xarray_dataarray(
    data: pd.DataFrame,
    name: str | None = None,
    coords: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Creates an xarray DataArray from a DataFrame with standardized options.

    Args:
        data: DataFrame to convert
        name: Optional name for the DataArray
        coords: Optional coordinates to add
        attrs: Optional attributes to add

    Returns:
        xarray DataArray
    """
    result = xr.DataArray(data)
    if name:
        result.name = name
    if coords:
        for key, value in coords.items():
            result.coords[key] = value
    if attrs:
        for key, value in attrs.items():
            result.attrs[key] = value
    return result


def camel_to_snake(name: str) -> str:
    """Transforms CamelCase to snake_case."""
    from re import sub

    re = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    result = sub("([a-z0-9])([A-Z])", r"\1_\2", re).lower()
    result = result.replace("co2", "CO2")
    result = result.replace("ch4", "CH4")
    result = result.replace("n2_o", "N2O")
    result = result.replace("f-gases", "F-gases")
    return result


def to_numeric(x):
    """Converts a value to numeric if possible.

    Args:
        x: The value to convert.

    Returns:
        The value converted to numeric if possible, otherwise the original value.
    """
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x


def convert_column_types(data: pd.DataFrame) -> pd.DataFrame:
    """Converts DataFrame columns to their expected types.

    Args:
        data: DataFrame to convert

    Returns:
        DataFrame with converted column types
    """
    result = data.copy()
    for col, expected_type in COLUMN_TYPES.items():
        if col in result.columns:
            try:
                if expected_type is int:
                    result[col] = pd.to_numeric(result[col], downcast="integer")
                elif expected_type is float:
                    result[col] = pd.to_numeric(result[col])
                elif expected_type is str:
                    result[col] = result[col].astype(str)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Could not convert column '{col}' to {expected_type.__name__}: {e}"
                )
    return result

    if "scaling_size" in data.columns:
        msg = (
            f"The 'ScalingSize' column in {source} has been deprecated. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)

    data = data.rename(columns=camel_to_snake)
    data = data[data.process_name != "Unit"]

def read_csv(
    filename: Path,
    float_precision: str = "high",
    required_columns: list[str] | None = None,
    msg: str | None = None,
) -> pd.DataFrame:
    """Reads and standardizes a CSV file into a DataFrame.

    Args:
        filename: Path to the CSV file
        float_precision: Precision to use when reading floats
        required_columns: List of column names that must be present (optional)
        msg: Message to log (optional)

    Returns:
        DataFrame containing the standardized data
    """
    # Check if file exists
    if not filename.is_file():
        raise OSError(f"{filename} does not exist.")

    # Log message
    if msg:
        getLogger(__name__).info(msg)

    # Check if there's a units row (in which case we need to skip it)
    with open(filename) as f:
        next(f)  # Skip header row
        first_data_row = f.readline().strip()
    skiprows = [1] if first_data_row.startswith("Unit") else None

    # Read the file
    data = pd.read_csv(
        filename,
        float_precision=float_precision,
        low_memory=False,
        skiprows=skiprows,
    )

    # Standardize column names
    data = standardize_columns(data)

    # Convert specified columns from camelCase to snake_case
    for col in CAMEL_TO_SNAKE_COLUMNS:
        if col in data.columns:
            data = data.rename(columns={col: camel_to_snake(col)})

    # Check/convert data types
    data = convert_column_types(data)

    # Validate required columns if provided
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {filename}: {missing_columns}"
            )

    return data


def read_technodictionary(path: Path) -> xr.Dataset:
    df = read_technodictionary_csv(path)
    return process_technodictionary(df)


def read_technodictionary_csv(filename: Path) -> pd.DataFrame:
    """Reads and formats technodata into a DataFrame.

    Args:
        filename: Path to the technodictionary CSV file

    Returns:
        DataFrame containing the technodictionary data
    """
    required_columns = {
        "cap_exp",
        "region",
        "var_par",
        "fix_exp",
        "interest_rate",
        "utilization_factor",
        "scaling_size",
        "year",
        "cap_par",
        "var_exp",
        "technology",
        "technical_life",
        "fix_par",
    }
    csv = read_csv(filename, required_columns=required_columns)

    # Check for deprecated columns
    if "fuel" in csv.columns:
        msg = (
            f"The 'Fuel' column in {filename} has been deprecated. "
            "This information is now determined from CommIn files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)
    if "end_use" in csv.columns:
        msg = (
            f"The 'EndUse' column in {filename} has been deprecated. "
            "This information is now determined from CommOut files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)

    return csv


def process_technodictionary(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodictionary DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the technodictionary data

    Returns:
        xarray Dataset containing the processed technodictionary
    """
    # Create multiindex for technology, region, and year
    data = create_multiindex(
        data,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    # Convert time to integers
    data.index = data.index.set_levels(
        [
            data.index.levels[0],
            data.index.levels[1],
            [int(u) for u in data.index.levels[2]],
        ],
        level=[0, 1, 2],
    )

    # Set column and index names
    data.columns.name = "technodata"
    data.index.name = "technology"

    # Create dataset
    result = create_xarray_dataset(data.sort_index())

    # Handle tech_type if present
    if "type" in result.variables:
        result["tech_type"] = result.type.isel(region=0, year=0)

    # Sanity checks for year dimension
    if "year" in result.dims:
        assert len(set(result.year.data)) == result.year.data.size
        result = result.sortby("year")
        if len(result.year) == 1:
            result = result.isel(year=0, drop=True)

    return result


def read_technodata_timeslices_csv(filename: Path) -> pd.DataFrame:
    """Reads and formats technodata timeslices into a DataFrame.

    Args:
        filename: Path to the technodata timeslices CSV file

    Returns:
        DataFrame containing the technodata timeslices data
    """
    required_columns = {
        "utilization_factor",
        "technology",
        "minimum_service_factor",
        "region",
        "year",
    }
    csv = read_csv(filename, required_columns=required_columns)
    return csv


def process_technodata_timeslices(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodata timeslices DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the technodata timeslices data

    Returns:
        xarray Dataset containing the processed technodata timeslices
    """
    from muse.timeslices import sort_timeslices

    # Create multiindex excluding factor columns
    factor_columns = ["utilization_factor", "minimum_service_factor", "obj_sort"]
    index_columns = [col for col in data.columns if col not in factor_columns]
    data = create_multiindex(
        data, index_columns=index_columns, index_names=["technology"], drop_columns=True
    )

    # Set column names
    data.columns.name = "technodata_timeslice"
    data.index.name = "technology"

    # Filter to only factor columns
    data = data.filter(factor_columns)

    # Create dataset
    result = create_xarray_dataset(data)

    # Stack timeslice levels
    timeslice_levels = [
        item
        for item in list(result.coords)
        if item not in ["technology", "region", "year"]
    ]
    result = result.stack(timeslice=timeslice_levels)

    return sort_timeslices(result)


def read_io_technodata_csv(filename: Path) -> pd.DataFrame:
    """Reads process inputs or outputs into a DataFrame.

    Args:
        filename: Path to the IO technodata CSV file

    Returns:
        DataFrame containing the IO technodata
    """
    csv = read_csv(filename, required_columns=["technology", "region", "year"])

    # Unspecified Level values default to "fixed"
    if "level" in csv.columns:
        csv["level"] = csv["level"].fillna("fixed")
    else:
        # Particularly relevant to outputs files where the Level column is omitted by
        # default, as only "fixed" outputs are allowed.
        csv["level"] = "fixed"

    return csv


def process_io_technodata(data: pd.DataFrame) -> xr.Dataset:
    """Processes IO technodata DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the IO technodata

    Returns:
        xarray Dataset containing the processed IO technodata
    """
    # Create multiindex for technology, region, and year
    data = create_multiindex(
        data,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    # Convert time to integers
    data.index = data.index.set_levels(
        [
            data.index.levels[0],
            data.index.levels[1],
            [int(u) for u in data.index.levels[2]],
        ],
        level=[0, 1, 2],
    )

    # Set column names
    data.columns.name = "commodity"
    data.index.name = "technology"

    # Split into fixed and flexible sets
    fixed_set = create_xarray_dataset(
        data[data.level == "fixed"], name="fixed"
    ).drop_vars("level")

    flexible_set = create_xarray_dataset(
        data[data.level == "flexible"], name="flexible"
    ).drop_vars("level")

    # Create commodity dimension
    commodity = xr.DataArray(
        list(fixed_set.data_vars.keys()), dims="commodity", name="commodity"
    )

    # Concatenate fixed and flexible sets
    fixed = xr.concat(fixed_set.data_vars.values(), dim=commodity)
    flexible = xr.concat(flexible_set.data_vars.values(), dim=commodity)

    # Create result dataset
    result = create_xarray_dataset(data_vars={"fixed": fixed, "flexible": flexible})
    result["flexible"] = result.flexible.fillna(0)

    return result


def read_initial_assets(path: Path) -> xr.DataArray:
    df = read_initial_assets_csv(path)
    return process_initial_assets(df)


def read_initial_assets_csv(filename: Path) -> pd.DataFrame:
    """Reads and formats data about initial capacity into a DataFrame.

    Args:
        filename: Path to the initial assets CSV file

    Returns:
        DataFrame containing the initial assets data
    """
    required_columns = {
        "region",
        "technology",
    }
    data = read_csv(filename, required_columns=required_columns)
    return data


def process_initial_assets(data: pd.DataFrame) -> xr.DataArray:
    """Processes initial assets DataFrame into an xarray DataArray.

    Args:
        data: DataFrame containing the initial assets data

    Returns:
        xarray DataArray containing the processed initial assets
    """
    if "year" in data.columns:  # TODO: need a different way to identify trade file
        result = process_trade(data)
    else:
        result = process_initial_capacity(data)

    # Rename technology to asset
    technology = result.technology
    result = result.drop_vars("technology").rename(technology="asset")
    result["technology"] = "asset", technology.values

    # Add installed year
    result["installed"] = ("asset", [int(result.year.min())] * len(result.technology))
    return result


def process_initial_capacity(data: pd.DataFrame) -> xr.DataArray:
    """Processes initial capacity DataFrame into an xarray DataArray.

    Args:
        data: DataFrame containing the initial capacity data

    Returns:
        xarray DataArray containing the processed initial capacity
    """
    # Create multiindex for region, technology, and year
    data = create_multiindex(
        data,
        index_columns=["technology", "region"],
        index_names=["technology", "region"],
        drop_columns=True,
    )

    # Melt year columns into rows
    data = data.melt(var_name="year", value_name="value")
    data = data.set_index(["region", "technology", "year"])

    # Create DataArray
    result = create_xarray_dataarray(data["value"])
    return result


def read_technologies(
    technodata_path: Path,
    comm_out_path: Path,
    comm_in_path: Path,
    technodata_timeslices_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    df = read_technologies_csv(
        technodata_path, comm_out_path, comm_in_path, technodata_timeslices_path
    )
    return process_technologies(df)


def read_technologies_csv(
    technodata_path: Path,
    comm_out_path: Path,
    comm_in_path: Path,
    technodata_timeslices_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Reads data characterising technologies from files into DataFrames.

    Args:
        technodata_path: Path to the the technodata file.
        technodata_timeslices_path: This argument refers to the TechnodataTimeslices
            file which specifies the utilization factor per timeslice for the specified
            technology.
        comm_out_path: Refers to the path of the file specifying output commmodities.
        comm_in_path: Refers to the path of the file specifying input commmodities.

    Returns:
        Tuple of (technodata_df, comm_out_df, comm_in_df, technodata_timeslices_df)
        where technodata_timeslices_df may be None if not provided
    """
    tpath = technodata_path
    opath = comm_out_path
    ipath = comm_in_path

    msg = f"""Reading technology information from:
    - technodata: {tpath}
    - outputs: {opath}
    - inputs: {ipath}
    """
    if technodata_timeslices_path:
        ttpath = technodata_timeslices_path
        msg += f"""- technodata_timeslices: {ttpath}
        """
    else:
        ttpath = None

    getLogger(__name__).info(msg)

    # Read all data
    technodata_df = read_technodictionary_csv(tpath)
    comm_out_df = read_io_technodata_csv(opath)
    comm_in_df = read_io_technodata_csv(ipath)
    technodata_timeslices_df = (
        read_technodata_timeslices_csv(ttpath) if ttpath else None
    )

    return technodata_df, comm_out_df, comm_in_df, technodata_timeslices_df


def process_technologies(
    technodata_df: pd.DataFrame,
    comm_out_df: pd.DataFrame,
    comm_in_df: pd.DataFrame,
    technodata_timeslices_df: pd.DataFrame | None = None,
) -> xr.Dataset:
    """Processes technology data DataFrames into an xarray Dataset.

    Args:
        technodata_df: DataFrame containing technodata
        comm_out_df: DataFrame containing output commodities
        comm_in_df: DataFrame containing input commodities
        technodata_timeslices_df: Optional DataFrame containing technodata timeslices

    Returns:
        xarray Dataset containing the processed technology data
    """
    from muse.commodities import CommodityUsage

    # Process technodata
    result = process_technodictionary(technodata_df)
    if any(result[u].isnull().any() for u in result.data_vars):
        raise ValueError("Inconsistent data in technodata (e.g. inconsistent years)")

    # Process outputs
    outs = process_io_technodata(comm_out_df).rename(
        flexible="flexible_outputs", fixed="fixed_outputs"
    )
    if not (outs["flexible_outputs"] == 0).all():
        raise ValueError(
            "'flexible' outputs are not permitted. All outputs must be 'fixed'"
        )
    outs = outs.drop_vars("flexible_outputs")

    # Process inputs
    ins = process_io_technodata(comm_in_df).rename(
        flexible="flexible_inputs", fixed="fixed_inputs"
    )

    # Interpolate if needed
    if "year" in result.dims and len(result.year) > 1:
        if all(len(outs[d]) > 1 for d in outs.dims if outs[d].dtype.kind in "uifc"):
            outs = outs.interp(year=result.year)
        if all(len(ins[d]) > 1 for d in ins.dims if ins[d].dtype.kind in "uifc"):
            ins = ins.interp(year=result.year)

    try:
        result = result.merge(outs).merge(ins)
    except xr.core.merge.MergeError:
        raise UnitsConflictInCommodities

    # Process timeslices if provided
    if technodata_timeslices_df is not None:
        technodata_timeslice = process_technodata_timeslices(technodata_timeslices_df)
        result = result.drop_vars("utilization_factor")
        result = result.merge(technodata_timeslice)

    result["comm_usage"] = (
        "commodity",
        CommodityUsage.from_technologies(result).values,
    )
    result = result.set_coords("comm_usage")

    # Check UF and MSF
    # TODO: perform checks directly of CSVs instead
    check_utilization_and_minimum_service_factors(
        result.to_dataframe(), [technodata_df, technodata_timeslices_df]
    )

    return result


def read_global_commodities(path: Path) -> pd.DataFrame:
    df = read_global_commodities_csv(path)
    return process_global_commodities(df)


def read_global_commodities_csv(path: Path) -> pd.DataFrame:
    """Reads commodities information from input into a DataFrame.

    Args:
        path: Path to the global commodities CSV file

    Returns:
        DataFrame containing the global commodities data

    """
    required_columns = {
        "commodity",
        "comm_type",
    }
    data = read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading global commodities from {path}.",
    )
    return data


def process_global_commodities(data: pd.DataFrame) -> xr.Dataset:
    """Processes global commodities DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the global commodities data

    Returns:
        xarray Dataset containing the processed global commodities
    """
    data.index = [u for u in data.commodity]
    data = data.drop("commodity", axis=1)
    data.index.name = "commodity"
    return create_xarray_dataset(data)


def read_timeslice_shares(path: Path) -> pd.DataFrame:
    df = read_timeslice_shares_csv(path)
    return process_timeslice_shares(df)


def read_timeslice_shares_csv(path: Path) -> pd.DataFrame:
    """Reads sliceshare information into a DataFrame.

    Args:
        path: Path to the timeslice shares CSV file

    Returns:
        DataFrame containing the timeslice shares data
    """
    data = read_csv(
        path,
        required_columns=["region", "timeslice"],
        msg=f"Reading timeslice shares from {path}.",
    )
    return data


def process_timeslice_shares(data: pd.DataFrame) -> xr.DataArray:
    """Processes timeslice shares DataFrame into an xarray DataArray.

    Args:
        data: DataFrame containing the timeslice shares data

    Returns:
        xarray DataArray containing the processed timeslice shares
    """
    # Create multiindex for region and timeslice
    data = create_multiindex(
        data,
        index_columns=["region", "timeslice"],
        index_names=["region", "timeslice"],
        drop_columns=True,
    )

    # Set index and column names
    data.index.name = "rt"
    data.columns.name = "commodity"

    # Create DataArray and unstack
    result = create_xarray_dataarray(data)
    result = result.unstack("rt").to_dataset(name="shares")

    return result.shares


def read_agent_parameters(path: Path) -> pd.DataFrame:
    df = read_agent_parameters_csv(path)
    return process_agent_parameters(df, path)


def read_agent_parameters_csv(filename: Path) -> pd.DataFrame:
    """Reads standard MUSE agent-declaration csv-files into a DataFrame.

    Args:
        filename: Path to the agent parameters CSV file

    Returns:
        DataFrame with validated agent parameters
    """
    required_columns = {
        "search_rule",
        "quantity",
        "region",
        "type",
        "name",
        "agent_share",
        "decision_method",
    }
    data = read_csv(filename, required_columns=required_columns)

    # Check for deprecated retrofit agents
    if "type" in data.columns:
        retrofit_agents = data[data.type.str.lower().isin(["retrofit", "retro"])]
        if not retrofit_agents.empty:
            msg = (
                "Retrofit agents will be deprecated in a future release. "
                "Please modify your model to use only agents of the 'New' type."
            )
            getLogger(__name__).warning(msg)

    # Legacy: drop AgentNumber column
    if "agent_number" in data.columns:
        data = data.drop(["agent_number"], axis=1)

    # Check consistency of objectives data columns
    objectives = [col for col in data.columns if col.startswith("objective")]
    floats = [col for col in data.columns if col.startswith("obj_data")]
    sorting = [col for col in data.columns if col.startswith("objsort")]

    if len(objectives) != len(floats) or len(objectives) != len(sorting):
        raise ValueError(
            f"Agent Objective, ObjData, and Objsort columns are inconsistent in {filename}"  # noqa: E501
        )

    return data


def process_agent_parameters(data: pd.DataFrame, filename: Path) -> list[dict]:
    """Processes agent parameters DataFrame into a list of agent dictionaries.

    Args:
        data: DataFrame containing validated agent parameters
        filename: Path to the original CSV file (used for error messages)

    Returns:
        List of dictionaries, where each dictionary can be used to instantiate an
        agent in :py:func:`muse.agents.factories.factory`.
    """
    result = []
    for _, row in data.iterrows():
        # Get objectives data
        objectives = (
            row[[i.startswith("objective") for i in row.index]].dropna().to_list()
        )
        sorting = row[[i.startswith("obj_sort") for i in row.index]].dropna().to_list()
        floats = row[[i.startswith("obj_data") for i in row.index]].dropna().to_list()

        # Create decision parameters
        decision_params = list(zip(objectives, sorting, floats))

        agent_type = {
            "new": "newcapa",
            "newcapa": "newcapa",
            "retrofit": "retrofit",
            "retro": "retrofit",
            "agent": "agent",
            "default": "agent",
        }[getattr(row, "type", "agent").lower()]

        # Create agent data dictionary
        data = {
            "name": row.name,
            "region": row.region,
            "objectives": objectives,
            "search_rules": row.search_rule,
            "decision": {"name": row.decision_method, "parameters": decision_params},
            "agent_type": agent_type,
            "quantity": row.quantity,
            "share": row.agent_share,
        }

        # Add optional parameters
        if hasattr(row, "maturity_threshold"):
            data["maturity_threshold"] = row.maturity_threshold
        if hasattr(row, "spend_limit"):
            data["spend_limit"] = row.spend_limit

        # Add agent data to result
        result.append(data)

    return result


def read_macro_drivers(path: Path) -> pd.DataFrame:
    df = read_macro_drivers_csv(path)
    return process_macro_drivers(df)


def read_macro_drivers_csv(path: Path) -> pd.DataFrame:
    """Reads a standard MUSE csv file for macro drivers into a DataFrame.

    Args:
        path: Path to the macro drivers CSV file

    Returns:
        DataFrame containing the macro drivers data
    """
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


def process_macro_drivers(table: pd.DataFrame) -> xr.Dataset:
    """Processes macro drivers DataFrame into an xarray Dataset.

    Args:
        table: DataFrame containing the macro drivers data

    Returns:
        xarray Dataset containing the processed macro drivers
    """
    # Set index and column names
    table.index = table.region
    table.index.name = "region"
    table.columns.name = "year"

    # Drop unit and region columns
    table = table.drop(["unit", "region"], axis=1)

    # Split into population and GDP data
    population = table[table.variable == "Population"].drop("variable", axis=1)
    gdp = table[table.variable == "GDP|PPP"].drop("variable", axis=1)

    # Create dataset with standardized types
    result = create_xarray_dataset(
        data_vars={"gdp": gdp, "population": population},
        coords={
            "year": ("year", table.columns.values.astype(int)),
            "region": ("region", table.index.values.astype(str)),
        },
    )

    return result


def read_initial_market(
    projections: Path,
    base_year_import: Path | None = None,
    base_year_export: Path | None = None,
) -> xr.Dataset:
    df = read_initial_market_csv(projections, base_year_import, base_year_export)
    return process_initial_market(df)


def read_initial_market_csv(
    projections: Path,
    base_year_import: Path | None = None,
    base_year_export: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Read projections, import and export csv files into DataFrames.

    Args:
        projections: Path to projections CSV file
        base_year_import: Optional path to base year import CSV file
        base_year_export: Optional path to base year export CSV file

    Returns:
        Tuple of (projections_df, import_df, export_df) where import_df and export_df
        may be None if their respective files were not provided
    """
    # Projections must always be present
    required_columns = {
        "region",
        "attribute",
        "year",
    }
    projections_df = read_csv(
        projections,
        required_columns=required_columns,
        msg=f"Reading projections from {projections}.",
    )

    # Base year export is optional
    if base_year_export:
        export_df = read_csv(
            base_year_export,
            msg=f"Reading base year export from {base_year_export}.",
        )
    else:
        export_df = None

    # Base year import is optional
    if base_year_import:
        import_df = read_csv(
            base_year_import,
            msg=f"Reading base year import from {base_year_import}.",
        )
    else:
        import_df = None

    return projections_df, import_df, export_df


def process_initial_market(
    projections_df: pd.DataFrame,
    import_df: pd.DataFrame | None,
    export_df: pd.DataFrame | None,
) -> xr.Dataset:
    """Process market data DataFrames into an xarray Dataset.

    Args:
        projections_df: DataFrame containing projections data
        import_df: Optional DataFrame containing import data
        export_df: Optional DataFrame containing export data

    Returns:
        xarray Dataset containing processed market data
    """
    from muse.timeslices import TIMESLICE, distribute_timeslice

    # Process projections
    projections = process_attribute_table(projections_df)

    # Process optional trade data
    if export_df is not None:
        base_year_export = process_attribute_table(export_df)
    else:
        base_year_export = xr.zeros_like(projections)

    if import_df is not None:
        base_year_import = process_attribute_table(import_df)
    else:
        base_year_import = xr.zeros_like(projections)

    base_year_export = distribute_timeslice(base_year_export, level=None)
    base_year_import = distribute_timeslice(base_year_import, level=None)
    base_year_export.name = "exports"
    base_year_import.name = "imports"

    static_trade = base_year_import - base_year_export
    static_trade.name = "static_trade"

    # Assemble into xarray
    result = xr.Dataset(
        {
            projections.name: projections,
            base_year_export.name: base_year_export,
            base_year_import.name: base_year_import,
            static_trade.name: static_trade,
        }
    )

    # Expand prices over timeslices
    result["prices"] = (
        result["prices"].expand_dims({"timeslice": TIMESLICE}).drop_vars("timeslice")
    )

    return result


def read_attribute_table_csv(path: Path) -> pd.DataFrame:
    """Read a standard MUSE csv file for price projections into a DataFrame.

    Args:
        path: Path to the attribute table CSV file

    Returns:
        DataFrame containing the attribute table data
    """
    table = read_csv(
        path,
        required_columns=["region", "attribute", "year"],
        msg=f"Reading prices from {path}.",
    )
    return table


def process_attribute_table(table: pd.DataFrame) -> xr.DataArray:
    """Process attribute table DataFrame into an xarray DataArray.

    Args:
        table: DataFrame containing the attribute table data

    Returns:
        xarray DataArray containing the processed attribute table
    """
    # Set column names and standardize
    table.columns.name = "commodity"

    # Create multiindex for region and year
    table = create_multiindex(
        table,
        index_columns=["region", "year"],
        index_names=["region", "year"],
        drop_columns=True,
    )

    # Convert year to int
    table.index = table.index.set_levels(
        [table.index.levels[0], table.index.levels[1].astype(int)], level=[0, 1]
    )

    # Get attribute name and drop column
    attribute = table.attribute.unique()[0]
    table = table.drop(["attribute"], axis=1)

    # Create DataArray
    result = create_xarray_dataarray(table, name=attribute)

    # Fill missing values
    result = result.unstack("dim_0").fillna(0)

    return result


def read_regression_parameters(
    path: Path,
) -> xr.Dataset:
    df = read_regression_parameters_csv(path)
    return process_regression_parameters(df)


def read_regression_parameters_csv(
    path: Path,
) -> pd.DataFrame:
    """Reads the regression parameters from a standard MUSE csv file into a DataFrame.

    Args:
        path: Path to the regression parameters CSV file

    Returns:
        DataFrame containing the regression parameters data
    """
    table = read_csv(
        path,
        required_columns=["sector", "region", "function_type", "coeff"],
        msg=f"Reading regression parameters from {path}.",
    )
    return table


def process_regression_parameters(
    table: pd.DataFrame,
) -> xr.Dataset:
    """Processes regression parameters DataFrame into an xarray Dataset.

    Args:
        table: DataFrame containing the regression parameters data
        sector: Series of sector names
        function_type: Series of function types

    Returns:
        xarray Dataset containing the processed regression parameters
    """
    # Set column names
    table.columns.name = "commodity"

    # Create multiindex for sector and region
    sector = table.sector
    function_type = table.function_type

    table = create_multiindex(
        table.drop(["sector", "region", "function_type"], axis=1),
        index_columns=["sector", "region"],
        index_names=["sector", "region"],
        drop_columns=True,
    )

    # Create dataset with coefficients
    coeffs = create_xarray_dataset(
        {
            k: xr.DataArray(table[table.coeff == k].drop("coeff", axis=1))
            for k in table.coeff.unique()
        }
    )

    # Unstack multiindex and fill missing values
    coeffs = coeffs.unstack("dim_0").fillna(0)

    # Add function type coordinate
    function_type = list(zip(*set(zip(sector, function_type))))
    coeffs["function_type"] = xr.DataArray(
        list(function_type[1]),
        dims=["sector"],
        coords={"sector": list(function_type[0])},
    )

    return coeffs


def read_presets(paths: Path) -> dict[int, pd.DataFrame]:
    df = read_presets_csv(paths)
    return process_presets(df)


def read_presets_csv(paths: Path) -> dict[int, pd.DataFrame]:
    """Read consumption or supply files for preset sectors into DataFrames.

    Args:
        paths: Path pattern to match preset files

    Returns:
        Dictionary mapping years to DataFrames containing preset data

    """
    from glob import glob
    from re import match

    allfiles = [Path(p) for p in glob(str(paths))]
    if len(allfiles) == 0:
        raise OSError(f"No files found with paths {paths}")

    datas = {}
    for path in allfiles:
        data = read_csv(path, required_columns=["region", "timeslice"])

        reyear = match(r"\S*.(\d{4})\S*\.csv", path.name)
        if reyear is None:
            raise OSError(f"Unexpected filename {path.name}")
        year = int(reyear.group(1))
        if year in datas:
            raise OSError(f"Year f{year} was found twice")
        data.year = year

        # Legacy: drop ProcessName column and sum data (PR #448)
        if "process" in data.columns:
            getLogger(__name__).warning(
                f"The ProcessName column (in file {path}) is deprecated. "
                "Data has been summed across processes, and this column has been "
                "dropped."
            )
            data = (
                data.drop(columns=["process"])
                .groupby(["region", "timeslice"])
                .sum()
                .reset_index()
            )

        datas[year] = data

    return datas


def process_presets(
    datas: dict[int, pd.DataFrame],
) -> xr.Dataset:
    """Processes preset DataFrames into an xarray Dataset.

    Args:
        datas: Dictionary mapping years to DataFrames containing preset data

    Returns:
        xarray Dataset containing the processed preset data
    """
    processed_datas = {}
    for year, data in datas.items():
        # Create multiindex
        data = create_multiindex(
            data,
            index_columns=["region", "timeslice"],
            index_names=["asset"],
            drop_columns=True,
        )

        # Set column names
        data.columns.name = "commodity"

        # Create DataArray
        processed_datas[year] = create_xarray_dataarray(data)

    # Combine into dataset
    result = (
        xr.Dataset(processed_datas)
        .to_array(dim="year")
        .sortby("year")
        .fillna(0)
        .unstack("asset")
    )

    return result


def process_trade(data: pd.DataFrame) -> xr.DataArray | xr.Dataset:
    """Processes trade DataFrame into an xarray DataArray or Dataset.

    Args:
        data: DataFrame containing the trade data

    Returns:
        xarray DataArray or Dataset containing the processed trade data
    """
    col_region = "src_region"
    row_region = "dst_region"

    # Standardize column names
    data = data.rename({"region": row_region})

    # Get indices for melting
    indices = list(
        {"commodity", "year", "src_region", "dst_region", "technology"}.intersection(
            data.columns
        )
    )

    # Melt data
    data = data.melt(id_vars=indices, var_name=col_region)

    # Create result based on parameters
    result = create_xarray_dataarray(data.set_index([*indices, col_region])["value"])

    return result.rename(src_region="region")


def check_utilization_and_minimum_service_factors(
    data: pd.DataFrame, filename: Path | list[Path]
) -> None:
    filename = [filename] if isinstance(filename, Path) else filename
    filename = [name for name in filename if name is not None]
    if "utilization_factor" not in data.columns:
        raise ValueError(
            f"""A technology needs to have a utilization factor defined for every
             timeslice. Please check files: {filename}."""
        )

    # Check UF not all zero
    utilization_sum = data.groupby(["technology", "region", "year"]).sum()
    if (utilization_sum.utilization_factor == 0).any():
        raise ValueError(
            f"""A technology can not have a utilization factor of 0 for every
                timeslice. Please check files: {filename}."""
        )

    # Check UF in range
    utilization = data["utilization_factor"]
    if not np.all((0 <= utilization) & (utilization <= 1)):
        raise ValueError(
            f"""Utilization factor values must all be between 0 and 1 inclusive.
            Please check files: {filename}."""
        )

    if "minimum_service_factor" in data.columns:
        # Check MSF in range
        min_service_factor = data["minimum_service_factor"]
        if not np.all((0 <= min_service_factor) & (min_service_factor <= 1)):
            raise ValueError(
                f"""Minimum service factor values must all be between 0 and 1 inclusive.
                Please check files: {filename}."""
            )

        # Check UF not below MSF
        if (data["utilization_factor"] < data["minimum_service_factor"]).any():
            raise ValueError(f"""Utilization factors must all be greater than or equal
                        to their corresponding minimum service factors. Please check
                        {filename}.""")
