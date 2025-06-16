"""Ensemble of functions to read MUSE data."""

from __future__ import annotations

__all__ = [
    "read_agent_parameters",
    "read_attribute_table",
    "read_existing_trade",
    "read_global_commodities",
    "read_initial_capacity",
    "read_initial_market",
    "read_io_technodata",
    "read_macro_drivers",
    "read_presets",
    "read_regression_parameters",
    "read_technodata_timeslices",
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
    "commodity_price": "prices",
    "units_commodity_price": "units_prices",
    "enduse": "end_use",
    "sn": "timeslice",
    "commodity_emission_factor_CO2": "emmission_factor",
}

# Columns who's values should be converted from camelCase to snake_case
CAMEL_TO_SNAKE_COLUMNS = [
    "tech_type",
    "commodity",
    "comm_type",
    "agent_share",
    "attribute",
    "sector",
    "region",
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
    "timeslice": int,  # Some tables require int timeslice instead of month, day etc.
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
    "coeff": str,
    "value": float,
    "utilization_factor": float,
    "minimum_service_factor": float,
    "maturity_threshold": float,
    "spend_limit": float,
    "prices": float,
    "emmission_factor": float,
}


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names in a DataFrame.

    This function:
    1. Converts column names to snake_case
    2. Applies the global COLUMN_RENAMES mapping
    3. Preserves any columns not in the mapping

    Args:
        data: DataFrame to standardize

    Returns:
        DataFrame with standardized column names
    """
    # First convert to snake_case
    data = data.rename(columns=camel_to_snake)

    # Drop any columns that start with "Unname"
    data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

    # Then apply global mapping
    data = data.rename(columns=COLUMN_RENAMES)

    # Make sure there are no duplicate columns
    if len(data.columns) != len(set(data.columns)):
        raise ValueError(f"Duplicate columns in {data.columns}")

    return data


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


def camel_to_snake(name: str) -> str:
    """Transforms CamelCase to snake_case."""
    from re import sub

    pattern = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    result = sub("([a-z0-9])([A-Z])", r"\1_\2", pattern).lower()
    result = result.replace("co2", "CO2")
    result = result.replace("ch4", "CH4")
    result = result.replace("n2_o", "N2O")
    result = result.replace("f-gases", "F-gases")
    return result


def convert_column_types(data: pd.DataFrame) -> pd.DataFrame:
    """Converts DataFrame columns to their expected types.

    Args:
        data: DataFrame to convert

    Returns:
        DataFrame with converted column types
    """
    result = data.copy()
    for column, expected_type in COLUMN_TYPES.items():
        if column in result.columns:
            try:
                if expected_type is int:
                    result[column] = pd.to_numeric(result[column], downcast="integer")
                elif expected_type is float:
                    result[column] = pd.to_numeric(result[column])
                elif expected_type is str:
                    result[column] = result[column].astype(str)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Could not convert column '{column}' to {expected_type.__name__}: {e}"  # noqa: E501
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
    path: Path | pd.DataFrame,
    float_precision: str = "high",
    required_columns: list[str] | None = None,
    msg: str | None = None,
) -> pd.DataFrame:
    """Reads and standardizes a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file
        float_precision: Precision to use when reading floats
        required_columns: List of column names that must be present (optional)
        msg: Message to log (optional)

    Returns:
        DataFrame containing the standardized data
    """
    # Log message
    if msg:
        getLogger(__name__).info(msg)

    assert isinstance(path, (Path, pd.DataFrame)), "Only accepts Path or DataFrame"

    # If a Path is passed, read the file
    if isinstance(path, Path):
        # Check if file exists
        if not path.is_file():
            raise OSError(f"{path} does not exist.")

        # Check if there's a units row (in which case we need to skip it)
        with open(path) as f:
            next(f)  # Skip header row
            first_data_row = f.readline().strip()
        skiprows = [1] if first_data_row.startswith("Unit") else None

        # Read the file
        data = pd.read_csv(
            path,
            float_precision=float_precision,
            low_memory=False,
            skiprows=skiprows,
        )
    else:  # Must be DataFrame
        data = path

    assert isinstance(data, pd.DataFrame)

    # Standardize column names
    data = standardize_columns(data)

    # Convert specified column values from camelCase to snake_case
    for col in CAMEL_TO_SNAKE_COLUMNS:
        if col in data.columns:
            data[col] = data[col].apply(camel_to_snake)

    # Check/convert data types
    data = convert_column_types(data)

    # Validate required columns if provided
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    return data


def read_technodictionary(path: Path) -> xr.Dataset:
    df = read_technodictionary_csv(path)
    return process_technodictionary(df)


def read_technodictionary_csv(path: Path) -> pd.DataFrame:
    """Reads and formats technodata into a DataFrame.

    Args:
        path: Path to the technodictionary CSV file

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
    data = read_csv(path, required_columns=required_columns)

    # Check for deprecated columns
    if "fuel" in data.columns:
        msg = (
            f"The 'Fuel' column in {path} has been deprecated. "
            "This information is now determined from CommIn files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)
    if "end_use" in data.columns:
        msg = (
            f"The 'EndUse' column in {path} has been deprecated. "
            "This information is now determined from CommOut files. "
            "Please remove this column from your Technodata files."
        )
        getLogger(__name__).warning(msg)

    return data


def process_technodictionary(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodictionary DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the technodictionary data

    Returns:
        xarray Dataset containing the processed technodictionary
    """
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

    # Sanity checks for year dimension
    if "year" in result.dims:
        assert len(set(result.year.data)) == result.year.data.size
        result = result.sortby("year")
        if len(result.year) == 1:
            result = result.isel(year=0, drop=True)

    # TODO: what is this?
    if any(result[u].isnull().any() for u in result.data_vars):
        raise ValueError("Inconsistent data in technodata (e.g. inconsistent years)")

    return result


def read_technodata_timeslices(path: Path) -> xr.Dataset:
    df = read_technodata_timeslices_csv(path)
    return process_technodata_timeslices(df)


def read_technodata_timeslices_csv(path: Path) -> pd.DataFrame:
    """Reads and formats technodata timeslices into a DataFrame.

    Args:
        path: Path to the technodata timeslices CSV file

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
    return read_csv(path, required_columns=required_columns)


def process_technodata_timeslices(data: pd.DataFrame) -> xr.Dataset:
    """Processes technodata timeslices DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the technodata timeslices data

    Returns:
        xarray Dataset containing the processed technodata timeslices
    """
    from muse.timeslices import sort_timeslices

    # Create multiindex for all columns except factor columns (i.e. timeslice columns)
    # This has to be dynamic because timeslice columns can be different for each model
    # TODO: is there a better way to do this?
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

    # Convert year to int64 (from int16) and sort
    # TODO: why is year int16 in the first place?
    result = result.assign_coords(year=result.year.astype(int))
    result = result.sortby("year")

    # Stack timeslice levels (month, day, hour) into a single timeslice dimension
    timeslice_levels = ["month", "day", "hour"]
    if all(level in result.dims for level in timeslice_levels):
        result = result.stack(timeslice=timeslice_levels)

    return sort_timeslices(result)


def read_io_technodata(path: Path) -> xr.Dataset:
    df = read_io_technodata_csv(path)
    return process_io_technodata(df)


def read_io_technodata_csv(path: Path) -> pd.DataFrame:
    """Reads process inputs or outputs into a DataFrame.

    Args:
        path: Path to the IO technodata CSV file

    Returns:
        DataFrame containing the IO technodata
    """
    data = read_csv(path, required_columns=["technology", "region", "year"])

    # Unspecified Level values default to "fixed"
    if "level" in data.columns:
        data["level"] = data["level"].fillna("fixed")
    else:
        # Particularly relevant to outputs files where the Level column is omitted by
        # default, as only "fixed" outputs are allowed.
        data["level"] = "fixed"

    return data


def process_io_technodata(data: pd.DataFrame) -> xr.Dataset:
    """Processes IO technodata DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the IO technodata

    Returns:
        xarray Dataset containing the processed IO technodata
    """
    # Extract commodity columns
    # TODO: a bit hacky as the user may include extra columns aside from commodities
    commodities = [
        col
        for col in data.columns
        if col not in ["technology", "region", "year", "level"]
    ]

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

    # Convert year to int64
    result = result.assign_coords(year=result.year.astype(int))
    result = result.sortby("year")

    return result


def read_technologies(
    technodata_path: Path,
    comm_out_path: Path,
    comm_in_path: Path,
    commodities: xr.Dataset,
    technodata_timeslices_path: Path | None = None,
) -> xr.Dataset:
    # Log message
    msg = f"""Reading technology information from:
    - technodata: {technodata_path}
    - outputs: {comm_out_path}
    - inputs: {comm_in_path}
    """
    if technodata_timeslices_path:
        msg += f"""- technodata_timeslices: {technodata_timeslices_path}
        """
    getLogger(__name__).info(msg)

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
        technodata, comm_out, comm_in, technodata_timeslices, commodities
    )


def process_technologies(
    technodata: xr.Dataset,
    comm_out: xr.Dataset,
    comm_in: xr.Dataset,
    technodata_timeslices: xr.Dataset | None = None,
    commodities: xr.Dataset = None,
) -> xr.Dataset:
    """Processes technology data DataFrames into an xarray Dataset.

    Args:
        technodata: xarray Dataset containing technodata
        comm_out: xarray Dataset containing output commodities
        comm_in: xarray Dataset containing input commodities
        technodata_timeslices: Optional xarray Dataset containing technodata timeslices
        commodities: xarray Dataset containing commodities

    Returns:
        xarray Dataset containing the processed technology data
    """
    from muse.commodities import CommodityUsage

    # Process inputs/outputs
    ins = comm_in.rename(flexible="flexible_inputs", fixed="fixed_inputs")
    outs = comm_out.rename(flexible="flexible_outputs", fixed="fixed_outputs")

    # Legacy: Remove flexible outputs
    if not (outs["flexible_outputs"] == 0).all():
        raise ValueError(
            "'flexible' outputs are not permitted. All outputs must be 'fixed'"
        )
    outs = outs.drop_vars("flexible_outputs")

    # Interpolate inputs/outputs if needed
    if "year" in technodata.dims and len(technodata.year) > 1:
        outs = outs.interp(year=technodata.year)
        ins = ins.interp(year=technodata.year)

    # Merge inputs/outputs with technodata
    try:
        technodata = technodata.merge(outs).merge(ins)
    except xr.core.merge.MergeError:
        # TODO: what is this?
        raise UnitsConflictInCommodities

    # Process timeslices if provided
    if technodata_timeslices:
        technodata = technodata.drop_vars("utilization_factor")
        technodata = technodata.merge(technodata_timeslices)

    # Add info about commodities
    if isinstance(commodities, xr.Dataset):
        if technodata.commodity.isin(commodities.commodity).all():
            technodata = technodata.merge(
                commodities.sel(commodity=technodata.commodity)
            )
        else:
            raise OSError("Commodities not found in global commodities file")

    # Add commodity usage flags
    technodata["comm_usage"] = (
        "commodity",
        CommodityUsage.from_technologies(technodata).values,
    )
    technodata = technodata.set_coords("comm_usage")
    technodata = technodata.drop_vars("comm_type")

    # TODO: Check UF and MSF

    return technodata


def read_initial_capacity(path: Path) -> xr.DataArray:
    df = read_initial_capacity_csv(path)
    return process_initial_capacity(df)


def read_initial_capacity_csv(path: Path) -> pd.DataFrame:
    """Reads and formats data about initial capacity into a DataFrame.

    Args:
        path: Path to the initial assets CSV file

    Returns:
        DataFrame containing the initial assets data
    """
    required_columns = {
        "region",
        "technology",
    }
    return read_csv(path, required_columns=required_columns)


def process_initial_capacity(data: pd.DataFrame) -> xr.DataArray:
    """Processes initial capacity DataFrame into an xarray DataArray.

    Args:
        data: DataFrame containing the initial capacity data

    Returns:
        xarray DataArray containing the processed initial capacity
    """
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

    # Convert year column to int64
    data["year"] = data["year"].astype(int)

    # Create multiindex for region, technology, and year
    data = create_multiindex(
        data,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    # Create Dataarray
    result = create_xarray_dataset(data).value

    # Rename technology to asset
    technology = result.technology
    result = result.drop_vars("technology").rename(technology="asset")
    result["technology"] = "asset", technology.values

    # Add installed year
    result["installed"] = ("asset", [int(result.year.min())] * len(result.technology))
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
    # Due to legacy reasons, users can supply both Commodity and CommodityName columns
    # In this case, we need to remove the Commodity column to avoid conflicts
    # This is fine because Commodity just contains a long description that isn't needed
    df = pd.read_csv(path)
    df = df.rename(columns=camel_to_snake)
    if "commodity" in df.columns and "commodity_name" in df.columns:
        df = df.drop(columns=["commodity"])

    required_columns = {
        "commodity",
        "comm_type",
    }
    data = read_csv(
        df,
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


def read_agent_parameters(path: Path) -> pd.DataFrame:
    df = read_agent_parameters_csv(path)
    return process_agent_parameters(df)


def read_agent_parameters_csv(path: Path) -> pd.DataFrame:
    """Reads standard MUSE agent-declaration csv-files into a DataFrame.

    Args:
        path: Path to the agent parameters CSV file

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
    data = read_csv(path, required_columns=required_columns)

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
            f"Agent Objective, ObjData, and Objsort columns are inconsistent in {path}"
        )

    return data


def process_agent_parameters(data: pd.DataFrame) -> list[dict]:
    """Processes agent parameters DataFrame into a list of agent dictionaries.

    Args:
        data: DataFrame containing validated agent parameters

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
        sorting = row[[i.startswith("objsort") for i in row.index]].dropna().to_list()
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
            "name": row["name"],
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


def read_initial_market(
    projections: Path,
    base_year_import: Path | None = None,
    base_year_export: Path | None = None,
) -> xr.Dataset:
    projections_df = read_projections_csv(projections)

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

    # Assemble into xarray Dataset
    return process_initial_market(projections_df, import_df, export_df)


def read_projections_csv(path: Path) -> pd.DataFrame:
    required_columns = {
        "region",
        "attribute",
        "year",
    }
    projections_df = read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading projections from {path}.",
    )
    return projections_df


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
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    # Process projections
    projections = process_attribute_table(projections_df).commodity_price

    # Process optional trade data
    if export_df is not None:
        base_year_export = process_attribute_table(export_df).exports
    else:
        base_year_export = xr.zeros_like(projections)

    if import_df is not None:
        base_year_import = process_attribute_table(import_df).imports
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

    return result


def read_attribute_table(path: Path) -> xr.DataArray:
    df = read_attribute_table_csv(path)
    return process_attribute_table(df)


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


def process_attribute_table(data: pd.DataFrame) -> xr.Dataset:
    """Process attribute table DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the attribute table data

    Returns:
        xarray Dataset containing the processed attribute table
    """
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


def read_presets(paths: Path) -> xr.Dataset:
    from glob import glob
    from re import match

    # Find all files matching the path pattern
    allfiles = [Path(p) for p in glob(str(paths))]
    if len(allfiles) == 0:
        raise OSError(f"No files found with paths {paths}")

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
    data = read_csv(path, required_columns=["region", "timeslice"])

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

    return data


def process_presets(datas: dict[int, pd.DataFrame]) -> xr.Dataset:
    """Processes preset DataFrames into an xarray Dataset.

    Args:
        datas: Dictionary mapping years to DataFrames containing preset data

    Returns:
        xarray Dataset containing the processed preset data
    """
    # Combine into a single DataFrame
    data = pd.concat(datas.values())

    # Extract commodity columns
    commodities = [
        col for col in data.columns if col not in ["region", "year", "timeslice"]
    ]

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
    return create_xarray_dataset(data).value


def read_trade_technodata(path: Path) -> xr.DataArray:
    df = read_trade_technodata_csv(path)
    return process_trade_technodata(df)


def read_trade_technodata_csv(path: Path) -> pd.DataFrame:
    required_columns = {"technology", "region", "parameter"}
    return read_csv(path, required_columns=required_columns)


def process_trade_technodata(data: pd.DataFrame) -> xr.DataArray:
    # Drop unit column if present
    if "unit" in data.columns:
        data = data.drop(columns=["unit"])

    # Select region columns
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

    # Convert CamelCase to snake_case
    data = standardize_columns(data)

    # TODO: Make sure no nan values

    # Create DataSet
    return create_xarray_dataset(data)


def read_existing_trade(path: Path) -> xr.DataArray:
    df = read_existing_trade_csv(path)
    return process_existing_trade(df)


def read_existing_trade_csv(path: Path) -> pd.DataFrame:
    required_columns = {
        "region",
        "technology",
        "year",
    }
    return read_csv(path, required_columns=required_columns)


def process_existing_trade(data: pd.DataFrame) -> xr.DataArray:
    # Select region columns
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
    result = create_xarray_dataset(data).value
    return result


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
    # Extract commodity columns
    commodities = [col for col in data.columns if col not in ["timeslice", "region"]]

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

    # Create DataSet
    result = create_xarray_dataset(data).value
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


def process_macro_drivers(data: pd.DataFrame) -> xr.Dataset:
    """Processes macro drivers DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the macro drivers data

    Returns:
        xarray Dataset containing the processed macro drivers
    """
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

    # Convert year column to int64
    data["year"] = data["year"].astype(int)

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
    df = read_regression_parameters_csv(path)
    return process_regression_parameters(df)


def read_regression_parameters_csv(path: Path) -> pd.DataFrame:
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


def process_regression_parameters(data: pd.DataFrame) -> xr.Dataset:
    """Processes regression parameters DataFrame into an xarray Dataset.

    Args:
        data: DataFrame containing the regression parameters data

    Returns:
        xarray Dataset containing the processed regression parameters
    """
    # Extract commodity columns
    commodities = [
        col
        for col in data.columns
        if col not in ["sector", "region", "function_type", "coeff"]
    ]

    # Convert commodity columns to long format (i.e. single "commodity" column)
    data = data.melt(
        id_vars=["sector", "region", "function_type", "coeff"],
        value_vars=commodities,
        var_name="commodity",
        value_name="value",
    )

    # Pivot over coeff
    data = data.pivot(
        index=["sector", "region", "commodity", "function_type"],
        columns="coeff",
        values="value",
    )

    # Remove function_type from multiindex
    data = data.reset_index(level="function_type")
    # TODO: function_type may have to have sector dimension only

    # Convert to Dataset
    return create_xarray_dataset(data)


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
