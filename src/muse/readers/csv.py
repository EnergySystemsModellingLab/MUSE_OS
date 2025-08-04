"""Ensemble of functions to read MUSE data.

In general, there are three functions per input file:
`read_x`: This is the overall function that is called to read the data. It takes a
    `Path` as input, and returns the relevant data structure (usually an xarray). The
    process is generally broken down into two functions that are called by `read_x`:

`read_x_csv`: This takes a path to a csv file as input and returns a pandas dataframe.
    There are some consistency checks, such as checking data types and columns. There
    is also some minor processing at this stage, such as standardising column names,
    but no structural changes to the data. The general rule is that anything returned
    by this function should still be valid as an input file if saved to csv.
`process_x`: This is where more major processing and reformatting of the data is done.
    It takes the dataframe from `read_x_csv` and returns the final data structure
    (usually an xarray). There are also some more checks (e.g. checking for nan
    values).

Most of the processing is shared by a few helper functions:
- read_csv: reads a csv file and returns a dataframe
- standardize_dataframe: standardizes the dataframe to a common format
- create_multiindex: creates a multiindex from a dataframe
- create_xarray_dataset: creates an xarray dataset from a dataframe

A few other helpers perform common operations on xarrays:
- create_assets: creates assets from technologies
- check_commodities: checks commodities and fills missing values

"""

from __future__ import annotations

__all__ = [
    "read_agent_parameters",
    "read_attribute_table",
    "read_csv",
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
    "read_trade_technodata",
]

from logging import getLogger
from pathlib import Path

import pandas as pd
import xarray as xr

from muse.utilities import camel_to_snake

# Global mapping of column names to their standardized versions
# This is for backwards compatibility with old file formats
COLUMN_RENAMES = {
    "process_name": "technology",
    "process": "technology",
    "sector_name": "sector",
    "region_name": "region",
    "time": "year",
    "commodity_name": "commodity",
    "comm_type": "commodity_type",
    "commodity_price": "prices",
    "units_commodity_price": "units_prices",
    "enduse": "end_use",
    "sn": "timeslice",
    "commodity_emission_factor_CO2": "emmission_factor",
    "utilisation_factor": "utilization_factor",
    "objsort": "obj_sort",
    "objsort1": "obj_sort1",
    "objsort2": "obj_sort2",
    "objsort3": "obj_sort3",
    "time_slice": "timeslice",
    "price": "prices",
}

# Columns who's values should be converted from camelCase to snake_case
CAMEL_TO_SNAKE_COLUMNS = [
    "tech_type",
    "commodity",
    "commodity_type",
    "agent_share",
    "attribute",
    "sector",
    "region",
    "parameter",
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
    "timeslice": int,  # For tables that require int timeslice instead of month etc.
    "name": str,
    "commodity_type": str,
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

DEFAULTS = {
    "cap_par": 0,
    "cap_exp": 1,
    "fix_par": 0,
    "fix_exp": 1,
    "var_par": 0,
    "var_exp": 1,
    "interest_rate": 0,
    "utilization_factor": 1,
    "minimum_service_factor": 0,
    "search_rule": "all",
    "decision_method": "single",
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
    # Drop index column if present
    if data.columns[0] == "" or data.columns[0].startswith("Unnamed"):
        data = data.iloc[:, 1:]

    # Convert columns to snake_case
    data = data.rename(columns=camel_to_snake)

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
    disallow_nan: bool = True,
) -> xr.Dataset:
    """Creates an xarray Dataset from a DataFrame with standardized options.

    Args:
        data: DataFrame to convert
        disallow_nan: Whether to raise an error if NaN values are found

    Returns:
        xarray Dataset
    """
    result = xr.Dataset.from_dataframe(data)
    if disallow_nan:
        nan_coords = get_nan_coordinates(result)
        if nan_coords:
            raise ValueError(f"Missing data for coordinates: {nan_coords}")

    if "year" in result.coords:
        result = result.assign_coords(year=result.year.astype(int))
        result = result.sortby("year")
        assert len(set(result.year.values)) == result.year.data.size  # no duplicates

    return result


def get_nan_coordinates(dataset: xr.Dataset) -> list[tuple]:
    """Get coordinates of a Dataset where any data variable has NaN values."""
    any_nan = sum(var.isnull() for var in dataset.data_vars.values())
    if any_nan.any():
        return any_nan.where(any_nan, drop=True).to_dataframe(name="").index.to_list()
    return []


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
                    result[column] = pd.to_numeric(result[column]).astype(float)
                elif expected_type is str:
                    result[column] = result[column].astype(str)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Could not convert column '{column}' to {expected_type.__name__}: {e}"  # noqa: E501
                )
    return result


def standardize_dataframe(
    data: pd.DataFrame,
    required_columns: list[str] | None = None,
    exclude_extra_columns: bool = False,
) -> pd.DataFrame:
    """Standardizes a DataFrame to a common format.

    Args:
        data: DataFrame to standardize
        required_columns: List of column names that must be present (optional)
        exclude_extra_columns: If True, exclude any columns not in required_columns list
            (optional). This can be important if extra columns can mess up the resulting
            xarray object.

    Returns:
        DataFrame containing the standardized data
    """
    if required_columns is None:
        required_columns = []

    # Standardize column names
    data = standardize_columns(data)

    # Convert specified column values from camelCase to snake_case
    for col in CAMEL_TO_SNAKE_COLUMNS:
        if col in data.columns:
            data[col] = data[col].apply(camel_to_snake)

    # Fill missing values with defaults
    data = data.fillna(DEFAULTS)
    for col, default in DEFAULTS.items():
        if col not in data.columns and col in required_columns:
            data[col] = default

    # Check/convert data types
    data = convert_column_types(data)

    # Validate required columns if provided
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Exclude extra columns if requested
        if exclude_extra_columns:
            data = data[list(required_columns)]

    return data


def read_csv(
    path: Path,
    float_precision: str = "high",
    required_columns: list[str] | None = None,
    exclude_extra_columns: bool = False,
    msg: str | None = None,
) -> pd.DataFrame:
    """Reads and standardizes a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file
        float_precision: Precision to use when reading floats
        required_columns: List of column names that must be present (optional)
        exclude_extra_columns: If True, exclude any columns not in required_columns list
            (optional). This can be important if extra columns can mess up the resulting
            xarray object.
        msg: Message to log (optional)

    Returns:
        DataFrame containing the standardized data
    """
    # Log message
    if msg:
        getLogger(__name__).info(msg)

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

    # Standardize the DataFrame
    return standardize_dataframe(
        data,
        required_columns=required_columns,
        exclude_extra_columns=exclude_extra_columns,
    )


def check_commodities(
    data: xr.Dataset | xr.DataArray, fill_missing: bool = True, fill_value: float = 0
) -> xr.Dataset | xr.DataArray:
    """Validates and optionally fills missing commodities in data."""
    from muse.commodities import COMMODITIES

    # Make sure there are no commodities in data but not in global commodities
    extra_commodities = [
        c for c in data.commodity.values if c not in COMMODITIES.commodity.values
    ]
    if extra_commodities:
        raise ValueError(
            "The following commodities were not found in global commodities file: "
            f"{extra_commodities}"
        )

    # Add any missing commodities with fill_value
    if fill_missing:
        data = data.reindex(
            commodity=COMMODITIES.commodity.values, fill_value=fill_value
        )
    return data


def create_assets(data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Creates assets from technology data."""
    # Rename technology to asset
    result = data.drop_vars("technology").rename(technology="asset")
    result["technology"] = "asset", data.technology.values

    # Add installed year
    result["installed"] = ("asset", [int(result.year.min())] * len(result.technology))
    return result


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


def read_initial_capacity(path: Path) -> xr.DataArray:
    """Reads and processes initial capacity data from a CSV file."""
    df = read_initial_capacity_csv(path)
    return process_initial_capacity(df)


def read_initial_capacity_csv(path: Path) -> pd.DataFrame:
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


def process_initial_capacity(data: pd.DataFrame) -> xr.DataArray:
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


def read_agent_parameters(path: Path) -> pd.DataFrame:
    """Reads and processes agent parameters from a CSV file."""
    df = read_agent_parameters_csv(path)
    return process_agent_parameters(df)


def read_agent_parameters_csv(path: Path) -> pd.DataFrame:
    """Reads standard MUSE agent-declaration csv-files into a DataFrame."""
    required_columns = {
        "search_rule",
        "quantity",
        "region",
        "type",
        "name",
        "agent_share",
        "decision_method",
    }
    data = read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading agent parameters from {path}.",
    )

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
    sorting = [col for col in data.columns if col.startswith("obj_sort")]

    if len(objectives) != len(floats) or len(objectives) != len(sorting):
        raise ValueError(
            "Agent objective, obj_data, and obj_sort columns are inconsistent in "
            f"{path}"
        )

    return data


def process_agent_parameters(data: pd.DataFrame) -> list[dict]:
    """Processes agent parameters DataFrame into a list of agent dictionaries."""
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
