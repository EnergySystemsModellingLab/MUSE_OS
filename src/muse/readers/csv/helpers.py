from __future__ import annotations

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
