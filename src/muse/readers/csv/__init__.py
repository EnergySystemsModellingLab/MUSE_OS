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

from .agents import read_agent_parameters
from .assets import read_initial_capacity
from .commodities import read_global_commodities
from .general import read_attribute_table
from .helpers import read_csv
from .market import read_initial_market
from .presets import read_presets
from .regression import (
    read_macro_drivers,
    read_regression_parameters,
    read_timeslice_shares,
)
from .technologies import (
    read_io_technodata,
    read_technodata_timeslices,
    read_technodictionary,
    read_technologies,
)
from .trade import read_existing_trade, read_trade_technodata

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
