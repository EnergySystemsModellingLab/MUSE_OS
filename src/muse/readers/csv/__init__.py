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

The code in this module is spread over multiple files. In general, we have one `read_x`
function per file, and as many `read_x_csv` and `process_x` functions as are required
(e.g. if a dataset is assembled from three csv files we will have three `read_x_csv`
functions, and potentially multiple `process_x` functions).

Most of the processing is shared by a few helper functions (in `helpers.py`):
- `read_csv`: reads a csv file and returns a dataframe
- `standardize_dataframe`: standardizes the dataframe to a common format
- `create_multiindex`: creates a multiindex from a dataframe
- `create_xarray_dataset`: creates an xarray dataset from a dataframe

A few other helpers perform common operations on xarrays:
- `create_assets`: creates assets from technologies
- `check_commodities`: checks commodities and fills missing values

"""

from .agents import read_agents
from .assets import read_assets
from .commodities import read_commodities
from .correlation_consumption import read_correlation_consumption
from .general import read_attribute_table
from .helpers import read_csv
from .market import read_initial_market
from .presets import read_presets
from .technologies import read_technologies
from .trade_assets import read_trade_assets
from .trade_technodata import read_trade_technodata

__all__ = [
    "read_agents",
    "read_assets",
    "read_attribute_table",
    "read_commodities",
    "read_correlation_consumption",
    "read_csv",
    "read_initial_market",
    "read_presets",
    "read_technologies",
    "read_trade_assets",
    "read_trade_technodata",
]
