"""Output quantities.

Functions that compute sectorial quantities for post-simulation analysis should all
follow the same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
<<<<<<< HEAD
        capacity: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset
    ) -> Union[xr.DataArray, DataFrame]:
=======
        capacity: DataArray,
        market: Dataset,
        technologies: Dataset
    ) -> Union[DataArray, DataFrame]:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        pass

They take as input the current capacity profile, aggregated across a sectoar,
a dataset containing market-related quantities, and a dataset characterizing the
<<<<<<< HEAD
technologies in the market. It returns a single xr.DataArray object.

The function should never modify it's arguments.
"""
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Union

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg
=======
technologies in the market. It returns a single DataArray object.

The function should never modify it's arguments.
"""
from typing import Any, Callable, List, Mapping, Optional, Text, Union

from mypy_extensions import KwArg
from xarray import DataArray, Dataset
import pandas as pd
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

from muse.registration import registrator

OUTPUT_QUANTITY_SIGNATURE = Callable[
<<<<<<< HEAD
    [xr.Dataset, xr.DataArray, xr.Dataset, KwArg(Any)],
    Union[pd.DataFrame, xr.DataArray],
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: MutableMapping[Text, OUTPUT_QUANTITY_SIGNATURE] = {}
=======
    [Dataset, DataArray, Dataset, KwArg()], Union[pd.DataFrame, DataArray]
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: Mapping[Text, OUTPUT_QUANTITY_SIGNATURE] = {}
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
"""Quantity for post-simulation analysis."""

OUTPUTS_PARAMETERS = Union[Text, Mapping]
"""Acceptable Datastructures for outputs parameters"""


@registrator(registry=OUTPUT_QUANTITIES)
def register_output_quantity(function: OUTPUT_QUANTITY_SIGNATURE = None) -> Callable:
    """Registers a function to compute an output quantity."""
    from functools import wraps

    assert function is not None

    @wraps(function)
    def decorated(*args, **kwargs):
        result = function(*args, **kwargs)
<<<<<<< HEAD
        if isinstance(result, (pd.DataFrame, xr.DataArray)):
=======
        if isinstance(result, (pd.DataFrame, DataArray)):
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            result.name = function.__name__
        return result

    return decorated


def _quantity_factory(
    parameters: Mapping, registry: Mapping[Text, Callable]
) -> Callable:
    from functools import partial
<<<<<<< HEAD
    from inspect import isclass
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    config = dict(**parameters)
    params = config.pop("quantity")
    if isinstance(params, Mapping):
        params = dict(**params)
        quantity = params.pop("name")
    else:
        quantity = params
        params = {}
    if registry is None:
        registry = OUTPUT_QUANTITIES
<<<<<<< HEAD
    quantity_function = registry[quantity]
    if isclass(quantity_function):
        return quantity_function(**params)  # type: ignore
    else:
        return partial(quantity_function, **params)
=======
    return partial(registry[quantity], **params)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


def _factory(
    registry: Mapping[Text, Callable],
    *parameters: OUTPUTS_PARAMETERS,
    sector_name: Text = "default",
) -> Callable:
    from muse.outputs.sinks import factory as sink_factory

    if isinstance(parameters, Text):
        params: List = [{"quantity": parameters}]
    elif isinstance(parameters, Mapping):
        params = [parameters]
    else:
        params = [  # type: ignore
            {"quantity": o} if isinstance(o, Text) else o for o in parameters
        ]

    quantities = [_quantity_factory(param, registry) for param in params]
    sinks = [sink_factory(param, sector_name=sector_name) for param in params]

    def save_multiple_outputs(market, *args, year: Optional[int] = None) -> List[Any]:

        if year is None:
            year = int(market.year.min())
        return [
            sink(quantity(market, *args), year=year)
            for quantity, sink in zip(quantities, sinks)
        ]

    return save_multiple_outputs


def factory(
    *parameters: OUTPUTS_PARAMETERS, sector_name: Text = "default"
<<<<<<< HEAD
) -> Callable[[xr.Dataset, xr.DataArray, xr.Dataset], List[Any]]:
=======
) -> Callable[[Dataset, DataArray, Dataset], List[Any]]:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    """Creates outputs functions for post-mortem analysis.

    Each parameter is a dictionary containing the following:

    - quantity (mandatory): name of the quantity to output. Mandatory.
    - sink (optional): name of the storage procedure, e.g. the file format
      or database format. When it cannot be guessed from `filename`, it defaults to
      "csv".
    - filename (optional): path to a directory or a file where to store the quantity. In
      the latter case, if sink is not given, it will be determined from the file
      extension. The filename can incorporate markers. By default, it is
      "{default_output_dir}/{sector}{year}{quantity}{suffix}".
    - any other parameter relevant to the sink, e.g. `pandas.to_csv` keyword
      arguments.

    For simplicity, it is also possible to given lone strings as input.
    They default to `{'quantity': string}` (and the sink will default to
    "csv").
    """
    return _factory(OUTPUT_QUANTITIES, *parameters, sector_name=sector_name)


@register_output_quantity
<<<<<<< HEAD
def capacity(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    rounding: int = 4,
) -> pd.DataFrame:
    """Current capacity."""
    result = capacity.to_dataframe().round(rounding)
    return result[result.capacity != 0]


def market_quantity(
    quantity: xr.DataArray,
    sum_over: Optional[Union[Text, List[Text]]] = None,
    drop: Optional[Union[Text, List[Text]]] = None,
) -> xr.DataArray:
    from muse.utilities import multiindex_to_coords
    from pandas import MultiIndex

    if isinstance(sum_over, Text):
        sum_over = [sum_over]
    if sum_over:
        sum_over = [s for s in sum_over if s in quantity.coords]
=======
def capacity(market: Dataset, capacity: DataArray, technologies: Dataset) -> DataArray:
    """Current capacity."""
    return capacity


def market_quantity(
    quantity: DataArray,
    sum_over: Optional[Union[Text, List[Text]]] = None,
    drop: Optional[Union[Text, List[Text]]] = None,
) -> DataArray:
    from muse.utilities import multiindex_to_coords
    from pandas import MultiIndex

>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    if sum_over:
        quantity = quantity.sum(sum_over)
    if "timeslice" in quantity.dims and isinstance(quantity.timeslice, MultiIndex):
        quantity = multiindex_to_coords(quantity, "timeslice")
    if drop:
<<<<<<< HEAD
        quantity = quantity.drop_vars([d for d in drop if d in quantity.coords])
=======
        quantity = quantity.drop_vars(drop)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    return quantity


@register_output_quantity
def consumption(
<<<<<<< HEAD
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current consumption."""
    result = (
        market_quantity(market.consumption, sum_over=sum_over, drop=drop)
        .rename("consumption")
        .to_dataframe()
        .round(rounding)
    )
    return result[result.consumption != 0]
=======
    market: Dataset,
    capacity: DataArray,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current consumption."""
    return market_quantity(market.consumption, sum_over=sum_over, drop=drop)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


@register_output_quantity
def supply(
<<<<<<< HEAD
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current supply."""
    result = (
        market_quantity(market.supply, sum_over=sum_over, drop=drop)
        .rename("supply")
        .to_dataframe()
        .round(rounding)
    )
    return result[result.supply != 0]
=======
    market: Dataset,
    capacity: DataArray,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current supply."""
    return market_quantity(market.supply, sum_over=sum_over, drop=drop)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


@register_output_quantity
def costs(
<<<<<<< HEAD
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current costs."""
    from muse.commodities import is_pollutant

    result = (
        market_quantity(
            market.costs.sel(commodity=~is_pollutant(market.comm_usage)),
            sum_over=sum_over,
            drop=drop,
        )
        .rename("costs")
        .to_dataframe()
        .round(rounding)
    )
    return result[result.costs != 0]
=======
    market: Dataset,
    capacity: DataArray,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current supply."""
    from muse.commodities import is_pollutant

    return market_quantity(
        market.costs.sel(commodity=~is_pollutant(market.comm_usage)),
        sum_over=sum_over,
        drop=drop,
    )
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
