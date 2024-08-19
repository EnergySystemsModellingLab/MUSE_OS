"""Output quantities.

Functions that compute sectorial quantities for post-simulation analysis should all
follow the same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        capacity: xr.DataArray,
        market: xr.Dataset,
    ) -> Union[xr.DataArray, DataFrame]:
        pass

They take as input the current capacity profile, aggregated across a sectoar,
a dataset containing market-related quantities, and a dataset characterizing the
technologies in the market. It returns a single xr.DataArray object.

The function should never modify it's arguments.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, Optional, Union

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator

OUTPUT_QUANTITY_SIGNATURE = Callable[
    [xr.Dataset, xr.DataArray, xr.Dataset, KwArg(Any)],
    Union[pd.DataFrame, xr.DataArray],
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: MutableMapping[str, OUTPUT_QUANTITY_SIGNATURE] = {}
"""Quantity for post-simulation analysis."""

OUTPUTS_PARAMETERS = Union[str, Mapping]
"""Acceptable Datastructures for outputs parameters"""


@registrator(registry=OUTPUT_QUANTITIES)
def register_output_quantity(function: OUTPUT_QUANTITY_SIGNATURE = None) -> Callable:
    """Registers a function to compute an output quantity."""
    from functools import wraps

    assert function is not None

    @wraps(function)
    def decorated(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, (pd.DataFrame, xr.DataArray)):
            result.name = function.__name__
        return result

    return decorated


def _quantity_factory(
    parameters: Mapping, registry: Mapping[str, Callable]
) -> Callable:
    from functools import partial
    from inspect import isclass

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
    quantity_function = registry[quantity]
    if isclass(quantity_function):
        return quantity_function(**params)  # type: ignore
    else:
        return partial(quantity_function, **params)


def _factory(
    registry: Mapping[str, Callable],
    *parameters: OUTPUTS_PARAMETERS,
    sector_name: str = "default",
) -> Callable:
    from muse.outputs.sinks import factory as sink_factory

    if isinstance(parameters, str):
        params: list = [{"quantity": parameters}]
    elif isinstance(parameters, Mapping):
        params = [parameters]
    else:
        params = [  # type: ignore
            {"quantity": o} if isinstance(o, str) else o for o in parameters
        ]

    quantities = [_quantity_factory(param, registry) for param in params]
    sinks = [sink_factory(param, sector_name=sector_name) for param in params]

    def save_multiple_outputs(market, *args, year: Optional[int] = None) -> list[Any]:
        if year is None:
            year = int(market.year.min())

        return [
            sink(quantity(market, *args), year=year)
            for quantity, sink in zip(quantities, sinks)
        ]

    return save_multiple_outputs


def factory(
    *parameters: OUTPUTS_PARAMETERS, sector_name: str = "default"
) -> Callable[[xr.Dataset, xr.DataArray, xr.Dataset], list[Any]]:
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
def capacity(
    market: xr.Dataset,
    capacity: xr.DataArray,
    rounding: int = 4,
) -> pd.DataFrame:
    """Current capacity."""
    result = capacity.to_dataframe().round(rounding)
    result = result.reset_index()
    return result[result.capacity != 0]


def market_quantity(
    quantity: xr.DataArray,
    sum_over: Optional[Union[str, list[str]]] = None,
    drop: Optional[Union[str, list[str]]] = None,
) -> xr.DataArray:
    from pandas import MultiIndex

    from muse.utilities import multiindex_to_coords

    if isinstance(sum_over, str):
        sum_over = [sum_over]
    if sum_over:
        sum_over = [s for s in sum_over if s in quantity.coords]
    if sum_over:
        quantity = quantity.sum(sum_over)
    if "timeslice" in quantity.coords and isinstance(
        quantity.indexes["timeslice"], MultiIndex
    ):
        quantity = multiindex_to_coords(quantity, "timeslice")
    if drop:
        quantity = quantity.drop_vars([d for d in drop if d in quantity.coords])
    return quantity


@register_output_quantity
def consumption(
    market: xr.Dataset,
    capacity: xr.DataArray,
    sum_over: Optional[list[str]] = None,
    drop: Optional[list[str]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current consumption."""
    moutput = market.copy(deep=True).reset_index("timeslice")
    result = (
        market_quantity(moutput.consumption, sum_over=sum_over, drop=drop)
        .rename("consumption")
        .to_dataframe()
        .reset_index()
        .round(rounding)
    )
    return result[result.consumption != 0]


@register_output_quantity
def supply(
    market: xr.Dataset,
    capacity: xr.DataArray,
    sum_over: Optional[list[str]] = None,
    drop: Optional[list[str]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current supply."""
    moutput = market.copy(deep=True).reset_index("timeslice")
    result = (
        market_quantity(moutput.supply, sum_over=sum_over, drop=drop)
        .rename("supply")
        .to_dataframe()
        .reset_index()
        .round(rounding)
    )
    return result[result.supply != 0]


@register_output_quantity
def costs(
    market: xr.Dataset,
    capacity: xr.DataArray,
    sum_over: Optional[list[str]] = None,
    drop: Optional[list[str]] = None,
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
        .reset_index()
        .round(rounding)
    )
    return result[result.costs != 0]
