"""Output quantities.

Functions that compute sectorial quantities for post-simulation analysis should all
follow the same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        capacity: DataArray,
        market: Dataset,
        technologies: Dataset
    ) -> DataArray:
        pass

They take as input the current capacity profile, aggregated across a sectoar,
a dataset containing market-related quantities, and a dataset characterizing the
technologies in the market. It returns a single DataArray object.

The function should never modify it's arguments.
"""
from typing import Callable, List, Mapping, Optional, Text, Union

from xarray import DataArray, Dataset

from muse.registration import registrator

OUTPUT_QUANTITY_SIGNATURE = Callable[[DataArray, Dataset, Dataset], DataArray]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: Mapping[Text, OUTPUT_QUANTITY_SIGNATURE] = {}
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
        if isinstance(result, DataArray):
            result.name = function.__name__
        return result

    return decorated


def factory(
    *parameters: OUTPUTS_PARAMETERS, sector_name: Text = "default"
) -> Callable[[DataArray, Dataset, Dataset], None]:
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

    if isinstance(parameters, Text):
        params: List = [{"quantity": parameters}]
    elif isinstance(parameters, Mapping):
        params = [parameters]
    else:
        params = [  # type: ignore
            {"quantity": o} if isinstance(o, Text) else o for o in parameters
        ]

    def save_multiple_outputs(
        capacity: DataArray, market: Dataset, technologies: Dataset
    ):

        for outputs in params:
            save_output(capacity, market, technologies, sector=sector_name, **outputs)

    return save_multiple_outputs


@register_output_quantity
def capacity(capacity: DataArray, market: Dataset, technologies: Dataset) -> DataArray:
    """Current capacity."""
    return capacity


def market_quantity(
    quantity: DataArray,
    sum_over: Optional[Union[Text, List[Text]]] = None,
    drop: Optional[Union[Text, List[Text]]] = None,
) -> DataArray:
    from muse.utilities import multiindex_to_coords

    if sum_over:
        quantity = quantity.sum(sum_over)
    if "timeslice" in quantity.dims:
        quantity = multiindex_to_coords(quantity, "timeslice")
    if drop:
        quantity = quantity.drop_vars(drop)
    return quantity


@register_output_quantity
def consumption(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current consumption."""
    return market_quantity(market.consumption, sum_over=sum_over, drop=drop)


@register_output_quantity
def supply(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current supply."""
    return market_quantity(market.supply, sum_over=sum_over, drop=drop)


@register_output_quantity
def costs(
    capacity: DataArray,
    market: Dataset,
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


def save_output(
    capacity: DataArray, market: Dataset, technologies: Dataset, **config
) -> Optional[Text]:
    """Computes output quantity and saves it to the sink.

    Arguments:
        config: A configuration dictionary which must contain at least `quantity`.
            It should also contain either `filename` or all three of `sink`, `sector`,
            and `year`. In the latter case, it can also optionally be given
            `dictionary`. If `sink` is not given, then it is guessed from the
            extension of the `filename`. If neither `sink` nor `filename` are given,
            then `sink` defaults to `csv`. Finally, the dictionary can be given
            any argument relevant to the sink.
        capacity: Asset capacity aggregated over the whole sector
        market: Market quantities
        technologies: Characteristics of the technologies

    Returns: Optionally, text describing where the data was saved, e.g. filename.
    """
    from pathlib import Path
    from muse.outputs.sinks import OUTPUT_SINKS

    config = dict(**config)
    quantity_params = config.pop("quantity")
    if isinstance(quantity_params, Mapping):
        quantity_params = dict(**quantity_params)
        quantity = quantity_params.pop("name")
    else:
        quantity = quantity_params
        quantity_params = {}

    sink_params = config.pop("sink", None)
    if isinstance(sink_params, Mapping):
        sink_params = dict(**sink_params)
        sink = sink_params.pop("name")
    elif isinstance(sink_params, Text):
        sink = sink_params
        sink_params = {}
    else:
        filename = config.get("filename", None)
        sink = config.get("suffix", Path(filename).suffix if filename else "csv")
        sink_params = {}

    if len(set(sink_params).intersection(config)) != 0:
        raise ValueError("duplicate settings in output section")
    sink_params.update(config)
    if "year" not in sink_params:
        sink_params["year"] = int(market.year.min())
    if sink[0] == ".":
        sink = sink[1:]
    data = OUTPUT_QUANTITIES[quantity](  # type: ignore
        capacity, market, technologies, **quantity_params
    )
    return OUTPUT_SINKS[sink](data, **sink_params)
