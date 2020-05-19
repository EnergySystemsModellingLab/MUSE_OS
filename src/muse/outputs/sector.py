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
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Text, Union

from xarray import DataArray, Dataset, concat

from muse.registration import registrator
from muse.sectors import AbstractSector

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


def quantities_factory(parameters: List[Mapping]) -> List[Callable]:
    from functools import partial

    quantities: List[Callable] = []
    for outputs in parameters:
        config = dict(**outputs)
        params = config.pop("quantity")
        if isinstance(params, Mapping):
            params = dict(**params)
            quantity = params.pop("name")
        else:
            quantity = params
            params = {}
        quantities.append(partial(OUTPUT_QUANTITIES[quantity], **params))
    return quantities


def factory(
    *parameters: OUTPUTS_PARAMETERS, sector_name: Text = "default"
) -> Callable[[DataArray, Dataset, Dataset], List[Path]]:
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
    from muse.outputs.sinks import factory as sink_factory

    if isinstance(parameters, Text):
        params: List = [{"quantity": parameters}]
    elif isinstance(parameters, Mapping):
        params = [parameters]
    else:
        params = [  # type: ignore
            {"quantity": o} if isinstance(o, Text) else o for o in parameters
        ]

    quantities = quantities_factory(params)
    sinks = [sink_factory(param, sector_name=sector_name) for param in params]

    def save_multiple_outputs(
        capacity: DataArray, market: Dataset, technologies: Dataset
    ) -> List[Path]:

        paths = []
        for quantity, sink in zip(quantities, sinks):
            data = quantity(capacity=capacity, market=market, technologies=technologies)
            paths.append(sink(data, year=int(market.year.min())))
        return paths

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


def aggregate_sector(sector: AbstractSector, year) -> DataArray:
    """Sector output to desired dimensions using reduce_assets"""
    from operator import attrgetter

    capa_sector = []
    agents = sorted(sector.agents, key=attrgetter("name"))
    for agent in agents:
        capa_agent = agent.assets.capacity.sel(year=year)
        capa_agent["agent"] = agent.name
        capa_agent["type"] = agent.category
        capa_agent["sector"] = sector.name
        capa_sector.append(capa_agent)
    capa_sector = concat(capa_sector, dim="asset")
    return capa_sector


def aggregate_sectors(sectors: List[AbstractSector], year) -> DataArray:
    """Aggregate outputs from all sectors"""
    alldata = [aggregate_sector(sector, year) for sector in sectors]
    return concat(alldata, dim="asset")
