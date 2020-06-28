"""Output quantities.

Functions that compute MCA quantities for post-simulation analysis should all follow the
same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        sectors: List[AbstractSector],
        market: Dataset, **kwargs
    ) -> Union[pd.DataFrame, DataArray]:
        pass

The function should never modify it's arguments. It can return either a pandas dataframe
or an xarray DataArray.
"""
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Text, Union

import pandas as pd
from mypy_extensions import KwArg
from xarray import DataArray, Dataset

from muse.registration import registrator
from muse.sectors import AbstractSector

OUTPUT_QUANTITY_SIGNATURE = Callable[
    [Dataset, List[AbstractSector], KwArg()], Union[DataArray, pd.DataFrame]
]
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
        if isinstance(result, (pd.DataFrame, DataArray)):
            result.name = function.__name__
        return result

    return decorated


def factory(
    *parameters: OUTPUTS_PARAMETERS,
) -> Callable[[Dataset, List[AbstractSector]], List[Path]]:
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
    from muse.outputs.sector import _factory

    return _factory(OUTPUT_QUANTITIES, *parameters, sector_name="MCA")


@register_output_quantity
def consumption(market: Dataset, sectors: List[AbstractSector], **kwargs) -> DataArray:
    """Current consumption."""
    from muse.outputs.sector import market_quantity

    return market_quantity(market.consumption, **kwargs)


@register_output_quantity
def supply(market: Dataset, sectors: List[AbstractSector], **kwargs) -> DataArray:
    """Current supply."""
    from muse.outputs.sector import market_quantity

    return market_quantity(market.supply, **kwargs)


@register_output_quantity
def prices(
    market: Dataset,
    sectors: List[AbstractSector],
    drop_empty: bool = True,
    keep_columns: Optional[Union[Sequence[Text], Text]] = "prices",
    **kwargs,
) -> pd.DataFrame:
    """Current MCA market prices."""
    from muse.outputs.sector import market_quantity

    result = market_quantity(market.prices, **kwargs).to_dataframe()
    if drop_empty:
        result = result[result.prices != 0]
    if isinstance(keep_columns, Text):
        result = result[[keep_columns]]
    elif keep_columns is not None and len(keep_columns) > 0:
        result = result[[u for u in result.columns if u in keep_columns]]
    return result


@register_output_quantity
def capacity(market: Dataset, sectors: List[AbstractSector], **kwargs) -> DataArray:
    """Current capacity across all sectors."""
    return sectors_capacity(sectors)


@register_output_quantity
def alcoe(market: Dataset, sectors: List[AbstractSector], **kwargs) -> DataArray:
    """Current annual levelised cost across all sectors."""
    return sectors_alcoe(market, sectors)


@register_output_quantity
def llcoe(market: Dataset, sectors: List[AbstractSector], **kwargs) -> DataArray:
    """Current lifetime levelised cost across all sectors."""
    return sectors_llcoe(market, sectors)


def sector_alcoe(market: Dataset, sector: AbstractSector, **kwargs) -> DataArray:
    """Sector annual levelised cost with agent annotations."""
    from pandas import DataFrame, concat
    from muse.quantities import annual_levelized_cost_of_energy
    from operator import attrgetter

    data_sector: List[DataArray] = []

    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    if len(technologies) > 0:
        annual_lcoe = annual_levelized_cost_of_energy(market.prices, technologies)

        for a in agents:
            data_agent = annual_lcoe.sel(
                technology=a.assets.technology.values, region=a.assets.region.values
            )
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")

            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        alcoe = concat([u.to_dataframe("alcoe") for u in data_sector])
        alcoe = alcoe[alcoe != 0]
        if "year" in alcoe.columns:
            alcoe = alcoe.ffill("year")
    else:
        alcoe = DataFrame()

    return alcoe


def sector_llcoe(market: Dataset, sector: AbstractSector, **kwargs) -> DataArray:
    """Sector lifetime levelised cost with agent annotations."""

    from pandas import DataFrame, concat
    from operator import attrgetter
    from muse.quantities import lifetime_levelized_cost_of_energy

    data_sector: List[DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    if len(technologies) > 0:
        life_lcoe = lifetime_levelized_cost_of_energy(market.prices, technologies)

        for a in agents:
            data_agent = life_lcoe.sel(
                technology=a.assets.technology.values, region=a.assets.region.values
            )
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")

            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        lcoe = concat([u.to_dataframe("lcoe") for u in data_sector])
        lcoe = lcoe[lcoe != 0]
        if "year" in lcoe.columns:
            lcoe = lcoe.ffill("year")
    else:
        lcoe = DataFrame()

    return lcoe


def sector_capacity(sector: AbstractSector) -> DataArray:
    """Sector capacity with agent annotations."""
    from operator import attrgetter
    from pandas import DataFrame, concat

    capa_sector: List[DataArray] = []
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for agent in agents:
        capa_agent = agent.assets.capacity
        capa_agent["agent"] = agent.name
        capa_agent["type"] = agent.category
        capa_agent["sector"] = getattr(sector, "name", "unnamed")

        if len(capa_agent) > 0 and len(capa_agent.technology.values) > 0:
            capa_sector.append(capa_agent.groupby("technology").sum("asset").fillna(0))
    if len(capa_sector) == 0:
        return DataFrame()

    capacity = concat([u.to_dataframe() for u in capa_sector])
    capacity = capacity[capacity.capacity != 0]
    if "year" in capacity.columns:
        capacity = capacity.ffill("year")
    return capacity


def sectors_capacity(sectors: List[AbstractSector]) -> DataArray:
    """Aggregate outputs from all sectors."""
    from pandas import concat, DataFrame

    alldata = [sector_capacity(sector) for sector in sectors]
    if len(alldata) == 0:
        return DataFrame()
    return concat(alldata)


def sectors_alcoe(market: Dataset, sectors: List[AbstractSector]) -> DataArray:
    """Aggregate annual levelised cost from all sectors."""
    from pandas import concat, DataFrame

    alldata = [sector_alcoe(market, sector) for sector in sectors]
    if len(alldata) == 0:
        return DataFrame()
    return concat(alldata)


def sectors_llcoe(market: Dataset, sectors: List[AbstractSector]) -> DataArray:
    """Aggregate life levelised cost from all sectors."""
    from pandas import concat, DataFrame

    alldata = [sector_llcoe(market, sector) for sector in sectors]
    if len(alldata) == 0:
        return DataFrame()
    return concat(alldata)
