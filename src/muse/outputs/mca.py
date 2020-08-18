"""Output quantities.

Functions that compute MCA quantities for post-simulation analysis should all follow the
same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        sectors: List[AbstractSector],
        market: xr.Dataset, **kwargs
    ) -> Union[pd.DataFrame, xr.DataArray]:
        pass

The function should never modify it's arguments. It can return either a pandas dataframe
or an xarray xr.DataArray.
"""
from pathlib import Path
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Text,
    Union,
    cast,
)

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator
from muse.sectors import AbstractSector

OUTPUT_QUANTITY_SIGNATURE = Callable[
    [xr.Dataset, List[AbstractSector], KwArg(Any)], Union[xr.DataArray, pd.DataFrame]
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: MutableMapping[Text, OUTPUT_QUANTITY_SIGNATURE] = {}
"""Quantity for post-simulation analysis."""

OUTPUTS_PARAMETERS = Union[Text, Mapping]
"""Acceptable Datastructures for outputs parameters"""


@registrator(registry=OUTPUT_QUANTITIES)
def register_output_quantity(
    function: Optional[OUTPUT_QUANTITY_SIGNATURE] = None,
) -> Callable:
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


def round_values(function: Callable) -> OUTPUT_QUANTITY_SIGNATURE:
    """Rounds the outputs to given number of decimals and drops columns with zeros."""
    from functools import wraps

    @wraps(function)
    def rounded(
        market: xr.Dataset, sectors: List[AbstractSector], rounding: int = 4, **kwargs
    ) -> xr.DataArray:
        result = function(market, sectors, **kwargs)
        if hasattr(result, "to_dataframe"):
            result = result.to_dataframe()
        result = result.round(rounding)
        name = getattr(result, "name", function.__name__)
        return result[result[name] != 0]

    return rounded


def factory(
    *parameters: OUTPUTS_PARAMETERS,
) -> Callable[[xr.Dataset, List[AbstractSector]], List[Path]]:
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

    def reformat_finite_resources(params):
        from muse.readers.toml import MissingSettings

        name = params["quantity"]
        if not isinstance(name, Text):
            name = name["name"]
        if name.lower() not in {"finite_resources", "finiteresources"}:
            return params

        quantity = params["quantity"]
        if isinstance(quantity, Text):
            quantity = dict(name=quantity)
        else:
            quantity = dict(**quantity)

        if "limits_path" in params:
            quantity["limits_path"] = params.pop("limits_path")
        if "commodities" in params:
            quantity["commodities"] = params.pop("commodities")
        if "limits_path" not in quantity:
            msg = "Missing limits_path tag indicating file with finite resource limits"
            raise MissingSettings(msg)
        params["sink"] = params.get("sink", "finite_resource_logger")
        params["quantity"] = quantity
        return params

    parameters = cast(
        OUTPUTS_PARAMETERS, [reformat_finite_resources(p) for p in parameters]
    )

    return _factory(OUTPUT_QUANTITIES, *parameters, sector_name="MCA")


@register_output_quantity
@round_values
def consumption(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> xr.DataArray:
    """Current consumption."""
    from muse.outputs.sector import market_quantity

    return market_quantity(market.consumption, **kwargs)


@register_output_quantity
@round_values
def supply(market: xr.Dataset, sectors: List[AbstractSector], **kwargs) -> xr.DataArray:
    """Current supply."""
    from muse.outputs.sector import market_quantity

    return market_quantity(market.supply, **kwargs)


@register_output_quantity
@round_values
def prices(
    market: xr.Dataset,
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
@round_values
def capacity(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current capacity across all sectors."""
    return _aggregate_sectors(sectors, op=sector_capacity)


@register_output_quantity(
    name=["ALCOE", "alcoe", "Annualized Levelized Cost of Energy"]
)
@round_values
def alcoe(market: xr.Dataset, sectors: List[AbstractSector], **kwargs) -> pd.DataFrame:
    """Current annual levelised cost across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_alcoe)


@register_output_quantity
@round_values
def llcoe(market: xr.Dataset, sectors: List[AbstractSector], **kwargs) -> pd.DataFrame:
    """Current lifetime levelised cost across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_llcoe)


def sector_alcoe(market: xr.Dataset, sector: AbstractSector, **kwargs) -> pd.DataFrame:
    """Sector annual levelised cost (ALCOE) with agent annotations."""
    from pandas import DataFrame, concat
    from muse.quantities import annual_levelized_cost_of_energy
    from operator import attrgetter

    data_sector: List[xr.DataArray] = []

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
        alcoe = concat([u.to_dataframe("ALCOE") for u in data_sector])
        alcoe = alcoe[alcoe != 0]
        if "year" in alcoe.columns:
            alcoe = alcoe.ffill("year")
    else:
        alcoe = DataFrame()

    return alcoe


def sector_llcoe(market: xr.Dataset, sector: AbstractSector, **kwargs) -> pd.DataFrame:
    """Sector lifetime levelised cost with agent annotations."""

    from pandas import DataFrame, concat
    from operator import attrgetter
    from muse.quantities import lifetime_levelized_cost_of_energy

    data_sector: List[xr.DataArray] = []
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
    else:
        lcoe = DataFrame()
    if "year" in lcoe.columns:
        lcoe = lcoe.ffill("year")

    return lcoe


def sector_capacity(sector: AbstractSector) -> pd.DataFrame:
    """Sector capacity with agent annotations."""
    from operator import attrgetter
    from pandas import DataFrame, concat

    capa_sector: List[xr.DataArray] = []
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


def _aggregate_sectors(
    sectors: List[AbstractSector], *args, op: Callable
) -> pd.DataFrame:
    """Aggregate outputs from all sectors."""
    alldata = [op(sector, *args) for sector in sectors]
    if len(alldata) == 0:
        return pd.DataFrame()
    return pd.concat(alldata)


@register_output_quantity
class AggregateResources:
    """Aggregates a set of commodities."""

    def __init__(
        self,
        commodities: Union[Text, Iterable[Hashable]] = (),
        metric: Text = "consumption",
    ):
        if isinstance(commodities, Text):
            commodities = [commodities]
        else:
            commodities = list(commodities)
        self.commodities: Sequence[Hashable] = commodities
        self.metric = metric
        self.aggregate: Optional[xr.DataArray] = None

    def __call__(
        self,
        market: xr.Dataset,
        sectors: List[AbstractSector],
        year: Optional[int] = None,
    ) -> Optional[xr.DataArray]:
        if len(self.commodities) == 0:
            return None
        if year is None:
            year = int(market.year.min())
        quantity = cast(xr.DataArray, market[self.metric]).sel(
            year=year, commodity=self.commodities, drop=True
        )
        if self.aggregate is None:
            self.aggregate = quantity
        else:
            self.aggregate += quantity
        return self.aggregate


@register_output_quantity(name=["finite_resources"])
class FiniteResources(AggregateResources):
    """Aggregates a set of commodities."""

    def __init__(
        self,
        limits_path: Union[Text, Path, xr.DataArray],
        commodities: Union[Text, Iterable[Hashable]] = (),
        metric: Text = "consumption",
    ):
        from muse.readers.csv import read_finite_resources

        super().__init__(commodities=commodities, metric=metric)
        if isinstance(limits_path, Text):
            limits_path = Path(limits_path)
        if isinstance(limits_path, Path):
            limits_path = read_finite_resources(limits_path)

        self.limits = limits_path

    def __call__(
        self,
        market: xr.Dataset,
        sectors: List[AbstractSector],
        year: Optional[int] = None,
    ) -> Optional[xr.DataArray]:
        if len(self.commodities) == 0:
            return None
        if year is None:
            year = int(market.year.min())

        limits = self.limits
        if "year" in self.limits.dims:
            limits = limits.interp(year=year)

        aggregate = super().__call__(market, sectors, year=year)
        if aggregate is None:
            return None
        aggregate = aggregate.sum([u for u in aggregate.dims if u not in limits.dims])
        assert aggregate is not None
        limits = limits.sum([u for u in limits.dims if u not in aggregate.dims])
        return aggregate <= limits
