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

from operator import attrgetter
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

import numpy as np
import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator
from muse.sectors import AbstractSector
from muse.timeslices import QuantityType, convert_timeslice

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
        if len(result) > 0:
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

    parameters = cast(  # type: ignore
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

    ts_coords = list(market.indexes["timeslice"].names)
    result = market_quantity(market.prices, **kwargs).to_dataframe()
    if drop_empty:
        result = result[result.prices != 0]

    if isinstance(keep_columns, Text):
        result = result[[*ts_coords, keep_columns]]

    elif keep_columns is not None and len(keep_columns) > 0:
        result = result[ts_coords + [u for u in result.columns if u in keep_columns]]

    # We assign back a timeslice column with the original coordinate names
    # Each timeslice is a tuple of the original coordinates (month, day, hour)
    index_names = result.index.names
    result = result.reset_index()
    result["timeslice"] = list(zip(*[result[name] for name in ts_coords]))
    result = result.set_index(index_names, drop=True).drop(ts_coords, axis=1)
    return result


@register_output_quantity
@round_values
def capacity(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current capacity across all sectors."""
    return _aggregate_sectors(sectors, op=sector_capacity)


def sector_capacity(sector: AbstractSector) -> pd.DataFrame:
    """Sector capacity with agent annotations."""
    capa_sector: List[xr.DataArray] = []
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for agent in agents:
        capa_agent = agent.assets.capacity
        capa_agent["agent"] = agent.name
        capa_agent["type"] = agent.category
        capa_agent["sector"] = getattr(sector, "name", "unnamed")

        if len(capa_agent) > 0 and len(capa_agent.technology.values) > 0:
            if "dst_region" not in capa_agent.coords:
                capa_agent["dst_region"] = agent.region
            a = capa_agent.to_dataframe()
            b = (
                a.groupby(
                    [
                        "technology",
                        "dst_region",
                        "region",
                        "agent",
                        "sector",
                        "type",
                        "year",
                        "installed",
                    ]
                )
                .sum()  # ("asset")
                .fillna(0)
            )
            c = b.reset_index()
            capa_sector.append(c)
    if len(capa_sector) == 0:
        return pd.DataFrame()

    capacity = pd.concat([u for u in capa_sector])
    capacity = capacity[capacity.capacity != 0]

    capacity = capacity.reset_index()
    return capacity


def _aggregate_sectors(
    sectors: List[AbstractSector], *args, op: Callable
) -> pd.DataFrame:
    """Aggregate outputs from all sectors."""
    alldata = [op(sector, *args) for sector in sectors]

    if len(alldata) == 0:
        return pd.DataFrame()
    return pd.concat(alldata, sort=True)


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
        return aggregate <= limits.assign_coords(timeslice=aggregate.timeslice)


@register_output_quantity(name=["timeslice_supply"])
def metric_supply(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current timeslice supply across all sectors."""
    market_out = market.copy(deep=True)
    return _aggregate_sectors(sectors, market_out, op=sector_supply)


def sector_supply(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Sector supply with agent annotations."""
    from muse.production import supply

    data_sector: List[xr.DataArray] = []
    techs = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    if len(techs) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            technologies = a.filter_input(techs, year=output_year).fillna(0.0)
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = (
                agent_market.consumption * a.quantity
            ).drop_vars(["timeslice", "month", "day", "hour"])
            included = [
                i
                for i in agent_market["commodity"].values
                if i in technologies.enduse.values
            ]
            excluded = [
                i for i in agent_market["commodity"].values if i not in included
            ]
            agent_market.loc[dict(commodity=excluded)] = 0

            result = convert_timeslice(
                supply(
                    agent_market,
                    capacity,
                    technologies,
                ),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )

            if "year" in result.dims:
                data_agent = result.sel(year=output_year)
            else:
                data_agent = result
                data_agent["year"] = output_year
            if "dst_region" not in data_agent.coords:
                data_agent["dst_region"] = a.region
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")

            a = data_agent.to_dataframe("supply")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.drop(
                    ["month", "day", "hour"], axis=1, errors="ignore"
                ).reset_index()
                b = b[b["supply"] != 0]
                data_sector.append(b)
    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector], sort=True)

    else:
        output = pd.DataFrame()

    # Combine timeslice columns into a single column, if present
    if "hour" in output.columns:
        output["timeslice"] = list(zip(output["month"], output["day"], output["hour"]))
        output = output.drop(["month", "day", "hour"], axis=1)

    return output.reset_index()


@register_output_quantity(name=["yearly_supply"])
def metricy_supply(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current yearlysupply across all sectors."""
    market_out = market.copy(deep=True)
    return _aggregate_sectors(sectors, market_out, op=sectory_supply)


def sectory_supply(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector supply with agent annotations."""
    from muse.production import supply

    def capacity(agents):
        """Aggregates capacity across agents.

        The capacities are aggregated leaving only two
        dimensions: asset (technology, installation date,
        region), year.
        """
        from muse.utilities import filter_input, reduce_assets

        traded = [
            u.assets.capacity for u in agents if "dst_region" in u.assets.capacity.dims
        ]
        nontraded = [
            u.assets.capacity
            for u in agents
            if "dst_region" not in u.assets.capacity.dims
        ]
        if not traded:
            full_list = [
                list(nontraded[i].year.values)
                for i in range(len(nontraded))
                if "year" in nontraded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            nontraded = [
                filter_input(u.assets.capacity, year=years)
                for u in agents
                if "dst_region" not in u.assets.capacity.dims
            ]

            return reduce_assets(nontraded)

        if not nontraded:
            full_list = [
                list(traded[i].year.values)
                for i in range(len(traded))
                if "year" in traded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            traded = [
                filter_input(u.assets.capacity, year=years)
                for u in agents
                if "dst_region" in u.assets.capacity.dims
            ]
            return reduce_assets(traded)
        traded_results = reduce_assets(traded)
        nontraded_results = reduce_assets(nontraded)
        return reduce_assets(
            [
                traded_results,
                nontraded_results
                * (nontraded_results.region == traded_results.dst_region),
            ]
        )

    data_sector: List[xr.DataArray] = []
    techs = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    if len(techs) > 0:
        if "dst_region" in techs.dims:
            output_year = agents[0].year - agents[0].forecast
            years = market.year.values
            capacity = (
                capacity(agents)
                .interp(year=years, method="linear")
                .sel(year=output_year)
            )
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = agent_market.consumption
            technologies = techs.sel(year=output_year)
            result = supply(
                agent_market,
                capacity,
                technologies,
            )

            if "year" in result.dims:
                data_agent = result.sel(year=output_year)
            else:
                data_agent = result
                data_agent["year"] = output_year

            data_agent["agent"] = agents[0].name
            data_agent["category"] = agents[0].category
            data_agent["sector"] = getattr(sector, "name", "unnamed")

            a = data_agent.to_dataframe("supply")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.reset_index()
                b = b[b["supply"] != 0]
                data_sector.append(b)
        else:
            for agent in agents:
                output_year = agent.year - agent.forecast
                capacity = agent.filter_input(
                    agent.assets.capacity, year=output_year
                ).fillna(0.0)
                technologies = techs.sel(year=output_year, region=agent.region)
                agent_market = market.sel(year=output_year).copy()
                agent_market["consumption"] = agent_market.consumption * agent.quantity
                included = [
                    i
                    for i in agent_market["commodity"].values
                    if i in technologies.enduse.values
                ]
                excluded = [
                    i for i in agent_market["commodity"].values if i not in included
                ]
                agent_market.loc[dict(commodity=excluded)] = 0

                result = supply(
                    agent_market,
                    capacity,
                    technologies,
                )

                if "year" in result.dims:
                    data_agent = result.sel(year=output_year)
                else:
                    data_agent = result
                    data_agent["year"] = output_year
                if "dst_region" not in data_agent.coords:
                    data_agent["dst_region"] = agent.region
                data_agent["agent"] = agent.name
                data_agent["category"] = agent.category
                data_agent["sector"] = getattr(sector, "name", "unnamed")

                a = data_agent.to_dataframe("supply")
                if len(a) > 0 and len(a.technology.values) > 0:
                    b = a.reset_index()
                    b = b[b["supply"] != 0]
                    data_sector.append(b)

    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector], sort=True)

    else:
        output = pd.DataFrame()
    output = output.reset_index()

    return output


@register_output_quantity(name=["timeslice_consumption"])
def metric_consumption(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current timeslice consumption across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_consumption)


def sector_consumption(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector fuel consumption with agent annotations."""
    from muse.production import supply
    from muse.quantities import consumption

    data_sector: List[xr.DataArray] = []
    techs = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market
    if len(techs) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            technologies = a.filter_input(techs, year=output_year).fillna(0.0)
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = (
                agent_market.consumption * a.quantity
            ).drop_vars(["timeslice", "month", "day", "hour"])
            included = [
                i
                for i in agent_market["commodity"].values
                if i in technologies.enduse.values
            ]
            excluded = [
                i for i in agent_market["commodity"].values if i not in included
            ]
            agent_market.loc[dict(commodity=excluded)] = 0

            production = convert_timeslice(
                supply(
                    agent_market,
                    capacity,
                    technologies,
                ),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )
            prices = a.filter_input(market.prices, year=output_year)
            result = consumption(
                technologies=technologies, production=production, prices=prices
            )
            if "year" in result.dims:
                data_agent = result.sel(year=output_year)
            else:
                data_agent = result
                data_agent["year"] = output_year
            if "dst_region" not in data_agent.coords:
                data_agent["dst_region"] = a.region
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            a = data_agent.to_dataframe("consumption")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.drop(
                    ["month", "day", "hour"], axis=1, errors="ignore"
                ).reset_index()
                b = b[b["consumption"] != 0]
                data_sector.append(b)
    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector], sort=True)

    else:
        output = pd.DataFrame()

    # Combine timeslice columns into a single column, if present
    if "hour" in output.columns:
        output["timeslice"] = list(zip(output["month"], output["day"], output["hour"]))
        output = output.drop(["month", "day", "hour"], axis=1)

    return output.reset_index()


@register_output_quantity(name=["yearly_consumption"])
def metricy_consumption(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current yearly consumption across all sectors."""
    return _aggregate_sectors(sectors, market, op=sectory_consumption)


def sectory_consumption(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector fuel consumption with agent annotations."""
    from muse.production import supply
    from muse.quantities import consumption

    data_sector: List[xr.DataArray] = []
    techs = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market
    if len(techs) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            technologies = a.filter_input(techs, year=output_year).fillna(0.0)
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = agent_market.consumption * a.quantity
            included = [
                i
                for i in agent_market["commodity"].values
                if i in technologies.enduse.values
            ]
            excluded = [
                i for i in agent_market["commodity"].values if i not in included
            ]
            agent_market.loc[dict(commodity=excluded)] = 0

            production = supply(
                agent_market,
                capacity,
                technologies,
            )

            prices = a.filter_input(market.prices, year=output_year)
            result = consumption(
                technologies=technologies, production=production, prices=prices
            )
            if "year" in result.dims:
                data_agent = result.sel(year=output_year)
            else:
                data_agent = result
                data_agent["year"] = output_year
            if "dst_region" not in data_agent.coords:
                data_agent["dst_region"] = a.region
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            a = data_agent.to_dataframe("consumption")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.reset_index()
                b = b[b["consumption"] != 0]
                data_sector.append(b)
    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector], sort=True)

    else:
        output = pd.DataFrame()
    output = output.reset_index()

    return output


@register_output_quantity(name=["fuel_costs"])
def metric_fuel_costs(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current lifetime levelised cost across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_fuel_costs)


def sector_fuel_costs(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector fuel costs with agent annotations."""
    from muse.commodities import is_fuel
    from muse.production import supply
    from muse.quantities import consumption

    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )
            commodity = is_fuel(technologies.comm_usage)

            capacity = a.filter_input(
                a.assets.capacity,
                year=output_year,
            ).fillna(0.0)

            production = convert_timeslice(
                supply(
                    agent_market,
                    capacity,
                    technologies,
                ),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )

            prices = a.filter_input(market.prices, year=output_year)
            fcons = consumption(
                technologies=technologies, production=production, prices=prices
            )

            data_agent = (fcons * prices).sel(commodity=commodity)
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat(
            [u.to_dataframe("fuel_consumption_costs") for u in data_sector], sort=True
        )
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["capital_costs"])
def metric_capital_costs(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current capital costs across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_capital_costs)


def sector_capital_costs(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector capital costs with agent annotations."""
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    if len(technologies) > 0:
        for a in agents:
            demand = market.consumption * a.quantity
            output_year = a.year - a.forecast
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            data = a.filter_input(
                technologies[["cap_par", "cap_exp"]],
                year=output_year,
                technology=capacity.technology,
            )
            result = data.cap_par * (capacity**data.cap_exp)
            data_agent = convert_timeslice(
                result,
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            a = data_agent.to_dataframe("capital_costs")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.reset_index()
                b = b[b["capital_costs"] != 0]
                data_sector.append(b)
    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector], sort=True)
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["emission_costs"])
def metric_emission_costs(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_emission_costs)


def sector_emission_costs(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector emission costs with agent annotations."""
    from muse.commodities import is_enduse, is_pollutant
    from muse.production import supply

    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )

            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            allemissions = a.filter_input(
                technologies.fixed_outputs,
                commodity=is_pollutant(technologies.comm_usage),
                technology=capacity.technology,
                year=output_year,
            )
            envs = is_pollutant(technologies.comm_usage)
            enduses = is_enduse(technologies.comm_usage)
            i = (np.where(envs))[0][0]
            red_envs = envs[i].commodity.values
            prices = a.filter_input(market.prices, year=output_year, commodity=red_envs)
            production = convert_timeslice(
                supply(
                    agent_market,
                    capacity,
                    technologies,
                ),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )
            total = production.sel(commodity=enduses).sum("commodity")
            data_agent = total * (allemissions * prices).sum("commodity")
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat(
            [u.to_dataframe("emission_costs") for u in data_sector], sort=True
        )
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["LCOE"])
def metric_lcoe(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_lcoe)


def sector_lcoe(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Levelized cost of energy () of technologies over their lifetime."""
    from muse.commodities import is_enduse, is_fuel, is_material, is_pollutant
    from muse.objectives import discount_factor
    from muse.quantities import consumption

    def capacity_to_service_demand(demand, technologies):
        from muse.timeslices import represent_hours

        hours = represent_hours(demand.timeslice)

        max_hours = hours.max() / hours.sum()

        commodity_output = technologies.fixed_outputs.sel(commodity=demand.commodity)

        max_demand = (
            demand.where(commodity_output > 0, 0)
            / commodity_output.where(commodity_output > 0, 1)
        ).max(("commodity", "timeslice"))

        return max_demand / technologies.utilization_factor / max_hours

    # Filtering of the inputs
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    retro = [a for a in agents if a.category == "retrofit"]
    new = [a for a in agents if a.category == "new"]
    agents = retro if len(retro) > 0 else new
    if len(technologies) > 0:
        for agent in agents:
            output_year = agent.year - agent.forecast
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = agent_market.consumption * agent.quantity
            included = [
                i
                for i in agent_market["commodity"].values
                if i in technologies.enduse.values
            ]
            excluded = [
                i for i in agent_market["commodity"].values if i not in included
            ]
            agent_market.loc[dict(commodity=excluded)] = 0
            years = [output_year, agent.year]

            agent_market["prices"] = agent.filter_input(market["prices"], year=years)

            tech = agent.filter_input(
                technologies[
                    [
                        "technical_life",
                        "interest_rate",
                        "cap_par",
                        "cap_exp",
                        "var_par",
                        "var_exp",
                        "fix_par",
                        "fix_exp",
                        "fixed_outputs",
                        "fixed_inputs",
                        "flexible_inputs",
                        "utilization_factor",
                    ]
                ],
                year=agent.year,
                region=agent.region,
            )
            nyears = tech.technical_life.astype(int)
            interest_rate = tech.interest_rate
            cap_par = tech.cap_par
            cap_exp = tech.cap_exp
            var_par = tech.var_par
            var_exp = tech.var_exp
            fix_par = tech.fix_par
            fix_exp = tech.fix_exp
            fixed_outputs = tech.fixed_outputs
            utilization_factor = tech.utilization_factor

            # All years the simulation is running
            # NOTE: see docstring about installation year
            iyears = range(
                agent.year,
                max(agent.year + nyears.values.max(), agent.forecast_year),
            )
            years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")

            prices = agent.filter_input(agent_market.prices, year=years.values)
            # Filters
            environmentals = is_pollutant(tech.comm_usage)
            e = np.where(environmentals)
            environmentals = environmentals[e].commodity.values
            material = is_material(tech.comm_usage)
            e = np.where(material)
            material = material[e].commodity.values
            products = is_enduse(tech.comm_usage)
            e = np.where(products)
            products = products[e].commodity.values
            fuels = is_fuel(tech.comm_usage)
            e = np.where(fuels)
            fuels = fuels[e].commodity.values
            # Capacity
            demand = agent_market.consumption.sel(commodity=included)
            capacity = capacity_to_service_demand(demand, tech)

            # Evolution of rates with time
            rates = discount_factor(
                years - agent.year + 1, interest_rate, years <= agent.year + nyears
            )

            production = capacity * fixed_outputs * utilization_factor
            production = convert_timeslice(
                production,
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )
            # raw costs --> make the NPV more negative
            # Cost of installed capacity
            installed_capacity_costs = convert_timeslice(
                cap_par * (capacity**cap_exp),
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )

            # Cost related to environmental products

            prices_environmental = agent.filter_input(prices, year=years.values).sel(
                commodity=environmentals
            )
            environmental_costs = (
                (production * prices_environmental * rates)
                .sel(commodity=environmentals, year=years.values)
                .sum(("commodity", "year"))  # , "timeslice")
            )

            # Fuel/energy costs
            prices_fuel = agent.filter_input(prices, year=years.values).sel(
                commodity=fuels
            )

            fuel = consumption(
                technologies=tech,
                production=production.sel(region=tech.region),
                prices=prices,
            )
            fuel_costs = (fuel * prices_fuel * rates).sum(("commodity", "year"))

            # Cost related to material other than fuel/energy and environmentals
            prices_material = agent.filter_input(prices, year=years.values).sel(
                commodity=material
            )
            material_costs = (production * prices_material * rates).sum(
                ("commodity", "year")
            )

            # Fixed and Variable costs
            fixed_costs = convert_timeslice(
                fix_par * (capacity**fix_exp),
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )
            variable_costs = (
                var_par * production.sel(commodity=products) ** var_exp
            ).sum("commodity")
            #    assert set(fixed_costs.dims) == set(variable_costs.dims)
            fixed_and_variable_costs = ((fixed_costs + variable_costs) * rates).sum(
                "year"
            )

            result = (
                installed_capacity_costs
                + fuel_costs
                + environmental_costs
                + material_costs
                + fixed_and_variable_costs
            ) / (production.sel(commodity=products).sum("commodity") * rates).sum(
                "year"
            )

            data_agent = result
            data_agent["agent"] = agent.name
            data_agent["category"] = agent.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("LCOE") for u in data_sector], sort=True)
        output = output.reset_index()

    else:
        output = pd.DataFrame()
    return output


@register_output_quantity(name=["EAC"])
def metric_eac(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    return _aggregate_sectors(sectors, market, op=sector_eac)


def sector_eac(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Net Present Value of technologies over their lifetime."""
    from muse.commodities import is_enduse, is_fuel, is_material, is_pollutant
    from muse.objectives import discount_factor
    from muse.quantities import consumption

    def capacity_to_service_demand(demand, technologies):
        from muse.timeslices import represent_hours

        hours = represent_hours(demand.timeslice)

        max_hours = hours.max() / hours.sum()

        commodity_output = technologies.fixed_outputs.sel(commodity=demand.commodity)

        max_demand = (
            demand.where(commodity_output > 0, 0)
            / commodity_output.where(commodity_output > 0, 1)
        ).max(("commodity", "timeslice"))

        return max_demand / technologies.utilization_factor / max_hours

    # Filtering of the inputs
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    retro = [a for a in agents if a.category == "retrofit"]
    new = [a for a in agents if a.category == "new"]
    agents = retro if len(retro) > 0 else new
    if len(technologies) > 0:
        for agent in agents:
            output_year = agent.year - agent.forecast
            agent_market = market.sel(year=output_year).copy()
            agent_market["consumption"] = agent_market.consumption * agent.quantity
            included = [
                i
                for i in agent_market["commodity"].values
                if i in technologies.enduse.values
            ]
            excluded = [
                i for i in agent_market["commodity"].values if i not in included
            ]
            agent_market.loc[dict(commodity=excluded)] = 0

            years = [output_year, agent.year]

            agent_market["prices"] = agent.filter_input(market["prices"], year=years)

            tech = agent.filter_input(
                technologies[
                    [
                        "technical_life",
                        "interest_rate",
                        "cap_par",
                        "cap_exp",
                        "var_par",
                        "var_exp",
                        "fix_par",
                        "fix_exp",
                        "fixed_outputs",
                        "fixed_inputs",
                        "flexible_inputs",
                        "utilization_factor",
                    ]
                ],
                year=agent.year,
                region=agent.region,
            )
            nyears = tech.technical_life.astype(int)
            interest_rate = tech.interest_rate
            cap_par = tech.cap_par
            cap_exp = tech.cap_exp
            var_par = tech.var_par
            var_exp = tech.var_exp
            fix_par = tech.fix_par
            fix_exp = tech.fix_exp
            fixed_outputs = tech.fixed_outputs
            utilization_factor = tech.utilization_factor

            # All years the simulation is running
            # NOTE: see docstring about installation year
            iyears = range(
                agent.year,
                max(agent.year + nyears.values.max(), agent.forecast_year),
            )
            years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")

            prices = agent.filter_input(agent_market.prices, year=years.values)
            # Filters
            environmentals = is_pollutant(tech.comm_usage)
            e = np.where(environmentals)
            environmentals = environmentals[e].commodity.values
            material = is_material(tech.comm_usage)
            e = np.where(material)
            material = material[e].commodity.values
            products = is_enduse(tech.comm_usage)
            e = np.where(products)
            products = products[e].commodity.values
            fuels = is_fuel(tech.comm_usage)
            e = np.where(fuels)
            fuels = fuels[e].commodity.values
            # Capacity
            demand = agent_market.consumption.sel(commodity=included)
            capacity = capacity_to_service_demand(demand, tech)

            # Evolution of rates with time
            rates = discount_factor(
                years - agent.year + 1, interest_rate, years <= agent.year + nyears
            )

            production = capacity * fixed_outputs * utilization_factor
            production = convert_timeslice(
                production,
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )
            # raw costs --> make the NPV more negative
            # Cost of installed capacity
            installed_capacity_costs = convert_timeslice(
                cap_par * (capacity**cap_exp),
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )

            # Cost related to environmental products

            prices_environmental = agent.filter_input(prices, year=years.values).sel(
                commodity=environmentals
            )
            environmental_costs = (
                (production * prices_environmental * rates)
                .sel(commodity=environmentals, year=years.values)
                .sum(("commodity", "year"))
            )

            # Fuel/energy costs
            prices_fuel = agent.filter_input(prices, year=years.values).sel(
                commodity=fuels
            )

            fuel = consumption(
                technologies=tech,
                production=production.sel(region=tech.region),
                prices=prices,
            )
            fuel_costs = (fuel * prices_fuel * rates).sum(("commodity", "year"))

            # Cost related to material other than fuel/energy and environmentals
            prices_material = agent.filter_input(prices, year=years.values).sel(
                commodity=material
            )  # .ffill("year")
            material_costs = (production * prices_material * rates).sum(
                ("commodity", "year")
            )

            # Fixed and Variable costs
            fixed_costs = convert_timeslice(
                fix_par * (capacity**fix_exp),
                demand.timeslice,
                QuantityType.EXTENSIVE,
            )
            variable_costs = (
                var_par * production.sel(commodity=products) ** var_exp
            ).sum("commodity")
            #    assert set(fixed_costs.dims) == set(variable_costs.dims)
            fixed_and_variable_costs = ((fixed_costs + variable_costs) * rates).sum(
                "year"
            )
            # raw revenues --> Make the NPV more positive
            # This production is the absolute maximum production,
            # given the capacity
            raw_revenues = (
                (production * prices * rates)
                .sel(commodity=products)
                .sum(("commodity", "year"))
            )

            result = (
                installed_capacity_costs
                + fuel_costs
                + environmental_costs
                + material_costs
                + fixed_and_variable_costs
            ) - raw_revenues
            crf = interest_rate / (1 - (1 / (1 + interest_rate) ** nyears))
            result *= crf
            data_agent = result
            data_agent["agent"] = agent.name
            data_agent["category"] = agent.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("EAC") for u in data_sector], sort=True)
        output = output.reset_index()

    else:
        output = pd.DataFrame()
    return output
