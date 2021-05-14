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
import numpy as np
from mypy_extensions import KwArg
from operator import attrgetter

from muse.registration import registrator
from muse.sectors import AbstractSector

from muse.timeslices import convert_timeslice, QuantityType
from muse.quantities import maximum_production

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
                .sum("asset")
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


@register_output_quantity(name=["timeslice_supply"])
def metric_supply(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current timeslice supply across all sectors."""
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_supply)


def sector_supply(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Sector fuel costs with agent annotations."""
    from muse.production import supply
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )
            result = convert_timeslice(
                supply(agent_market, capacity, technologies,),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )

            if "year" in result.dims:
                data_agent = result.sel(year=output_year)
            else:
                data_agent = result
                data_agent["year"] = output_year
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            a = data_agent.to_dataframe("supply")
            if len(a) > 0 and len(a.technology.values) > 0:
                b = a.groupby("technology").fillna(0)
                c = b.reset_index()
                data_sector.append(c)
    if len(data_sector) > 0:
        output = pd.concat([u for u in data_sector])
        
    else:
        output = pd.DataFrame()
    output = output.reset_index()

    return output


def costed_production(
    demand, costs, capacity, technologies, with_minimum_service, year,
) -> xr.DataArray:
    """Computes production from ranked assets.

    The assets are ranked according to their cost. The asset with least cost are allowed
    to service the demand first, up to the maximum production. By default, the mininum
    service is applied first.
    """

    from muse.utilities import broadcast_techs

    technodata = cast(xr.Dataset, broadcast_techs(technologies, capacity))

    if len(capacity.region.dims) == 0:

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return x.sum("asset")

    else:

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return xr.Dataset(dict(x=x)).groupby("region").sum("asset").x

    ranking = costs.rank("asset")
    maxprod = convert_timeslice(
        maximum_production(technodata, capacity),
        demand.timeslice,
        QuantityType.EXTENSIVE,
    )
    commodity = (maxprod > 0).any([i for i in maxprod.dims if i != "commodity"])
    commodity = commodity.drop_vars(
        [u for u in commodity.coords if u not in commodity.dims]
    )

    result = xr.zeros_like(maxprod)
    demand = demand.copy()

    constraints = (
        xr.Dataset(dict(maxprod=maxprod, ranking=ranking, has_output=maxprod > 0))
        .set_coords("ranking")
        .set_coords("has_output")
        .sel(commodity=commodity)
    )
    if maxprod.sum() > 1e-15:
        if not with_minimum_service:
            production = xr.zeros_like(constraints.maxprod)
        else:
            production = (
                getattr(technodata, "minimum_service_factor", 0) * constraints.maxprod
            )
            demand = np.maximum(demand - group_assets(production), 0)

        for rank in sorted(set(constraints.ranking.values.flatten())):

            condition = (constraints.ranking == rank) & constraints.has_output
            current_maxprod = constraints.maxprod.where(condition, 0)
            fullprod = group_assets(current_maxprod)

            if "region" in demand.dims:
                if "year" in demand.dims:
                    demand_prod = demand.sel(region=production.region, year=year)
                else:
                    demand_prod = demand.sel(region=production.region)
            com = [
                c for c in demand.commodity.values if c in constraints.commodity.values
            ]
            demand = demand.sel(commodity=com)
            if (fullprod <= demand + 1e-10).all():
                current_demand = fullprod.sel(year=year)
                current_prod = current_maxprod.sel(year=year)
            else:
                demand_prod = demand
                demand_prod = (
                    current_maxprod / current_maxprod.sum("asset") * demand_prod
                ).where(condition, 0)
                current_prod = np.minimum(demand_prod, current_maxprod).sel(year=year)
                current_demand = group_assets(current_prod)
            value = np.minimum(current_demand, demand)
            demand -= value
            if "region" in current_prod.dims:
                production += current_prod.sel(region=production.region)
            else:
                production += current_prod
        result[dict(commodity=commodity)] += production
    return result


def costed_production_export(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    costs: Union[xr.DataArray, Callable, Text] = "alcoe",
    with_minimum_service: bool = True,
    with_emission: bool = True,
    year=int,
) -> xr.DataArray:
    """Computes production from ranked assets.

    The assets are ranked according to their cost. The cost can be provided as an
    xarray, a callable creating an xarray, or as "alcoe". The asset with least cost are
    allowed to service the demand first, up to the maximum production. By default, the
    mininum service is applied first.
    """

    from muse.quantities import (
        annual_levelized_cost_of_energy,
        emission,
    )
    from muse.utilities import broadcast_techs
    from muse.commodities import is_pollutant, check_usage, CommodityUsage

    if isinstance(costs, Text) and costs.lower() == "alcoe":
        costs = annual_levelized_cost_of_energy
    elif isinstance(costs, Text):
        raise ValueError(f"Unknown cost {costs}")
    if callable(costs):
        technodata = cast(xr.Dataset, broadcast_techs(technologies, capacity))
        costs = costs(market.prices.sel(region=technodata.region), technodata)
    else:
        costs = costs
    assert isinstance(costs, xr.DataArray)

    production = costed_production(
        market.consumption,
        costs,
        capacity,
        technologies,
        with_minimum_service=with_minimum_service,
        year=year,
    )
    # add production of environmental pollutants
    if with_emission:
        env = is_pollutant(technologies.comm_usage)
        production[dict(commodity=env)] = emission(
            production, technologies.fixed_outputs
        ).transpose(*production.dims)
        production[
            dict(
                commodity=~check_usage(technologies.comm_usage, CommodityUsage.PRODUCT)
            )
        ] = 0
    return production


@register_output_quantity(name=["timeslice_consumption"])
def metric_consumption(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current timeslice consumption across all sectors."""
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_consumption)


def sector_consumption(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector fuel costs with agent annotations."""
    from muse.quantities import consumption
    from muse.production import supply

    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )

            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            production = convert_timeslice(
                supply(agent_market, capacity, technologies,),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )
            prices = a.filter_input(market.prices, year=output_year)
            data_agent = consumption(
                technologies=technologies, production=production, prices=prices
            )
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("consumption") for u in data_sector])
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["fuel_costs"])
def metric_fuel_costs(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current lifetime levelised cost across all sectors."""
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_fuel_costs)


def sector_fuel_costs(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector fuel costs with agent annotations."""
    from muse.commodities import is_fuel
    from muse.quantities import consumption
    from muse.production import supply

    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )
            commodity = is_fuel(
                technologies.comm_usage
            )  # .sel(commodity=market.commodity))

            capacity = a.filter_input(a.assets.capacity, year=output_year,).fillna(0.0)
            production = convert_timeslice(
                supply(agent_market, capacity, technologies,),
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
            [u.to_dataframe("fuel_consumption_costs") for u in data_sector]
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
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_capital_costs)


def sector_capital_costs(
    sector: AbstractSector, market: xr.Dataset, **kwargs
) -> pd.DataFrame:
    """Sector capital costs with agent annotations."""

    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
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
            result = data.cap_par * (capacity ** data.cap_exp)
            data_agent = convert_timeslice(
                result, demand.timeslice, QuantityType.EXTENSIVE,
            )
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("capital_costs") for u in data_sector])
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["emission_costs"])
def metric_emission_costs(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    print("Preparing output functions")
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

    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
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
                supply(agent_market, capacity, technologies,),
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
        output = pd.concat([u.to_dataframe("emission_costs") for u in data_sector])
        output = output.reset_index()

    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["LCOE"])
def metric_lcoe(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_lcoe)


def sector_lcoe(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Levelized cost of energy () of technologies over their lifetime.
    """
    from muse.commodities import is_pollutant, is_material, is_enduse, is_fuel
    from muse.objectives import discount_factor
    from muse.quantities import consumption
    from muse.production import supply

    # Filtering of the inputs
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
    agent_market = market.copy()
    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )

            # Capacity
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            tech = a.filter_input(
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
                    ]
                ],
                technology=capacity.technology,
                year=output_year,
            )
            nyears = tech.technical_life.astype(int)
            interest_rate = tech.interest_rate
            cap_par = tech.cap_par
            cap_exp = tech.cap_exp
            var_par = tech.var_par
            var_exp = tech.var_exp
            fix_par = tech.fix_par
            fix_exp = tech.fix_exp

            # All years the simulation is running
            # NOTE: see docstring about installation year
            iyears = range(output_year, a.year + nyears.values.max())
            years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")

            # Filters
            environmentals = is_pollutant(technologies.comm_usage)
            material = is_material(technologies.comm_usage)
            products = is_enduse(technologies.comm_usage)
            fuels = is_fuel(technologies.comm_usage)

            # Evolution of rates with time
            rates = discount_factor(
                years - output_year + 1, interest_rate, years <= output_year + nyears
            )

            production = convert_timeslice(
                supply(agent_market, capacity, technologies,),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )

            # raw costs --> make the NPV more negative
            # Cost of installed capacity
            prices = a.filter_input(market.prices, year=years.values).ffill("year")
            result = cap_par * (capacity ** cap_exp)
            installed_capacity_costs = convert_timeslice(
                result, agent_market["consumption"].timeslice, QuantityType.EXTENSIVE,
            )
            environmental_costs = (
                (production * prices * rates)
                .sel(commodity=environmentals)
                .sum(("commodity", "year"))
            )

            # Fuel/energy costs

            fcons = consumption(
                technologies=technologies, production=production, prices=prices
            )
            fuel_costs = (
                (fcons * prices * rates).sel(commodity=fuels).sum(("commodity", "year"))
            )
            # Cost related to material other than fuel/energy and
            # environmentals

            material_costs = (
                (production * prices * rates)
                .sel(commodity=material)
                .sum(("commodity", "year"))
            )

            # Fixed and Variable costs
            result = fix_par * (capacity ** fix_exp)
            fixed_costs = convert_timeslice(
                result, agent_market["consumption"].timeslice, QuantityType.EXTENSIVE,
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
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("LCOE") for u in data_sector])
        output = output.reset_index()

    else:
        output = pd.DataFrame()
    return output


@register_output_quantity(name=["EAC"])
def metric_eac(
    market: xr.Dataset, sectors: List[AbstractSector], **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    print("Preparing output functions")
    return _aggregate_sectors(sectors, market, op=sector_eac)


def sector_eac(sector: AbstractSector, market: xr.Dataset, **kwargs) -> pd.DataFrame:
    """Net Present Value of technologies over their lifetime.
    """
    from muse.commodities import is_pollutant, is_material, is_enduse, is_fuel
    from muse.objectives import discount_factor
    from muse.quantities import consumption
    from muse.production import supply
    
    # Filtering of the inputs
    data_sector: List[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    for a in agents:
        if hasattr(a, "quantity"):
            name = a.name
            attr = a.quantity
        if a.name == name and not hasattr(a, "quantity"):
            setattr(a, "quantity", attr)
    agent_market = market.copy()

    if len(technologies) > 0:
        for a in agents:
            output_year = a.year - a.forecast
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=output_year
            )

            # Capacity
            capacity = a.filter_input(a.assets.capacity, year=output_year).fillna(0.0)
            tech = a.filter_input(
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
                    ]
                ],
                technology=capacity.technology,
                year=output_year,
            )
            nyears = tech.technical_life.astype(int)
            interest_rate = tech.interest_rate
            cap_par = tech.cap_par
            cap_exp = tech.cap_exp
            var_par = tech.var_par
            var_exp = tech.var_exp
            fix_par = tech.fix_par
            fix_exp = tech.fix_exp

            # All years the simulation is running
            # NOTE: see docstring about installation year
            iyears = range(output_year, a.year + nyears.values.max())
            years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")

            # Filters
            environmentals = is_pollutant(technologies.comm_usage)
            material = is_material(technologies.comm_usage)
            products = is_enduse(technologies.comm_usage)
            fuels = is_fuel(technologies.comm_usage)

            # Evolution of rates with time
            rates = discount_factor(
                years - output_year + 1, interest_rate, years <= output_year + nyears
            )
            crf = interest_rate / (1 - (1 / (1 + interest_rate) ** nyears))

            production = convert_timeslice(
                supply(agent_market, capacity, technologies,),
                agent_market["consumption"].timeslice,
                QuantityType.EXTENSIVE,
            )
            prices = a.filter_input(market.prices, year=years.values).ffill("year")

            # raw revenues --> Make the NPV more positive
            # This production is the absolute maximum production,
            # given the capacity
            raw_revenues = (
                (production * prices * rates)
                .sel(commodity=products)
                .sum(("commodity", "year"))
            )

            # raw costs --> make the NPV more negative
            # Cost of installed capacity
            result = cap_par * (capacity ** cap_exp)
            installed_capacity_costs = convert_timeslice(
                result, agent_market["consumption"].timeslice, QuantityType.EXTENSIVE,
            )

            # Cost related to environmental products
            environmental_costs = (
                (production * prices * rates)
                .sel(commodity=environmentals)
                .sum(("commodity", "year"))
            )

            # Fuel/energy costs

            fcons = consumption(
                technologies=technologies, production=production, prices=prices
            )
            fuel_costs = (
                (fcons * prices * rates).sel(commodity=fuels).sum(("commodity", "year"))
            )
            # Cost related to material other than fuel/energy
            # and environmentals

            material_costs = (
                (production * prices * rates)
                .sel(commodity=material)
                .sum(("commodity", "year"))
            )

            # Fixed and Variable costs
            result = fix_par * (capacity ** fix_exp)
            fixed_costs = convert_timeslice(
                result, agent_market.consumption.timeslice, QuantityType.EXTENSIVE,
            )
            variable_costs = (
                var_par * production.sel(commodity=products) ** var_exp
            ).sum("commodity")
            fixed_and_variable_costs = ((fixed_costs + variable_costs) * rates).sum(
                "year"
            )

            result = -raw_revenues + (
                installed_capacity_costs
                + fuel_costs
                + environmental_costs
                + material_costs
                + fixed_and_variable_costs
            )
            result *= crf
            data_agent = result
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = output_year
            if len(data_agent) > 0 and len(data_agent.technology.values) > 0:
                data_sector.append(data_agent.groupby("technology").fillna(0))
    if len(data_sector) > 0:
        output = pd.concat([u.to_dataframe("EAC") for u in data_sector])
        output = output.reset_index()

    else:
        output = pd.DataFrame()
    return output
