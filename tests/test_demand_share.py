import xarray as xr
from pytest import approx, fixture, raises

from muse.timeslices import drop_timeslice


@fixture
def matching_market(technologies, stock, timeslice):
    """A market which matches stocks exactly."""
    return (
        _matching_market(technologies, stock, timeslice)
        .interp(year=[2010, 2015, 2020, 2025])
        .transpose("timeslice", "region", "commodity", "year")
    )


def _matching_market(technologies, stock, timeslice):
    """A market which matches stocks exactly."""
    from numpy.random import random

    from muse.quantities import consumption, maximum_production

    market = xr.Dataset()
    production = maximum_production(technologies, stock.capacity)
    market["supply"] = production.sum("asset")
    market["consumption"] = drop_timeslice(
        consumption(technologies, production).sum("asset") + market.supply
    )
    market["prices"] = market.supply.dims, random(market.supply.shape)

    return market


def test_new_retro_split_zero_unmet(technologies, stock, matching_market):
    from muse.demand_share import new_and_retro_demands

    share = new_and_retro_demands(
        stock.capacity, matching_market, technologies, current_year=2012, forecast=5
    )
    assert (share == 0).all()


def test_new_retro_split_zero_consumption_increase(
    technologies, stock, matching_market
):
    from muse.demand_share import new_and_retro_demands

    matching_market.consumption.loc[{"year": 2015}] = matching_market.consumption.sel(
        year=2010
    )
    share = new_and_retro_demands(
        stock.capacity, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share == 0).all()

    future_unmet = stock.capacity.interp(year=[2010, 2015])
    future_unmet.loc[{"year": 2015}] = 0.5 * future_unmet.sel(year=2010)
    share = new_and_retro_demands(
        future_unmet, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = stock.capacity.interp(year=[2010, 2015])
    current_unmet.loc[{"year": 2010}] = 0.5 * future_unmet.sel(year=2010)
    share = new_and_retro_demands(
        current_unmet, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(
        0.5 * stock.capacity,
        matching_market,
        technologies,
        current_year=2010,
        forecast=5,
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_split_zero_new_unmet(technologies, stock, matching_market):
    from muse.demand_share import new_and_retro_demands

    matching_market.consumption.loc[{"year": 2015}] = matching_market.supply.sel(
        year=2010, drop=True
    ).transpose(*matching_market.consumption.loc[{"year": 2015}].dims)
    share = new_and_retro_demands(
        stock.capacity, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share == 0).all()

    future_unmet = stock.capacity.interp(year=[2010, 2015])
    future_unmet.loc[{"year": 2015}] = 0.5 * future_unmet.sel(year=2010)
    share = new_and_retro_demands(
        future_unmet, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = stock.capacity.interp(year=[2010, 2015])
    current_unmet.loc[{"year": 2010}] = 0.5 * future_unmet.sel(year=2010)
    share = new_and_retro_demands(
        current_unmet, matching_market, technologies, current_year=2010, forecast=5
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(
        0.5 * stock.capacity,
        matching_market,
        technologies,
        current_year=2010,
        forecast=5,
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_accounting_identity(technologies, stock, market):
    from muse.demand_share import new_and_retro_demands
    from muse.production import factory

    share = new_and_retro_demands(
        stock.capacity, market, technologies, current_year=2010, forecast=5
    )
    assert (share >= 0).all()

    production_method = factory()
    serviced = (
        production_method(
            market.interp(year=2015), stock.capacity.interp(year=2015), technologies
        )
        .groupby("region")
        .sum("asset")
    )
    consumption = market.consumption.interp(year=2015)

    assert (share.new > -1e-8).all()
    assert (share.retrofit > -1e-8).all()
    assert ((share.new + share.retrofit).where(consumption < serviced, 0) < 1e-8).all()
    accounting = (
        (share.new + share.retrofit + serviced)
        .where(consumption - serviced > 0, consumption)
        .transpose(*consumption.dims)
    )
    assert accounting.values == approx(consumption.values)


def test_demand_split(technologies, stock, matching_market):
    from muse.commodities import is_enduse
    from muse.demand_share import _inner_split as inner_split

    def method(capacity):
        from muse.quantities import decommissioning_demand

        return decommissioning_demand(
            technologies.sel(region="USA"), capacity, year=[2012, 2017]
        )

    demand = matching_market.consumption.sel(year=2015, region="USA", drop=True).where(
        is_enduse(technologies.comm_usage.sel(commodity=matching_market.commodity))
    )
    capacity = stock.capacity
    agents = dict(scully=capacity, mulder=capacity)
    quantity = dict(scully=("scully", "USA", 0.3), mulder=("mulder", "USA", 0.7))
    share = inner_split(agents, demand, method, quantity)

    enduse = is_enduse(technologies.comm_usage)
    assert (share["scully"].sel(commodity=~enduse) == 0).all()
    assert (share["mulder"].sel(commodity=~enduse) == 0).all()

    total = (share["scully"] + share["mulder"]).sum("asset")
    demand = demand.where(enduse, 0)
    demand, total = xr.broadcast(demand, total)
    assert demand.values == approx(total.values)
    expected, actual = xr.broadcast(demand, share["scully"].sum("asset"))
    assert actual.values == approx(0.3 * expected.values)
    expected, actual = xr.broadcast(demand, share["mulder"].sum("asset"))
    assert actual.values == approx(0.7 * expected.values)


def test_demand_split_zero_share(technologies, stock, matching_market):
    """See issue SgiModel/StarMuse#688."""
    from muse.commodities import is_enduse
    from muse.demand_share import _inner_split as inner_split

    def method(capacity):
        from muse.quantities import decommissioning_demand

        return 0 * decommissioning_demand(
            technologies.sel(region="USA"), capacity, year=[2012, 2017]
        )

    demand = matching_market.consumption.sel(year=2015, region="USA", drop=True).where(
        is_enduse(technologies.comm_usage.sel(commodity=matching_market.commodity))
    )
    capacity = stock.capacity
    agents = dict(scully=0.3 * capacity, mulder=0.7 * capacity)
    quantity = dict(scully=("scully", "USA", 1), mulder=("mulder", "USA", 1))
    share = inner_split(agents, demand, method, quantity)

    enduse = is_enduse(technologies.comm_usage)
    assert (share["scully"].sel(commodity=~enduse) == 0).all()
    assert (share["mulder"].sel(commodity=~enduse) == 0).all()

    total = (share["scully"] + share["mulder"]).sum("asset")
    demand = demand.where(enduse, 0)
    demand, total = xr.broadcast(demand, total)

    assert demand.values == approx(total.values, abs=1e-10)
    expected, actual = xr.broadcast(demand, share["scully"].sum("asset"))

    assert actual.values == approx(0.5 * expected.values)
    expected, actual = xr.broadcast(demand, share["mulder"].sum("asset"))
    assert actual.values == approx(0.5 * expected.values)


def test_new_retro_demand_share(technologies, coords, market, timeslice, stock_factory):
    from dataclasses import dataclass
    from uuid import UUID, uuid4

    from muse.commodities import is_enduse
    from muse.demand_share import new_and_retro

    asia_stock = stock_factory(coords, technologies).expand_dims(region=["ASEAN"])
    usa_stock = stock_factory(coords, technologies).expand_dims(region=["USA"])

    asia_market = _matching_market(technologies, asia_stock, timeslice)
    usa_market = _matching_market(technologies, usa_stock, timeslice)
    market = xr.concat((asia_market, usa_market), dim="region")
    market.consumption.loc[{"year": 2031}] *= 2

    # spoof some agents
    @dataclass
    class Agent:
        assets: xr.Dataset
        category: str
        uuid: UUID
        name: str
        region: str
        quantity: float

    agents = [
        Agent(0.3 * usa_stock.squeeze("region"), "retrofit", uuid4(), "a", "USA", 0.3),
        Agent(0.0 * usa_stock.squeeze("region"), "new", uuid4(), "a", "USA", 0.0),
        Agent(0.7 * usa_stock.squeeze("region"), "retrofit", uuid4(), "b", "USA", 0.7),
        Agent(0.0 * usa_stock.squeeze("region"), "new", uuid4(), "b", "USA", 0.0),
        Agent(asia_stock.squeeze("region"), "retrofit", uuid4(), "a", "ASEAN", 1.0),
        Agent(0 * asia_stock.squeeze("region"), "new", uuid4(), "a", "ASEAN", 0.0),
    ]

    results = new_and_retro(agents, market, technologies, current_year=2010, forecast=5)

    for _, share in results.groupby("agent"):
        assert share.sel(
            commodity=~is_enduse(technologies.comm_usage)
        ).values == approx(0)

    uuid_to_category = {agent.uuid: agent.category for agent in agents}
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    for category in {"retrofit", "new"}:
        subset = {
            uuid_to_name[uuid]: share.sel(commodity=is_enduse(technologies.comm_usage))
            for uuid, share in results.groupby("agent")
            if uuid_to_category[uuid] == category and (share.region == "USA").all()
        }
        expected, actual = xr.broadcast(0.3 * sum(subset.values()), subset["a"])
        assert actual.values == approx(expected.values)


def test_standard_demand_share(technologies, coords, market, timeslice, stock_factory):
    from dataclasses import dataclass
    from uuid import UUID, uuid4

    from muse.commodities import is_enduse
    from muse.demand_share import standard_demand
    from muse.errors import RetrofitAgentInStandardDemandShare

    asia_stock = stock_factory(coords, technologies).expand_dims(region=["ASEAN"])
    usa_stock = stock_factory(coords, technologies).expand_dims(region=["USA"])

    asia_market = _matching_market(technologies, asia_stock, timeslice)
    usa_market = _matching_market(technologies, usa_stock, timeslice)
    market = xr.concat((asia_market, usa_market), dim="region")
    market.consumption.loc[{"year": 2031}] *= 2

    # spoof some agents
    @dataclass
    class Agent:
        assets: xr.Dataset
        category: str
        uuid: UUID
        name: str
        region: str
        quantity: float

    agents = [
        Agent(0.3 * usa_stock.squeeze("region"), "retrofit", uuid4(), "a", "USA", 0.3),
        Agent(0.0 * usa_stock.squeeze("region"), "new", uuid4(), "a", "USA", 0.0),
        Agent(0.7 * usa_stock.squeeze("region"), "retrofit", uuid4(), "b", "USA", 0.7),
        Agent(0.0 * usa_stock.squeeze("region"), "new", uuid4(), "b", "USA", 0.0),
        Agent(asia_stock.squeeze("region"), "retrofit", uuid4(), "a", "ASEAN", 1.0),
        Agent(0 * asia_stock.squeeze("region"), "new", uuid4(), "a", "ASEAN", 0.0),
    ]

    with raises(RetrofitAgentInStandardDemandShare):
        standard_demand(agents, market, technologies, current_year=2010, forecast=5)

    agents = [a for a in agents if a.category != "retrofit"]

    results = standard_demand(
        agents, market, technologies, current_year=2010, forecast=5
    )

    uuid_to_category = {agent.uuid: agent.category for agent in agents}
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    subset = {
        uuid_to_name[uuid]: share.sel(commodity=is_enduse(technologies.comm_usage))
        for uuid, share in results.groupby("agent")
        if uuid_to_category[uuid] == "new" and (share.region == "USA").all()
    }
    expected, actual = xr.broadcast(0.3 * sum(subset.values()), subset["a"])
    assert actual.values == approx(expected.values)


def test_unmet_forecast_demand(technologies, coords, timeslice, stock_factory):
    from dataclasses import dataclass

    from muse.commodities import is_enduse
    from muse.demand_share import unmet_forecasted_demand

    asia_stock = stock_factory(coords, technologies).expand_dims(region=["ASEAN"])
    usa_stock = stock_factory(coords, technologies).expand_dims(region=["USA"])

    asia_market = _matching_market(technologies, asia_stock, timeslice)
    usa_market = _matching_market(technologies, usa_stock, timeslice)
    market = xr.concat((asia_market, usa_market), dim="region")

    # spoof some agents
    @dataclass
    class Agent:
        assets: xr.Dataset

    # First ensure that the demand is fully met
    agents = [
        Agent(0.3 * usa_stock.squeeze("region")),
        Agent(0.7 * usa_stock.squeeze("region")),
        Agent(asia_stock.squeeze("region")),
    ]
    result = unmet_forecasted_demand(agents, market, technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Then try with too little demand
    agents = [
        Agent(0.4 * usa_stock.squeeze("region")),
        Agent(0.8 * usa_stock.squeeze("region")),
        Agent(1.1 * asia_stock.squeeze("region")),
    ]
    result = unmet_forecasted_demand(agents, market, technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Then try too little capacity
    agents = [
        Agent(0.5 * usa_stock.squeeze("region")),
        Agent(0.5 * asia_stock.squeeze("region")),
    ]
    result = unmet_forecasted_demand(agents, market, technologies)
    comm_usage = technologies.comm_usage.sel(commodity=market.commodity)
    enduse = is_enduse(comm_usage)
    assert (result.commodity == comm_usage.commodity).all()
    assert result.sel(commodity=~enduse).values == approx(0)
    assert result.sel(commodity=enduse).values == approx(
        0.5 * market.consumption.sel(commodity=enduse).interp(year=2015).values
    )
