import xarray as xr
from pytest import approx, fixture, raises

from muse.timeslices import drop_timeslice
from muse.utilities import interpolate_capacity

CURRENT_YEAR = 2010
INVESTMENT_YEAR = 2015


@fixture
def _capacity(stock):
    return interpolate_capacity(stock.capacity, year=[CURRENT_YEAR, INVESTMENT_YEAR])


@fixture
def _technologies(technologies, _capacity):
    """Technology parameters for the sector."""
    return technologies.interp(year=INVESTMENT_YEAR)


@fixture
def _market(_technologies, _capacity, timeslice):
    """A market which matches stocks exactly."""
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)
    return _matching_market(_technologies, _capacity).transpose(
        "timeslice", "region", "commodity", "year"
    )


def _matching_market(technologies, capacity):
    """A market which matches stocks exactly."""
    from numpy.random import random

    from muse.quantities import consumption, maximum_production

    market = xr.Dataset()
    production = maximum_production(technologies, capacity)
    consumption = consumption(technologies, production)
    if "region" in production.coords:
        production = production.groupby("region")
        consumption = consumption.groupby("region")
    market["supply"] = production.sum("asset")
    market["consumption"] = drop_timeslice(consumption.sum("asset") + market.supply)
    market["prices"] = market.supply.dims, random(market.supply.shape)
    return market


def test_fixtures(_capacity, _market, _technologies):
    assert set(_capacity.dims) == {"asset", "year"}
    assert set(_market.dims) == {"commodity", "region", "year", "timeslice"}
    assert set(_technologies.dims) == {"technology", "region", "commodity"}


def test_new_retro_split_zero_unmet(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)
    share = new_and_retro_demands(_capacity, _market, _technologies)
    assert (share == 0).all()


def test_new_retro_split_zero_consumption_increase(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.consumption.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(_capacity, _market, _technologies)
    assert (share == 0).all()

    future_unmet = _capacity.copy()
    future_unmet.loc[{"year": INVESTMENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(future_unmet, _market, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(0.5 * _capacity, _market, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_split_zero_new_unmet(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.supply.sel(
        year=CURRENT_YEAR, drop=True
    ).transpose(*_market.consumption.loc[{"year": INVESTMENT_YEAR}].dims)
    share = new_and_retro_demands(_capacity, _market, _technologies)
    assert (share == 0).all()

    future_unmet = _capacity.copy()
    future_unmet.loc[{"year": INVESTMENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(future_unmet, _market, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(
        0.5 * _capacity,
        _market,
        _technologies,
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_accounting_identity(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.quantities import maximum_production
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    share = new_and_retro_demands(_capacity, _market, _technologies)
    assert (share >= 0).all()

    serviced = (
        maximum_production(
            capacity=_capacity.sel(year=INVESTMENT_YEAR),
            technologies=_technologies,
        )
        .groupby("region")
        .sum("asset")
    )
    consumption = _market.consumption.sel(year=INVESTMENT_YEAR)

    assert (share.new > -1e-8).all()
    assert (share.retrofit > -1e-8).all()
    assert ((share.new + share.retrofit).where(consumption < serviced, 0) < 1e-8).all()
    accounting = (
        (share.new + share.retrofit + serviced)
        .where(consumption - serviced > 0, consumption)
        .transpose(*consumption.dims)
    )
    assert accounting.values == approx(consumption.values)


def test_demand_split(_capacity, _market, _technologies):
    from muse.commodities import is_enduse
    from muse.demand_share import _inner_split as inner_split
    from muse.utilities import broadcast_over_assets

    def method(capacity, technologies):
        from muse.demand_share import decommissioning_demand

        return decommissioning_demand(
            technologies,
            capacity,
        )

    demand = _market.consumption.sel(
        year=INVESTMENT_YEAR, region="USA", drop=True
    ).where(is_enduse(_technologies.comm_usage.sel(commodity=_market.commodity)))
    agents = dict(scully=_capacity, mulder=_capacity)
    _technologies = broadcast_over_assets(_technologies, _capacity)
    technodata = dict(scully=_technologies, mulder=_technologies)
    quantity = dict(scully=("scully", "USA", 0.3), mulder=("mulder", "USA", 0.7))
    share = inner_split(agents, technodata, demand, method, quantity)

    enduse = is_enduse(_technologies.comm_usage)
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


def test_demand_split_zero_share(_capacity, _market, _technologies):
    """See issue SgiModel/StarMuse#688."""
    from muse.commodities import is_enduse
    from muse.demand_share import _inner_split as inner_split
    from muse.utilities import broadcast_over_assets

    def method(capacity, technologies):
        from muse.demand_share import decommissioning_demand

        return 0 * decommissioning_demand(
            technologies,
            capacity,
        )

    demand = _market.consumption.sel(
        year=INVESTMENT_YEAR, region="USA", drop=True
    ).where(is_enduse(_technologies.comm_usage.sel(commodity=_market.commodity)))
    agents = dict(scully=0.3 * _capacity, mulder=0.7 * _capacity)
    _technologies = broadcast_over_assets(_technologies, _capacity)
    technodata = dict(scully=_technologies, mulder=_technologies)
    quantity = dict(scully=("scully", "USA", 1), mulder=("mulder", "USA", 1))
    share = inner_split(agents, technodata, demand, method, quantity)

    enduse = is_enduse(_technologies.comm_usage)
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


def test_new_retro_demand_share(_technologies, market, timeslice, stock):
    from dataclasses import dataclass
    from uuid import UUID, uuid4

    from muse.commodities import is_enduse
    from muse.demand_share import new_and_retro
    from muse.utilities import broadcast_over_assets

    asia_stock = stock.where(stock.region == "ASEAN", drop=True)
    usa_stock = stock.where(stock.region == "USA", drop=True)

    asia_market = _matching_market(
        broadcast_over_assets(_technologies, asia_stock), asia_stock.capacity
    )
    usa_market = _matching_market(
        broadcast_over_assets(_technologies, usa_stock), usa_stock.capacity
    )
    market = xr.concat((asia_market, usa_market), dim="region")
    market.consumption.loc[{"year": 2030}] *= 2

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
        Agent(0.3 * usa_stock, "retrofit", uuid4(), "a", "USA", 0.3),
        Agent(0.0 * usa_stock, "new", uuid4(), "a", "USA", 0.0),
        Agent(0.7 * usa_stock, "retrofit", uuid4(), "b", "USA", 0.7),
        Agent(0.0 * usa_stock, "new", uuid4(), "b", "USA", 0.0),
        Agent(asia_stock, "retrofit", uuid4(), "a", "ASEAN", 1.0),
        Agent(0 * asia_stock, "new", uuid4(), "a", "ASEAN", 0.0),
    ]

    results = new_and_retro(agents, market, _technologies)

    for _, share in results.groupby("agent"):
        assert share.sel(
            commodity=~is_enduse(_technologies.comm_usage)
        ).values == approx(0)

    uuid_to_category = {agent.uuid: agent.category for agent in agents}
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    for category in {"retrofit", "new"}:
        subset = {
            uuid_to_name[uuid]: share.sel(commodity=is_enduse(_technologies.comm_usage))
            for uuid, share in results.groupby("agent")
            if uuid_to_category[uuid] == category and (share.region == "USA").all()
        }
        expected, actual = xr.broadcast(0.3 * sum(subset.values()), subset["a"])
        assert actual.values == approx(expected.values)


def test_standard_demand_share(_technologies, timeslice, stock):
    from dataclasses import dataclass
    from uuid import UUID, uuid4

    from muse.commodities import is_enduse
    from muse.demand_share import standard_demand
    from muse.errors import RetrofitAgentInStandardDemandShare
    from muse.utilities import broadcast_over_assets

    asia_stock = stock.where(stock.region == "ASEAN", drop=True)
    usa_stock = stock.where(stock.region == "USA", drop=True)

    asia_market = _matching_market(
        broadcast_over_assets(_technologies, asia_stock), asia_stock.capacity
    )
    usa_market = _matching_market(
        broadcast_over_assets(_technologies, usa_stock), usa_stock.capacity
    )
    market = xr.concat((asia_market, usa_market), dim="region")
    market.consumption.loc[{"year": 2030}] *= 2

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
        Agent(0.3 * usa_stock, "retrofit", uuid4(), "a", "USA", 0.3),
        Agent(0.0 * usa_stock, "new", uuid4(), "a", "USA", 0.0),
        Agent(0.7 * usa_stock, "retrofit", uuid4(), "b", "USA", 0.7),
        Agent(0.0 * usa_stock, "new", uuid4(), "b", "USA", 0.0),
        Agent(asia_stock, "retrofit", uuid4(), "a", "ASEAN", 1.0),
        Agent(0 * asia_stock, "new", uuid4(), "a", "ASEAN", 0.0),
    ]

    with raises(RetrofitAgentInStandardDemandShare):
        standard_demand(agents, market, _technologies)

    agents = [a for a in agents if a.category != "retrofit"]

    results = standard_demand(agents, market, _technologies)

    uuid_to_category = {agent.uuid: agent.category for agent in agents}
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    subset = {
        uuid_to_name[uuid]: share.sel(commodity=is_enduse(_technologies.comm_usage))
        for uuid, share in results.groupby("agent")
        if uuid_to_category[uuid] == "new" and (share.region == "USA").all()
    }
    expected, actual = xr.broadcast(0.3 * sum(subset.values()), subset["a"])
    assert actual.values == approx(expected.values)


def test_unmet_forecast_demand(_technologies, timeslice, stock):
    from dataclasses import dataclass

    from muse.commodities import is_enduse
    from muse.demand_share import unmet_forecasted_demand
    from muse.utilities import broadcast_over_assets

    asia_stock = stock.where(stock.region == "ASEAN", drop=True)
    usa_stock = stock.where(stock.region == "USA", drop=True)

    asia_market = _matching_market(
        broadcast_over_assets(_technologies, asia_stock), asia_stock.capacity
    )
    usa_market = _matching_market(
        broadcast_over_assets(_technologies, usa_stock), usa_stock.capacity
    )
    market = xr.concat((asia_market, usa_market), dim="region")

    # spoof some agents
    @dataclass
    class Agent:
        assets: xr.Dataset

    # First ensure that the demand is fully met
    agents = [
        Agent(0.3 * usa_stock),
        Agent(0.7 * usa_stock),
        Agent(asia_stock),
    ]
    result = unmet_forecasted_demand(agents, market, _technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Then try with too little demand
    agents = [
        Agent(0.4 * usa_stock),
        Agent(0.8 * usa_stock),
        Agent(1.1 * asia_stock),
    ]
    result = unmet_forecasted_demand(
        agents,
        market,
        _technologies,
    )
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Then try too little capacity
    agents = [
        Agent(0.5 * usa_stock),
        Agent(0.5 * asia_stock),
    ]
    result = unmet_forecasted_demand(agents, market, _technologies)
    comm_usage = _technologies.comm_usage.sel(commodity=market.commodity)
    enduse = is_enduse(comm_usage)
    assert (result.commodity == comm_usage.commodity).all()
    assert result.sel(commodity=~enduse).values == approx(0)
    assert result.sel(commodity=enduse).values == approx(
        0.5 * market.consumption.sel(commodity=enduse, year=2030).values
    )


def test_decommissioning_demand(_technologies, _capacity, timeslice):
    from muse.commodities import is_enduse
    from muse.demand_share import decommissioning_demand
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    _capacity.loc[{"year": CURRENT_YEAR}] = current = 1.3
    _capacity.loc[{"year": INVESTMENT_YEAR}] = forecast = 1.0
    _technologies.fixed_outputs[:] = fouts = 0.5
    _technologies.utilization_factor[:] = ufac = 0.4
    decom = decommissioning_demand(_technologies, _capacity)
    assert set(decom.dims) == {"asset", "commodity", "timeslice"}
    assert decom.sel(commodity=is_enduse(_technologies.comm_usage)).sum(
        "timeslice"
    ).values == approx(ufac * fouts * (current - forecast))
