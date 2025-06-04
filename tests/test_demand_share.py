from dataclasses import dataclass
from uuid import UUID, uuid4

import xarray as xr
from pytest import approx, fixture, raises

from muse.commodities import is_enduse
from muse.quantities import maximum_production
from muse.timeslices import drop_timeslice
from muse.utilities import broadcast_over_assets, interpolate_capacity

CURRENT_YEAR = 2010
INVESTMENT_YEAR = 2015


@dataclass
class Agent:
    """Test agent with required attributes."""

    assets: xr.Dataset
    category: str = ""
    uuid: UUID = None
    name: str = ""
    region: str = ""
    quantity: float = 0.0


def create_test_agents(usa_stock, asia_stock=None, with_new=True):
    """Helper to create test agents with standard configuration."""
    agents = [
        Agent(0.3 * usa_stock, "retrofit", uuid4(), "a", "USA", 0.3),
        Agent(0.7 * usa_stock, "retrofit", uuid4(), "b", "USA", 0.7),
    ]
    if with_new:
        agents.extend(
            [
                Agent(0.0 * usa_stock, "new", uuid4(), "a", "USA", 0.0),
                Agent(0.0 * usa_stock, "new", uuid4(), "b", "USA", 0.0),
            ]
        )
    if asia_stock is not None:
        agents.extend(
            [
                Agent(asia_stock, "retrofit", uuid4(), "a", "ASEAN", 1.0),
                Agent(0 * asia_stock, "new", uuid4(), "a", "ASEAN", 0.0)
                if with_new
                else None,
            ]
        )
    return [a for a in agents if a is not None]


def create_regional_market(technologies, stock):
    """Create market data for given regions."""
    asia_stock = stock.where(stock.region == "ASEAN", drop=True)
    usa_stock = stock.where(stock.region == "USA", drop=True)

    asia_market = _matching_market(
        broadcast_over_assets(technologies, asia_stock), asia_stock.capacity
    )
    usa_market = _matching_market(
        broadcast_over_assets(technologies, usa_stock), usa_stock.capacity
    )
    market = xr.concat((asia_market, usa_market), dim="region")
    return market, asia_stock, usa_stock


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
    _technologies = broadcast_over_assets(_technologies, _capacity)
    return _matching_market(_technologies, _capacity).transpose(
        "timeslice", "region", "commodity", "year"
    )


def _matching_market(technologies, capacity):
    """A market which matches stocks exactly."""
    from numpy.random import random

    from muse.quantities import consumption as calc_consumption

    market = xr.Dataset()
    production = maximum_production(technologies, capacity)
    cons = calc_consumption(technologies, production)
    if "region" in production.coords:
        production = production.groupby("region")
        cons = cons.groupby("region")
    market["supply"] = production.sum("asset")
    market["consumption"] = drop_timeslice(cons.sum("asset") + market.supply)
    market["prices"] = market.supply.dims, random(market.supply.shape)
    return market


def test_fixtures(_capacity, _market, _technologies):
    assert set(_capacity.dims) == {"asset", "year"}
    assert set(_market.dims) == {"commodity", "region", "year", "timeslice"}
    assert set(_technologies.dims) == {"technology", "region", "commodity"}


def test_new_retro_split_zero_unmet(_capacity, _market, _technologies):
    """Test that new and retro demands are zero when demand is fully met."""
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()


def test_new_retro_split_scenarios(_capacity, _market, _technologies):
    """Test various scenarios for new and retro demand splits."""
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)

    def check_share_values(share, expect_new_zero=True, expect_retrofit_nonzero=True):
        """Helper to check share values under different scenarios."""
        assert (share.new == 0).all() if expect_new_zero else (share.new != 0).any()
        assert (
            (share.retrofit != 0).any()
            if expect_retrofit_nonzero
            else (share.retrofit == 0).all()
        )

    # Test with same consumption in investment year
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.consumption.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()

    # Test with reduced future capacity
    future_unmet = _capacity.copy()
    future_unmet.loc[{"year": INVESTMENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(future_unmet, _market.consumption, _technologies)
    check_share_values(share)

    # Test with reduced current capacity
    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market.consumption, _technologies)
    check_share_values(share)

    # Test with overall reduced capacity
    share = new_and_retro_demands(0.5 * _capacity, _market.consumption, _technologies)
    check_share_values(share)

    # Test with market supply matching consumption
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.supply.sel(
        year=CURRENT_YEAR, drop=True
    ).transpose(*_market.consumption.loc[{"year": INVESTMENT_YEAR}].dims)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()


def test_new_retro_split_zero_consumption_increase(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.consumption.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()

    future_unmet = _capacity.copy()
    future_unmet.loc[{"year": INVESTMENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(future_unmet, _market.consumption, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market.consumption, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(0.5 * _capacity, _market.consumption, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_split_zero_new_unmet(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.supply.sel(
        year=CURRENT_YEAR, drop=True
    ).transpose(*_market.consumption.loc[{"year": INVESTMENT_YEAR}].dims)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()

    future_unmet = _capacity.copy()
    future_unmet.loc[{"year": INVESTMENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(future_unmet, _market.consumption, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market.consumption, _technologies)
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()

    share = new_and_retro_demands(
        0.5 * _capacity,
        _market.consumption,
        _technologies,
    )
    assert (share.new == 0).all()
    assert (share.retrofit != 0).any()


def test_new_retro_accounting_identity(_capacity, _market, _technologies):
    from muse.demand_share import new_and_retro_demands
    from muse.utilities import broadcast_over_assets

    _technologies = broadcast_over_assets(_technologies, _capacity)

    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
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


def test_demand_split_scenarios(_capacity, _market, _technologies):
    """Test demand split scenarios with different agent configurations."""
    from muse.demand_share import _inner_split as inner_split
    from muse.demand_share import decommissioning_demand

    def get_test_demand():
        """Get test demand data for USA region."""
        return _market.consumption.sel(
            year=INVESTMENT_YEAR, region="USA", drop=True
        ).where(is_enduse(_technologies.comm_usage.sel(commodity=_market.commodity)))

    def check_share_results(share, agents_data, expected_shares):
        """Verify share results match expectations."""
        enduse = is_enduse(_technologies.comm_usage)
        for agent_name in agents_data.keys():
            assert (share[agent_name].sel(commodity=~enduse) == 0).all()

        total = sum(share.values()).sum("asset")
        demand = get_test_demand().where(enduse, 0)
        demand, total = xr.broadcast(demand, total)
        assert demand.values == approx(total.values)

        for agent_name, expected_share in expected_shares.items():
            expected, actual = xr.broadcast(demand, share[agent_name].sum("asset"))
            assert actual.values == approx(expected_share * expected.values)

    # Test normal demand split
    _technologies = broadcast_over_assets(_technologies, _capacity)
    agents = dict(scully=_capacity, mulder=_capacity)
    technodata = dict(scully=_technologies, mulder=_technologies)
    quantity = dict(scully=("scully", "USA", 0.3), mulder=("mulder", "USA", 0.7))

    share = inner_split(
        agents, technodata, get_test_demand(), decommissioning_demand, quantity
    )
    check_share_results(share, agents, {"scully": 0.3, "mulder": 0.7})

    # Test zero share scenario
    agents = dict(scully=0.3 * _capacity, mulder=0.7 * _capacity)
    quantity = dict(scully=("scully", "USA", 1), mulder=("mulder", "USA", 1))

    def zero_decom(technologies, capacity):
        """Return zero decommissioning demand."""
        return 0 * decommissioning_demand(technologies=technologies, capacity=capacity)

    share = inner_split(agents, technodata, get_test_demand(), zero_decom, quantity)
    check_share_results(share, agents, {"scully": 0.5, "mulder": 0.5})


def test_new_retro_demand_share(_technologies, market, timeslice, stock):
    """Test new and retro demand share calculations."""
    from muse.demand_share import new_and_retro

    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)
    market.consumption.loc[{"year": 2030}] *= 2
    agents = create_test_agents(usa_stock, asia_stock)
    results = new_and_retro(agents, market.consumption, _technologies)

    # Verify results for each agent
    uuid_to_category = {agent.uuid: agent.category for agent in agents}
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    for category in {"retrofit", "new"}:
        subset = {
            uuid_to_name[uuid]: share.sel(commodity=is_enduse(_technologies.comm_usage))
            for uuid, share in results.groupby("agent")
            if uuid_to_category[uuid] == category and (share.region == "USA").all()
        }
        if subset:
            expected, actual = xr.broadcast(0.3 * sum(subset.values()), subset["a"])
            assert actual.values == approx(expected.values)


def test_standard_demand_share(_technologies, timeslice, stock):
    """Test standard demand share calculations."""
    from muse.demand_share import standard_demand
    from muse.errors import RetrofitAgentInStandardDemandShare

    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)
    market.consumption.loc[{"year": 2030}] *= 2

    # Test that retrofit agents raise error
    with raises(RetrofitAgentInStandardDemandShare):
        standard_demand(
            create_test_agents(usa_stock, asia_stock), market.consumption, _technologies
        )

    # Test with only new agents
    agents = [
        Agent(0.3 * usa_stock, "new", uuid4(), "a", "USA", 0.3),
        Agent(0.7 * usa_stock, "new", uuid4(), "b", "USA", 0.7),
        Agent(asia_stock, "new", uuid4(), "a", "ASEAN", 1.0),
    ]
    results = standard_demand(agents, market.consumption, _technologies)

    # Verify results
    uuid_to_name = {agent.uuid: agent.name for agent in agents}
    subset = {
        uuid_to_name[uuid]: share.sel(commodity=is_enduse(_technologies.comm_usage))
        for uuid, share in results.groupby("agent")
        if (share.region == "USA").all()
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
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
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
        market.consumption,
        _technologies,
    )
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Then try too little capacity
    agents = [
        Agent(0.5 * usa_stock),
        Agent(0.5 * asia_stock),
    ]
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
    comm_usage = _technologies.comm_usage.sel(commodity=market.commodity)
    enduse = is_enduse(comm_usage)
    assert (result.commodity == comm_usage.commodity).all()
    assert result.sel(commodity=~enduse).values == approx(0)
    assert result.sel(commodity=enduse).values == approx(
        0.5 * market.consumption.sel(commodity=enduse, year=2030).values
    )


def test_decommissioning_demand(_technologies, _capacity, timeslice):
    """Test decommissioning demand calculations."""
    from muse.demand_share import decommissioning_demand

    _technologies = broadcast_over_assets(_technologies, _capacity)

    # Set test values
    current, forecast = 1.3, 1.0
    fixed_outputs, utilization = 0.5, 0.4

    _capacity.loc[{"year": CURRENT_YEAR}] = current
    _capacity.loc[{"year": INVESTMENT_YEAR}] = forecast
    _technologies.fixed_outputs[:] = fixed_outputs
    _technologies.utilization_factor[:] = utilization

    # Calculate and verify decommissioning demand
    decom = decommissioning_demand(_technologies, _capacity)
    assert set(decom.dims) == {"asset", "commodity", "timeslice"}

    expected_decom = utilization * fixed_outputs * (current - forecast)
    assert decom.sel(commodity=is_enduse(_technologies.comm_usage)).sum(
        "timeslice"
    ).values == approx(expected_decom)
