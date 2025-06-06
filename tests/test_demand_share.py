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
    """Helper to create test agents with standard configuration.

    Args:
        usa_stock: Stock data for USA region
        asia_stock: Optional stock data for ASEAN region
        with_new: Whether to include new capacity agents

    Returns:
        List of Agent objects with specified configurations
    """
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
    """Create market data for given regions.

    Args:
        technologies: Technology parameters
        stock: Stock data containing regional information

    Returns:
        Tuple of (market data, asia stock subset, usa stock subset)
    """
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
    """Create interpolated capacity fixture."""
    return interpolate_capacity(stock.capacity, year=[CURRENT_YEAR, INVESTMENT_YEAR])


@fixture
def _technologies(technologies, _capacity):
    """Create technology parameters fixture for the sector."""
    return technologies.interp(year=INVESTMENT_YEAR)


@fixture
def _market(_technologies, _capacity, timeslice):
    """Create a market fixture which matches stocks exactly."""
    _technologies = broadcast_over_assets(_technologies, _capacity)
    return _matching_market(_technologies, _capacity).transpose(
        "timeslice", "region", "commodity", "year"
    )


def _matching_market(technologies, capacity):
    """Create a market which matches stocks exactly.

    Args:
        technologies: Technology parameters
        capacity: Capacity data

    Returns:
        Market dataset with supply, consumption and prices
    """
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


def verify_share_values(share, expect_new_zero=True, expect_retrofit_nonzero=True):
    """Helper to verify share values under different scenarios.

    Args:
        share: Share values to verify
        expect_new_zero: Whether new capacity share should be zero
        expect_retrofit_nonzero: Whether retrofit share should be non-zero
    """
    assert (share.new == 0).all() if expect_new_zero else (share.new != 0).any()
    assert (
        (share.retrofit != 0).any()
        if expect_retrofit_nonzero
        else (share.retrofit == 0).all()
    )


def test_fixtures(_capacity, _market, _technologies):
    """Verify that test fixtures have the expected dimensions."""
    assert set(_capacity.dims) == {"asset", "year"}
    assert set(_market.dims) == {"commodity", "region", "year", "timeslice"}
    assert set(_technologies.dims) == {"technology", "region", "commodity"}


def test_new_retro_split_zero_unmet(_capacity, _market, _technologies):
    """Test that new and retrofit demands are zero when demand is fully met."""
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()


def test_new_retro_split_scenarios(_capacity, _market, _technologies):
    """Test various scenarios for new and retrofit demand splits.

    Tests:
    1. Same consumption in investment year
    2. Reduced future capacity
    3. Reduced current capacity
    4. Overall reduced capacity
    5. Market supply matching consumption
    """
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)

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
    verify_share_values(share)

    # Test with reduced current capacity
    current_unmet = _capacity.copy()
    current_unmet.loc[{"year": CURRENT_YEAR}] = 0.5 * future_unmet.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(current_unmet, _market.consumption, _technologies)
    verify_share_values(share)

    # Test with overall reduced capacity
    share = new_and_retro_demands(0.5 * _capacity, _market.consumption, _technologies)
    verify_share_values(share)

    # Test with market supply matching consumption
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.supply.sel(
        year=CURRENT_YEAR, drop=True
    ).transpose(*_market.consumption.loc[{"year": INVESTMENT_YEAR}].dims)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()


def test_new_retro_split_zero_consumption_increase(_capacity, _market, _technologies):
    """Test new and retrofit demand splits with no consumption increase.

    Tests various capacity scenarios when consumption remains constant.
    """
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)

    # Base case - same consumption
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.consumption.sel(
        year=CURRENT_YEAR
    )
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()

    # Test capacity reduction scenarios
    scenarios = [
        ("future", lambda x: x.loc[{"year": INVESTMENT_YEAR}], 0.5),
        ("current", lambda x: x.loc[{"year": CURRENT_YEAR}], 0.5),
        ("overall", lambda x: x, 0.5),
    ]

    for name, selector, factor in scenarios:
        modified_capacity = _capacity.copy()
        selector(modified_capacity)[:] = factor * modified_capacity.sel(
            year=CURRENT_YEAR
        )
        share = new_and_retro_demands(
            modified_capacity, _market.consumption, _technologies
        )
        verify_share_values(share)


def test_new_retro_split_zero_new_unmet(_capacity, _market, _technologies):
    """Test new and retrofit demand splits with zero new unmet demand.

    Tests that retrofit demand is properly allocated when there is no new unmet demand.
    """
    from muse.demand_share import new_and_retro_demands

    _technologies = broadcast_over_assets(_technologies, _capacity)

    # Set market consumption to match current supply
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.supply.sel(
        year=CURRENT_YEAR, drop=True
    ).transpose(*_market.consumption.loc[{"year": INVESTMENT_YEAR}].dims)
    share = new_and_retro_demands(_capacity, _market.consumption, _technologies)
    assert (share == 0).all()

    # Test capacity reduction scenarios
    scenarios = [
        ("future", lambda x: x.loc[{"year": INVESTMENT_YEAR}], 0.5),
        ("current", lambda x: x.loc[{"year": CURRENT_YEAR}], 0.5),
        ("overall", lambda x: x, 0.5),
    ]

    for name, selector, factor in scenarios:
        modified_capacity = _capacity.copy()
        selector(modified_capacity)[:] = factor * modified_capacity.sel(
            year=CURRENT_YEAR
        )
        share = new_and_retro_demands(
            modified_capacity, _market.consumption, _technologies
        )
        verify_share_values(share)


def test_new_retro_accounting_identity(_capacity, _market, _technologies):
    """Test that new and retrofit demands satisfy accounting identity.

    Verifies that the sum of new and retrofit demands plus serviced demand equals total
    demand.
    """
    from muse.demand_share import new_and_retro_demands

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

    # Verify accounting identity components
    assert (share.new > -1e-8).all()
    assert (share.retrofit > -1e-8).all()
    assert ((share.new + share.retrofit).where(consumption < serviced, 0) < 1e-8).all()

    # Verify total accounting identity
    accounting = (
        (share.new + share.retrofit + serviced)
        .where(consumption - serviced > 0, consumption)
        .transpose(*consumption.dims)
    )
    assert accounting.values == approx(consumption.values)


def test_demand_split_scenarios(_capacity, _market, _technologies):
    """Test demand split scenarios with different agent configurations.

    Tests demand splitting between agents with different quantities and verifies
    that shares are properly allocated.
    """
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
    """Test new and retrofit demand share calculations.

    Verifies that demand is properly shared between new and retrofit agents
    across different regions.
    """
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
    """Test standard demand share calculations.

    Verifies that:
    1. Retrofit agents raise appropriate error
    2. New agents receive proper demand shares
    """
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
    """Test unmet forecast demand calculations.

    Tests three scenarios:
    1. Fully met demand - agents have exact capacity to meet demand
    2. Excess capacity - agents have more capacity than needed
    3. Insufficient capacity - agents have less capacity than needed
    """
    from muse.commodities import is_enduse
    from muse.demand_share import unmet_forecasted_demand

    # Setup market data
    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)

    # Test scenario 1: Fully met demand
    agents = create_test_agents(usa_stock, asia_stock)
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Test scenario 2: Excess capacity (120% capacity)
    agents = create_test_agents(1.2 * usa_stock, 1.2 * asia_stock)
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Test scenario 3: Insufficient capacity (50% capacity)
    agents = create_test_agents(0.5 * usa_stock, 0.5 * asia_stock)
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)

    # Verify results for insufficient capacity
    enduse = is_enduse(_technologies.comm_usage.sel(commodity=market.commodity))
    assert (result.commodity == market.commodity).all()
    assert result.sel(commodity=~enduse).values == approx(0)
    assert result.sel(commodity=enduse).values == approx(
        0.5 * market.consumption.sel(commodity=enduse, year=2030).values
    )


def test_decommissioning_demand(_technologies, _capacity, timeslice):
    """Test decommissioning demand calculations.

    Verifies that decommissioning demand is correctly calculated based on:
    1. Capacity changes between current and forecast years
    2. Fixed outputs and utilization factors
    """
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
