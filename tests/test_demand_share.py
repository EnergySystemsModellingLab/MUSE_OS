from dataclasses import dataclass
from uuid import UUID, uuid4

import xarray as xr
from pytest import approx, fixture

from muse.commodities import is_enduse
from muse.quantities import maximum_production
from muse.timeslices import drop_timeslice
from muse.utilities import broadcast_over_assets, interpolate_capacity

CURRENT_YEAR = 2010
INVESTMENT_YEAR = 2030


@dataclass
class Agent:
    """Test agent with required attributes."""

    assets: xr.Dataset
    category: str = ""
    uuid: UUID = None
    name: str = ""
    region: str = ""
    quantity: float = 0.0


def create_test_agents(usa_stock, asia_stock=None, categories=None):
    """Helper to create test agents with standard configuration.

    Args:
        usa_stock: Stock data for USA region
        asia_stock: Optional stock data for ASEAN region
        categories: List of agent categories to create. If None, creates both "retrofit"
            and "newcapa"

    Returns:
        List of Agent objects with specified configurations
    """
    if categories is None:
        categories = ["retrofit", "newcapa"]

    # Agent configurations
    usa_configs = {
        "retrofit": [(0.3, "a"), (0.7, "b")],
        "newcapa": [(0.5, "a"), (0.5, "b")],
    }
    asia_configs = {
        "retrofit": [(0.6, "a"), (0.4, "b")],
        "newcapa": [(0.5, "a"), (0.5, "b")],
    }

    agents = []

    # Create USA agents
    for category in categories:
        for multiplier, name in usa_configs[category]:
            agents.append(
                Agent(
                    multiplier * usa_stock, category, uuid4(), name, "USA", multiplier
                )
            )

    # Create ASEAN agents if stock provided
    if asia_stock is not None:
        for category in categories:
            for multiplier, name in asia_configs[category]:
                agents.append(
                    Agent(
                        multiplier * asia_stock,
                        category,
                        uuid4(),
                        name,
                        "ASEAN",
                        multiplier,
                    )
                )

    return agents


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
def _technologies(technologies, _capacity, timeslice):
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
    """Create a market which matches stocks exactly."""
    from numpy.random import random

    from muse.commodities import is_enduse
    from muse.quantities import consumption as calc_consumption

    # Calculate production and consumption
    production = maximum_production(technologies, capacity)
    cons = calc_consumption(technologies, production)

    # Handle regional grouping if needed
    if "region" in production.coords:
        production = production.groupby("region")
        cons = cons.groupby("region")

    market = xr.Dataset()
    market["supply"] = production.sum("asset")

    # Create consumption with only enduse commodities having non-zero demand
    consumption = drop_timeslice(cons.sum("asset") + market.supply)
    enduse_names = technologies.commodity[is_enduse(technologies.comm_usage)]

    # Zero out non-enduse commodities
    non_enduse = consumption.commodity[~consumption.commodity.isin(enduse_names)]
    consumption.loc[{"commodity": non_enduse}] = 0

    market["consumption"] = consumption
    market["prices"] = market.supply.dims, random(market.supply.shape)
    return market


def verify_share_values(share, expect_new_zero=True, expect_retrofit_nonzero=True):
    """Helper to verify share values under different scenarios."""
    if expect_new_zero:
        assert (share.new == 0).all()
    else:
        assert (share.new != 0).any()

    if expect_retrofit_nonzero:
        assert (share.retrofit != 0).any()
    else:
        assert (share.retrofit == 0).all()


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
    """Test various scenarios for new and retrofit demand splits."""
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

    # Verify total accounting identity: Total Demand = Serviced + New + Retrofit
    total_demand = consumption
    total_accounted = serviced + share.new + share.retrofit
    assert total_accounted.values == approx(total_demand.values)


def test_unmet_forecast_demand(_technologies, stock):
    """Test unmet forecast demand calculations."""
    from muse.commodities import is_enduse
    from muse.demand_share import unmet_forecasted_demand

    # Setup market data
    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)

    # Test scenario 1: Fully met demand
    agents = create_test_agents(usa_stock, asia_stock, categories=["newcapa"])
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Test scenario 2: Excess capacity (120% capacity)
    agents = create_test_agents(
        1.2 * usa_stock, 1.2 * asia_stock, categories=["newcapa"]
    )
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)
    assert set(result.dims) == set(market.consumption.dims) - {"year"}
    assert result.values == approx(0)

    # Test scenario 3: Insufficient capacity (50% capacity)
    agents = create_test_agents(
        0.5 * usa_stock, 0.5 * asia_stock, categories=["newcapa"]
    )
    result = unmet_forecasted_demand(agents, market.consumption, _technologies)

    # Verify results for insufficient capacity
    enduse = is_enduse(_technologies.comm_usage.sel(commodity=market.commodity))
    assert (result.commodity == market.commodity).all()
    assert result.sel(commodity=~enduse).values == approx(0)
    assert result.sel(commodity=enduse).values == approx(
        0.5 * market.consumption.sel(commodity=enduse, year=INVESTMENT_YEAR).values
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


def test_inner_split_basic(_capacity, _market, _technologies):
    """Test basic functionality of _inner_split.

    Tests that demand is split proportionally according to the method function
    using a mock function that returns predetermined shares.
    """
    from muse.demand_share import _inner_split
    from muse.utilities import broadcast_over_assets

    # Select a region to test
    REGION = "ASEAN"
    _capacity = _capacity.where(_capacity.asset.region == REGION)
    _technologies = _technologies.sel(region=REGION)
    _market = _market.sel(region=REGION)

    # Broadcast technologies over assets
    tech_data = broadcast_over_assets(_technologies, _capacity)

    # Demand to split over assets
    demand = _market.consumption.sel(year=INVESTMENT_YEAR, drop=True)

    # Test with maximum production method
    shares = maximum_production(
        capacity=_capacity.sel(year=CURRENT_YEAR, drop=True), technologies=tech_data
    )

    result = _inner_split(
        demand=demand,
        shares=shares,
    )

    # Check dimensions
    assert set(result.dims) == {"asset", "commodity", "timeslice"}

    # Check total demand is preserved (conservation of demand)
    assert result.sum("asset").values == approx(demand.values)

    # Check all values are non-negative
    assert (result >= 0).all()


def test_inner_split_zero_shares(_capacity, _market, _technologies):
    """Test _inner_split when method returns zero shares.

    Tests that unassigned demand is split equally when method returns zero shares.
    """
    from muse.demand_share import _inner_split
    from muse.quantities import maximum_production
    from muse.utilities import broadcast_over_assets

    # Select a region to test
    REGION = "ASEAN"
    _capacity = _capacity.where(_capacity.asset.region == REGION)
    _technologies = _technologies.sel(region=REGION)
    _market = _market.sel(region=REGION)

    # Broadcast technologies over assets
    tech_data = broadcast_over_assets(_technologies, _capacity)

    # Demand in the investment year to split over assets
    demand = _market.consumption.sel(year=INVESTMENT_YEAR, drop=True)

    # Test with zero production method
    zero_shares = 0 * maximum_production(
        capacity=_capacity.sel(year=CURRENT_YEAR, drop=True), technologies=tech_data
    )

    result = _inner_split(
        demand=demand,
        shares=zero_shares,
    )

    # Check dimensions
    assert set(result.dims) == {"asset", "commodity", "timeslice"}

    # Check total demand is preserved (conservation of demand)
    assert result.sum("asset").values == approx(demand.values)

    # Check demand is split equally among assets
    expected_per_asset = demand / len(result.asset)
    assert (result == expected_per_asset).all()


def test_unmet_demand(_capacity, _market, _technologies):
    """Test unmet demand calculations."""
    from muse.demand_share import unmet_demand

    _technologies = broadcast_over_assets(_technologies, _capacity)

    # Select single year data (unmet_demand doesn't handle year dimension)
    capacity = _capacity.sel(year=CURRENT_YEAR, drop=True)
    demand = _market.consumption.sel(year=CURRENT_YEAR, drop=True)

    # Test scenario 1: Fully met demand (excess capacity)
    excess_capacity = 2.0 * capacity
    result = unmet_demand(demand, excess_capacity, _technologies)
    assert result.values == approx(0)

    # Test scenario 2: Insufficient capacity (50% capacity)
    insufficient_capacity = 0.5 * capacity
    result = unmet_demand(demand, insufficient_capacity, _technologies)
    assert (result > 0).any()
    assert (result <= demand).all()

    # Test scenario 3: Zero capacity
    zero_capacity = 0 * capacity
    result = unmet_demand(demand, zero_capacity, _technologies)
    assert result.values == approx(demand.values)


def test_new_consumption(_capacity, _market, _technologies):
    """Test new consumption calculation."""
    from muse.demand_share import new_consumption

    _technologies = broadcast_over_assets(_technologies, _capacity)

    # Test with no demand growth
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = _market.consumption.sel(
        year=CURRENT_YEAR
    )
    result = new_consumption(_capacity, _market.consumption, _technologies)
    assert (result == 0).all()

    # Test with demand growth but sufficient capacity
    _market.consumption.loc[{"year": INVESTMENT_YEAR}] = 1.5 * _market.consumption.sel(
        year=CURRENT_YEAR
    )
    result = new_consumption(_capacity, _market.consumption, _technologies)
    assert (result >= 0).all()

    # Test with demand growth and insufficient capacity
    reduced_capacity = _capacity.copy()
    reduced_capacity.loc[{"year": INVESTMENT_YEAR}] = 0.5 * _capacity.sel(
        year=CURRENT_YEAR
    )
    result = new_consumption(reduced_capacity, _market.consumption, _technologies)
    assert (result >= 0).all()


def test_standard_demand_share(_technologies, stock):
    """Test standard_demand with various scenarios."""
    from muse.demand_share import standard_demand

    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)
    agents = create_test_agents(usa_stock, categories=["newcapa"])
    usa_demand = market.consumption.sel(region=["USA"])

    # Basic functionality -> result should be non-negative
    result = standard_demand(agents, usa_demand, _technologies)
    assert (result >= 0).all()

    # Demand growth scenario -> result should be non-negative
    market.consumption.loc[{"year": INVESTMENT_YEAR}] = 1.5 * market.consumption.sel(
        year=CURRENT_YEAR
    )
    result_growth = standard_demand(agents, market.consumption, _technologies)
    assert (result_growth >= 0).all()

    # No demand growth scenario -> result should be smaller or equal to demand growth
    market.consumption.loc[{"year": INVESTMENT_YEAR}] = market.consumption.sel(
        year=CURRENT_YEAR
    )
    result_no_growth = standard_demand(agents, market.consumption, _technologies)
    assert (result_no_growth <= result_growth).all()

    # Capacity reduction scenario -> result should be non-negative
    for agent in agents:
        agent.assets = agent.assets.copy()
        agent.assets["capacity"].loc[{"year": INVESTMENT_YEAR}] = 0.5 * agent.assets[
            "capacity"
        ].sel(year=CURRENT_YEAR)
    result_capacity_reduction = standard_demand(
        agents, market.consumption, _technologies
    )
    assert (result_capacity_reduction >= 0).all()

    # Regional split scenario -> result should be non-negative
    multi_region_agents = create_test_agents(
        usa_stock, asia_stock=asia_stock, categories=["newcapa"]
    )
    result_regional = standard_demand(
        multi_region_agents, market.consumption, _technologies
    )
    assert (result_regional >= 0).all()

    # Zero demand scenario -> result should be zero
    zero_demand = market.consumption.copy()
    zero_demand.loc[:] = 0
    result_zero = standard_demand(agents, zero_demand, _technologies)
    assert (result_zero == 0).all()


def test_new_and_retro_demand_share(_technologies, stock):
    """Test new_and_retro with various scenarios."""
    from muse.demand_share import new_and_retro

    market, asia_stock, usa_stock = create_regional_market(_technologies, stock)
    agents = create_test_agents(usa_stock, categories=["retrofit", "newcapa"])

    # Basic functionality -> result should be non-negative
    result = new_and_retro(agents, market.consumption, _technologies)
    assert (result >= 0).all()

    # Demand growth scenario -> result should be non-negative
    market.consumption.loc[{"year": INVESTMENT_YEAR}] = 1.5 * market.consumption.sel(
        year=CURRENT_YEAR
    )
    result_growth = new_and_retro(agents, market.consumption, _technologies)
    assert (result_growth >= 0).all()

    # No demand growth scenario -> result should be smaller or equal to demand growth
    market.consumption.loc[{"year": INVESTMENT_YEAR}] = market.consumption.sel(
        year=CURRENT_YEAR
    )
    result_no_growth = new_and_retro(agents, market.consumption, _technologies)
    assert (result_no_growth <= result_growth).all()

    # Capacity reduction scenario -> result should be non-negative
    for agent in agents:
        agent.assets = agent.assets.copy()
        agent.assets["capacity"].loc[{"year": INVESTMENT_YEAR}] = 0.5 * agent.assets[
            "capacity"
        ].sel(year=CURRENT_YEAR)
    result_capacity_reduction = new_and_retro(agents, market.consumption, _technologies)
    assert (result_capacity_reduction >= 0).all()

    # Regional split scenario -> result should be non-negative
    multi_region_agents = create_test_agents(
        usa_stock, asia_stock=asia_stock, categories=["retrofit", "newcapa"]
    )
    result_regional = new_and_retro(
        multi_region_agents, market.consumption, _technologies
    )
    assert (result_regional >= 0).all()

    # Zero demand scenario -> result should be zero
    zero_demand = market.consumption.copy()
    zero_demand.loc[:] = 0
    result_zero = new_and_retro(agents, zero_demand, _technologies)
    assert (result_zero == 0).all()
