from numpy import isclose, isfinite
from pytest import fixture, mark, raises
from xarray.testing import assert_allclose

from muse.costs import (
    annual_to_lifetime,
    capital_costs,
    capital_recovery_factor,
    environmental_costs,
    equivalent_annual_cost,
    fixed_costs,
    fuel_costs,
    levelized_cost_of_energy,
    material_costs,
    net_present_cost,
    net_present_value,
    running_costs,
    supply_cost,
    variable_costs,
)
from muse.quantities import capacity_to_service_demand, production_amplitude
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import broadcast_over_assets

YEAR = 2030


@fixture
def cost_inputs(technologies, market, demand_share):
    """Creates the complete dataset needed for cost calculations.

    The transformation follows these steps:
    1. Extract year-specific data from technologies and market
    2. Transform data to asset level
    3. Calculate capacity for each asset
    4. Calculate production and consumption data

    Returns:
        dict: Contains all necessary data for cost calculations:
            - technologies: Technology parameters for each asset
            - prices: Prices relevant to each asset
            - capacity: Capacity for each asset
            - production: Production data for each asset
            - consumption: Consumption data for each asset
    """
    # Step 1: Extract year-specific data
    tech_year = technologies.sel(year=YEAR)
    prices_year = market.prices.sel(year=YEAR)

    # Step 2: Transform to asset level
    tech_assets = broadcast_over_assets(tech_year, demand_share)
    prices_assets = broadcast_over_assets(
        prices_year, demand_share, installed_as_year=False
    )

    # Step 3: Calculate capacity
    capacity = capacity_to_service_demand(technologies=tech_assets, demand=demand_share)

    # Step 4: Calculate production and consumption
    production = (
        broadcast_timeslice(capacity)
        * distribute_timeslice(tech_assets.fixed_outputs)
        * broadcast_timeslice(tech_assets.utilization_factor)
    )

    consumption = (
        broadcast_timeslice(capacity)
        * distribute_timeslice(tech_assets.fixed_inputs)
        * broadcast_timeslice(tech_assets.utilization_factor)
    )

    return {
        "technologies": tech_assets,
        "prices": prices_assets,
        "capacity": capacity,
        "production": production,
        "consumption": consumption,
    }


def test_fixtures(cost_data):
    """Validate fixture dimensions."""
    assert set(cost_data["technologies"].dims) == {"asset", "commodity"}
    assert set(cost_data["prices"].dims) == {"asset", "commodity", "timeslice"}
    assert set(cost_data["capacity"].dims) == {"asset"}
    assert set(cost_data["production"].dims) == {"asset", "commodity", "timeslice"}
    assert set(cost_data["consumption"].dims) == {"asset", "commodity", "timeslice"}


def test_capital_costs(cost_data):
    result = capital_costs(cost_data["technologies"], cost_data["capacity"])
    assert set(result.dims) == {"asset"}


def test_environmental_costs(cost_data):
    result = environmental_costs(
        cost_data["technologies"], cost_data["prices"], cost_data["production"]
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_fuel_costs(cost_data):
    result = fuel_costs(
        cost_data["technologies"], cost_data["prices"], cost_data["consumption"]
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_material_costs(cost_data):
    result = material_costs(
        cost_data["technologies"], cost_data["prices"], cost_data["consumption"]
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_fixed_costs(cost_data):
    result = fixed_costs(cost_data["technologies"], cost_data["capacity"])
    assert set(result.dims) == {"asset"}


def test_variable_costs(cost_data):
    result = variable_costs(cost_data["technologies"], cost_data["production"])
    assert set(result.dims) == {"asset"}


def test_running_costs(cost_data):
    result = running_costs(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_value(cost_data):
    result = net_present_value(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_cost(cost_data):
    result = net_present_cost(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_equivalent_annual_cost(cost_data):
    result = equivalent_annual_cost(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    assert set(result.dims) == {"asset", "timeslice"}


@mark.parametrize("method", ["annual", "lifetime"])
def test_levelized_cost_of_energy(cost_data, method):
    result = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_supply_cost(cost_data):
    lcoe = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method="annual",
    )
    result = supply_cost(cost_data["production"], lcoe)
    assert set(result.dims) == {"commodity", "region", "timeslice"}


def test_capital_recovery_factor(cost_data):
    result = capital_recovery_factor(cost_data["technologies"])
    assert set(result.dims) == set(cost_data["technologies"].interest_rate.dims)

    # Test zero interest rates
    cost_data["technologies"]["interest_rate"] = 0
    result = capital_recovery_factor(cost_data["technologies"])
    assert isfinite(result).all()


def test_annual_to_lifetime(cost_data):
    _fuel_costs = fuel_costs(
        cost_data["technologies"], cost_data["prices"], cost_data["consumption"]
    )
    _fuel_costs_lifetime = annual_to_lifetime(_fuel_costs, cost_data["technologies"])
    assert set(_fuel_costs.dims) == set(_fuel_costs_lifetime.dims)
    assert (_fuel_costs_lifetime > _fuel_costs).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_flow_scaling(cost_data, method):
    """Test LCOE independence of input/output flow scaling."""
    cost_data["technologies"]["var_exp"] = 1

    # Original LCOE
    lcoe1 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )

    # Scale inputs/outputs and var_par by 2
    technologies_scaled = cost_data["technologies"].copy()
    technologies_scaled["fixed_inputs"] *= 2
    technologies_scaled["flexible_inputs"] *= 2
    technologies_scaled["fixed_outputs"] *= 2
    technologies_scaled["var_par"] *= 2

    lcoe2 = levelized_cost_of_energy(
        technologies_scaled,
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    assert isclose(lcoe1, lcoe2).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_prod_scaling(cost_data, method):
    """Test LCOE independence of production scaling with linear costs."""
    cost_data["technologies"]["var_exp"] = 1
    cost_data["technologies"]["cap_exp"] = 1
    cost_data["technologies"]["fix_exp"] = 1

    lcoe1 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    lcoe2 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"] * 2,
        cost_data["production"] * 2,
        cost_data["consumption"] * 2,
        method=method,
    )
    assert isclose(lcoe1, lcoe2).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_equal_prices(cost_data, method):
    """Test LCOE behavior with uniform prices across timeslices."""
    lcoe1 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    with raises(AssertionError):
        assert_allclose(lcoe1, broadcast_timeslice(lcoe1.isel(timeslice=0)))

    # Test with uniform prices
    prices_uniform = broadcast_timeslice(cost_data["prices"].mean("timeslice"))
    lcoe2 = levelized_cost_of_energy(
        cost_data["technologies"],
        prices_uniform,
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    assert_allclose(lcoe2, broadcast_timeslice(lcoe2.isel(timeslice=0)))


def test_npv_equal_prices(cost_data):
    """Test NPV linearity with production under uniform prices."""
    npv1 = net_present_value(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    tech_activity = production_amplitude(
        cost_data["production"], cost_data["technologies"]
    )
    npv1_scaled = npv1 / tech_activity

    with raises(AssertionError):
        assert_allclose(npv1_scaled, broadcast_timeslice(npv1_scaled.isel(timeslice=0)))

    # Test with uniform prices
    prices_uniform = broadcast_timeslice(cost_data["prices"].mean("timeslice"))
    npv2 = net_present_value(
        cost_data["technologies"],
        prices_uniform,
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
    )
    npv2_scaled = npv2 / tech_activity
    assert_allclose(npv2_scaled, broadcast_timeslice(npv2_scaled.isel(timeslice=0)))


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_zero_production(cost_data, method):
    """Test LCOE behavior with zero production."""
    lcoe1 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
    )
    assert not (lcoe1.isel(timeslice=0) == 0).all()

    # Test with zero production in first timeslice
    production_zero = cost_data["production"].copy()
    consumption_zero = cost_data["consumption"].copy()
    production_zero.isel(timeslice=0)[:] = 0
    consumption_zero.isel(timeslice=0)[:] = 0

    lcoe2 = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        production_zero,
        consumption_zero,
        method=method,
    )
    assert (lcoe2.isel(timeslice=0) == 0).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_aggregate(cost_data, method):
    """Test LCOE aggregation over timeslices."""
    result = levelized_cost_of_energy(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        method=method,
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}


def test_npv_aggregate(cost_data):
    """Test NPV aggregation over timeslices."""
    result = net_present_value(
        cost_data["technologies"],
        cost_data["prices"],
        cost_data["capacity"],
        cost_data["production"],
        cost_data["consumption"],
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}
