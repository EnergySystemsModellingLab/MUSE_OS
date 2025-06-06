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
from muse.quantities import production_amplitude
from muse.timeslices import broadcast_timeslice

YEAR = 2030


@fixture
def _capacity(_technologies, demand_share):
    """Capacity for each asset."""
    from muse.quantities import capacity_to_service_demand

    return capacity_to_service_demand(technologies=_technologies, demand=demand_share)


@fixture
def _technologies(technologies, demand_share):
    """Technology parameters for each asset."""
    from muse.utilities import broadcast_over_assets

    return broadcast_over_assets(technologies.sel(year=YEAR), demand_share)


@fixture
def _prices(market, demand_share):
    """Prices relevant to each asset."""
    from muse.utilities import broadcast_over_assets

    prices = market.prices.sel(year=YEAR)
    return broadcast_over_assets(prices, demand_share, installed_as_year=False)


@fixture
def _production(_technologies, _capacity):
    """Production data for each asset."""
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    return (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_outputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )


@fixture
def _consumption(_technologies, _capacity):
    """Consumption data for each asset."""
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    return (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_inputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )


def test_fixtures(_technologies, _prices, _capacity, _production, _consumption):
    """Validate fixture dimensions."""
    assert set(_technologies.dims) == {"asset", "commodity"}
    assert set(_prices.dims) == {"asset", "commodity", "timeslice"}
    assert set(_capacity.dims) == {"asset"}
    assert set(_production.dims) == {"asset", "commodity", "timeslice"}
    assert set(_consumption.dims) == {"asset", "commodity", "timeslice"}


def test_capital_costs(_technologies, _capacity):
    result = capital_costs(_technologies, _capacity)
    assert set(result.dims) == {"asset"}


def test_environmental_costs(_technologies, _prices, _production):
    result = environmental_costs(_technologies, _prices, _production)
    assert set(result.dims) == {"asset", "timeslice"}


def test_fuel_costs(_technologies, _prices, _consumption):
    result = fuel_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_material_costs(_technologies, _prices, _consumption):
    result = material_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_fixed_costs(_technologies, _capacity):
    result = fixed_costs(_technologies, _capacity)
    assert set(result.dims) == {"asset"}


def test_variable_costs(_technologies, _production):
    result = variable_costs(_technologies, _production)
    assert set(result.dims) == {"asset"}


def test_running_costs(_technologies, _prices, _capacity, _production, _consumption):
    result = running_costs(_technologies, _prices, _capacity, _production, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_value(
    _technologies, _prices, _capacity, _production, _consumption
):
    result = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_cost(_technologies, _prices, _capacity, _production, _consumption):
    result = net_present_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_equivalent_annual_cost(
    _technologies, _prices, _capacity, _production, _consumption
):
    result = equivalent_annual_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


@mark.parametrize("method", ["annual", "lifetime"])
def test_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    result = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_supply_cost(_technologies, _prices, _capacity, _production, _consumption):
    lcoe = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method="annual"
    )
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {"commodity", "region", "timeslice"}


def test_capital_recovery_factor(_technologies):
    result = capital_recovery_factor(_technologies)
    assert set(result.dims) == set(_technologies.interest_rate.dims)

    # Test zero interest rates
    _technologies["interest_rate"] = 0
    result = capital_recovery_factor(_technologies)
    assert isfinite(result).all()


def test_annual_to_lifetime(_technologies, _prices, _consumption):
    _fuel_costs = fuel_costs(_technologies, _prices, _consumption)
    _fuel_costs_lifetime = annual_to_lifetime(_fuel_costs, _technologies)
    assert set(_fuel_costs.dims) == set(_fuel_costs_lifetime.dims)
    assert (_fuel_costs_lifetime > _fuel_costs).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_flow_scaling(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Test LCOE independence of input/output flow scaling."""
    _technologies["var_exp"] = 1

    # Original LCOE
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )

    # Scale inputs/outputs and var_par by 2
    _technologies_scaled = _technologies.copy()
    _technologies_scaled["fixed_inputs"] *= 2
    _technologies_scaled["flexible_inputs"] *= 2
    _technologies_scaled["fixed_outputs"] *= 2
    _technologies_scaled["var_par"] *= 2

    lcoe2 = levelized_cost_of_energy(
        _technologies_scaled,
        _prices,
        _capacity,
        _production,
        _consumption,
        method=method,
    )
    assert isclose(lcoe1, lcoe2).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_prod_scaling(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Test LCOE independence of production scaling with linear costs."""
    _technologies["var_exp"] = 1
    _technologies["cap_exp"] = 1
    _technologies["fix_exp"] = 1

    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    lcoe2 = levelized_cost_of_energy(
        _technologies,
        _prices,
        _capacity * 2,
        _production * 2,
        _consumption * 2,
        method=method,
    )
    assert isclose(lcoe1, lcoe2).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_equal_prices(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Test LCOE behavior with uniform prices across timeslices."""
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    with raises(AssertionError):
        assert_allclose(lcoe1, broadcast_timeslice(lcoe1.isel(timeslice=0)))

    # Test with uniform prices
    _prices = broadcast_timeslice(_prices.mean("timeslice"))
    lcoe2 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert_allclose(lcoe2, broadcast_timeslice(lcoe2.isel(timeslice=0)))


def test_npv_equal_prices(_technologies, _prices, _capacity, _production, _consumption):
    """Test NPV linearity with production under uniform prices."""
    npv1 = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    tech_activity = production_amplitude(_production, _technologies)
    npv1_scaled = npv1 / tech_activity

    with raises(AssertionError):
        assert_allclose(npv1_scaled, broadcast_timeslice(npv1_scaled.isel(timeslice=0)))

    # Test with uniform prices
    _prices = broadcast_timeslice(_prices.mean("timeslice"))
    npv2 = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    npv2_scaled = npv2 / tech_activity
    assert_allclose(npv2_scaled, broadcast_timeslice(npv2_scaled.isel(timeslice=0)))


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_zero_production(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Test LCOE behavior with zero production."""
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert not (lcoe1.isel(timeslice=0) == 0).all()

    # Test with zero production in first timeslice
    _production.isel(timeslice=0)[:] = 0
    _consumption.isel(timeslice=0)[:] = 0
    lcoe2 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert (lcoe2.isel(timeslice=0) == 0).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_aggregate(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Test LCOE aggregation over timeslices."""
    result = levelized_cost_of_energy(
        _technologies,
        _prices,
        _capacity,
        _production,
        _consumption,
        method=method,
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}


def test_npv_aggregate(_technologies, _prices, _capacity, _production, _consumption):
    """Test NPV aggregation over timeslices."""
    result = net_present_value(
        _technologies,
        _prices,
        _capacity,
        _production,
        _consumption,
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}
