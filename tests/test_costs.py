from numpy import isclose
from pytest import fixture, mark, raises

YEAR = 2030


@fixture
def _capacity(technologies, demand_share):
    """Capacity for a set of assets."""
    from muse.quantities import capacity_to_service_demand
    from muse.utilities import broadcast_techs

    techs = broadcast_techs(technologies.sel(year=YEAR), demand_share)
    capacity = capacity_to_service_demand(technologies=techs, demand=demand_share)
    return capacity


@fixture
def _technologies(technologies, _capacity):
    """Technology parameters for each asset."""
    from muse.utilities import broadcast_techs

    return broadcast_techs(technologies.sel(year=YEAR), _capacity)


@fixture
def _production(_technologies, _capacity):
    """Production data for each asset."""
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    production = (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_outputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )
    return production


@fixture
def _consumption(_technologies, _capacity):
    """Consumption data for each asset."""
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    consumption = (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_inputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )
    return consumption


@fixture
def _prices(market, _capacity):
    """Prices relevant to each asset."""
    from muse.utilities import broadcast_techs

    prices = market.prices.sel(year=YEAR)
    return broadcast_techs(prices, _capacity)


def test_fixtures(_technologies, _prices, _capacity, _production, _consumption):
    """Validating that the fixtures have appropriate dimensions."""
    assert set(_technologies.dims) == {"asset", "commodity"}
    assert set(_prices.dims) == {"asset", "commodity", "timeslice"}
    assert set(_capacity.dims) == {"asset"}
    assert (
        set(_production.dims)
        == set(_consumption.dims)
        == {
            "asset",
            "commodity",
            "timeslice",
        }
    )


def test_capital_costs(_technologies, _capacity):
    from muse.costs import capital_costs

    result = capital_costs(_technologies, _capacity)
    assert set(result.dims) == {"asset"}


def test_environmental_costs(_technologies, _prices, _production):
    from muse.costs import environmental_costs

    result = environmental_costs(_technologies, _prices, _production)
    assert set(result.dims) == {"asset", "timeslice"}


def test_fuel_costs(_technologies, _prices, _consumption):
    from muse.costs import fuel_costs

    result = fuel_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_material_costs(_technologies, _prices, _consumption):
    from muse.costs import material_costs

    result = material_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_fixed_costs(_technologies, _capacity):
    from muse.costs import fixed_costs

    result = fixed_costs(_technologies, _capacity)
    assert set(result.dims) == {"asset"}


def test_variable_costs(_technologies, _production):
    from muse.costs import variable_costs

    result = variable_costs(_technologies, _production)
    assert set(result.dims) == {"asset"}


def test_running_costs(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import running_costs

    result = running_costs(_technologies, _prices, _capacity, _production, _consumption)
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_value(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import net_present_value

    result = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_net_present_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import net_present_cost

    result = net_present_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_equivalent_annual_cost(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import equivalent_annual_cost

    result = equivalent_annual_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "timeslice"}


@mark.parametrize("method", ["annual", "lifetime"])
def test_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    from muse.costs import levelized_cost_of_energy

    result = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert set(result.dims) == {"asset", "timeslice"}


def test_supply_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import levelized_cost_of_energy, supply_cost

    lcoe = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method="annual"
    )
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {
        "commodity",
        "region",
        "timeslice",
    }


def test_capital_recovery_factor(_technologies):
    from muse.costs import capital_recovery_factor

    result = capital_recovery_factor(_technologies)
    assert set(result.dims) == set(_technologies.interest_rate.dims)
    # {"region", "technology"}


def test_annual_to_lifetime(_technologies, _prices, _consumption):
    from muse.costs import annual_to_lifetime, fuel_costs

    _fuel_costs = fuel_costs(_technologies, _prices, _consumption)
    _fuel_costs_lifetime = annual_to_lifetime(_fuel_costs, _technologies)
    assert set(_fuel_costs.dims) == set(_fuel_costs_lifetime.dims)
    assert (_fuel_costs_lifetime > _fuel_costs).all()


@mark.parametrize("method", ["annual", "lifetime"])
def test_lcoe_flow_scaling(
    _technologies, _prices, _capacity, _production, _consumption, method
):
    """Testing that LCOE is independent of input/output flow scaling.

    In other words, if we change technology flows by a constant factor, the LCOE (which
    is a cost per unit of production) should remain unchanged.

    This is a bit more complicated if the variable costs are nonlinear, so we'll set
    the exponent to 1 for simplicity.
    """
    from muse.costs import levelized_cost_of_energy

    _technologies["var_exp"] = 1

    # LCOE with original inputs
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )

    # Scale inputs and outputs by a constant factor -> LCOE should be unchanged
    # var_par also needs to be scaled as this relates to units of technology
    # activity, not units of commodity consumption/production
    _technologies_scaled = _technologies.copy()
    _technologies_scaled["fixed_inputs"] = _technologies["fixed_inputs"] * 2
    _technologies_scaled["flexible_inpits"] = _technologies["flexible_inputs"] * 2
    _technologies_scaled["fixed_outputs"] = _technologies["fixed_outputs"] * 2
    _technologies_scaled["var_par"] = _technologies["var_par"] * 2
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
    """Testing that LCOE is independent of production scaling.

    If all costs are linear (exponents = 1), then the LCOE should be independent of
    production as long as production, consumption, and capacity are scaled together.
    """
    from muse.costs import levelized_cost_of_energy

    _technologies["var_exp"] = 1
    _technologies["cap_exp"] = 1
    _technologies["fix_exp"] = 1

    # LCOE with original inputs
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )

    # Scale consumption, production, and capacity by a constant factor -> LCOE
    # should be unchanged
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
    """If commodity prices are equal in every timeslice, LCOE should always be equal."""
    from xarray.testing import assert_allclose

    from muse.costs import levelized_cost_of_energy
    from muse.timeslices import broadcast_timeslice

    # LCOE with original inputs -> should vary between timeslices
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    with raises(AssertionError):
        assert_allclose(lcoe1, broadcast_timeslice(lcoe1.isel(timeslice=0)))

    # LCOE with uniform prices -> should be the same for all timeslices
    _prices = broadcast_timeslice(_prices.mean("timeslice"))
    lcoe2 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert_allclose(lcoe2, broadcast_timeslice(lcoe2.isel(timeslice=0)))


def test_npv_equal_prices(_technologies, _prices, _capacity, _production, _consumption):
    """Test NPV with equal commodity prices in every timeslice.

    If commodity prices are equal in every timeslice, NPV should be proportional to
    production.
    """
    from xarray.testing import assert_allclose

    from muse.costs import net_present_value
    from muse.quantities import production_amplitude
    from muse.timeslices import broadcast_timeslice

    # NPV with original inputs -> should not be linear with production
    npv1 = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    tech_activity = production_amplitude(_production, _technologies)
    npv1_scaled = npv1 / tech_activity
    with raises(AssertionError):
        assert_allclose(npv1_scaled, broadcast_timeslice(npv1_scaled.isel(timeslice=0)))

    # NPV with uniform prices -> should be linear with production
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
    """If production and consumption are zero, LCOE should always be zero.

    Note: if production/consumption are zero in every timeslice, LCOE is undefined (nan)
    """
    from muse.costs import levelized_cost_of_energy

    # LCOE with original inputs
    lcoe1 = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method=method
    )
    assert not (lcoe1.isel(timeslice=0) == 0).all()

    # LCOE with zero production/consumption in first timeslice -> LCOE should be zero
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
    from muse.costs import levelized_cost_of_energy

    result = levelized_cost_of_energy(
        _technologies,
        _prices,
        _capacity,
        _production,
        _consumption,
        method=method,
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}  # no timeslice dim


def test_npv_aggregate(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import net_present_value

    result = net_present_value(
        _technologies,
        _prices,
        _capacity,
        _production,
        _consumption,
        aggregate_timeslices=True,
    )
    assert set(result.dims) == {"asset"}  # no timeslice dim
