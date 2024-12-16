from pytest import fixture

YEAR = 2030

"""Expected dimensions for the output data of the cost functions.

This should be the same for all cost functions. In general, this is the sum of all
dimensions from the input data, minus "commodity", plus "timeslice" (if not already
present).
"""
EXPECTED_DIMS = {"asset", "region", "technology", "timeslice"}


@fixture
def _prices(market):
    prices = market.prices
    return prices.sel(year=YEAR)


@fixture
def _technologies(technologies):
    return technologies.sel(year=YEAR)


@fixture
def _capacity(_technologies, demand_share):
    from muse.quantities import capacity_to_service_demand

    capacity = capacity_to_service_demand(
        technologies=_technologies, demand=demand_share
    )
    return capacity


@fixture
def _production(_technologies, _capacity):
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    production = (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_outputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )
    return production


@fixture
def _consumption(_technologies, _capacity):
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    consumption = (
        broadcast_timeslice(_capacity)
        * distribute_timeslice(_technologies.fixed_inputs)
        * broadcast_timeslice(_technologies.utilization_factor)
    )
    return consumption


def test_fixtures(_technologies, _prices, _capacity, _production, _consumption):
    """Validating that the fixtures have appropriate dimensions."""
    assert set(_technologies.dims) == {"commodity", "region", "technology"}
    assert set(_prices.dims) == {"commodity", "region", "timeslice"}
    assert set(_capacity.dims) == {"asset", "region", "technology"}
    assert (
        set(_production.dims)
        == set(_consumption.dims)
        == {
            "asset",
            "commodity",
            "region",
            "technology",
            "timeslice",
        }
    )


def test_capital_costs(_technologies, _capacity, _production):
    from muse.costs import capital_costs

    result = capital_costs(_technologies, _capacity, _production)
    assert set(result.dims) == EXPECTED_DIMS


def test_environmental_costs(_technologies, _prices, _production):
    from muse.costs import environmental_costs

    result = environmental_costs(_technologies, _prices, _production)
    assert set(result.dims) == EXPECTED_DIMS


def test_fuel_costs(_technologies, _prices, _consumption):
    from muse.costs import fuel_costs

    result = fuel_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == EXPECTED_DIMS


def test_material_costs(_technologies, _prices, _consumption):
    from muse.costs import material_costs

    result = material_costs(_technologies, _prices, _consumption)
    assert set(result.dims) == EXPECTED_DIMS


def test_fixed_costs(_technologies, _capacity, _production):
    from muse.costs import fixed_costs

    result = fixed_costs(_technologies, _capacity, _production)
    assert set(result.dims) == EXPECTED_DIMS


def test_variable_costs(_technologies, _production):
    from muse.costs import variable_costs

    result = variable_costs(_technologies, _production)
    assert set(result.dims) == EXPECTED_DIMS


def test_running_costs(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import running_costs

    result = running_costs(_technologies, _prices, _capacity, _production, _consumption)
    assert set(result.dims) == EXPECTED_DIMS


def test_net_present_value(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import net_present_value

    result = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == EXPECTED_DIMS


def test_net_present_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import net_present_cost

    result = net_present_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == EXPECTED_DIMS


def test_equivalent_annual_cost(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import equivalent_annual_cost

    result = equivalent_annual_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == EXPECTED_DIMS


def test_lifetime_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import levelized_cost_of_energy

    result = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method="lifetime"
    )
    assert set(result.dims) == EXPECTED_DIMS


def test_annual_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import levelized_cost_of_energy

    result = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method="annual"
    )
    assert set(result.dims) == EXPECTED_DIMS


def test_supply_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import levelized_cost_of_energy, supply_cost

    lcoe = levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption, method="annual"
    )
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {
        "commodity",
        "region",
        "technology",
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
