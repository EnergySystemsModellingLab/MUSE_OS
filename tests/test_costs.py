from pytest import fixture

YEAR = 2030


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
def _production(_technologies, _capacity, demand_share):
    from muse.timeslices import QuantityType, convert_timeslice

    production = (
        _capacity
        * convert_timeslice(
            _technologies.fixed_outputs, demand_share.timeslice, QuantityType.EXTENSIVE
        )
        * _technologies.utilization_factor
    )
    return production


@fixture
def _consumption(_technologies, _capacity, demand_share):
    from muse.timeslices import QuantityType, convert_timeslice

    consumption = (
        _capacity
        * convert_timeslice(
            _technologies.fixed_inputs, demand_share.timeslice, QuantityType.EXTENSIVE
        )
        * _technologies.utilization_factor
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


def test_net_present_value(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import net_present_value

    result = net_present_value(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice"}


def test_net_present_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import net_present_cost

    result = net_present_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice"}


def test_equivalent_annual_cost(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import equivalent_annual_cost

    result = equivalent_annual_cost(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice"}


def test_lifetime_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice"}


def test_annual_levelized_cost_of_energy(
    _technologies, _prices, _capacity, _production, _consumption
):
    from muse.costs import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice"}


def test_supply_cost(_technologies, _prices, _capacity, _production, _consumption):
    from muse.costs import annual_levelized_cost_of_energy, supply_cost

    lcoe = annual_levelized_cost_of_energy(
        _technologies, _prices, _capacity, _production, _consumption
    )
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {
        "commodity",
        "region",
        "technology",
        "timeslice",
    }


def test_capital_recovery_factor(technologies):
    from muse.costs import capital_recovery_factor

    result = capital_recovery_factor(technologies)
    assert set(result.dims) == {"region", "technology", "year"}
