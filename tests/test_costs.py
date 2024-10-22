from pytest import fixture


@fixture
def _prices(market):
    prices = market.prices
    return prices


@fixture
def _capacity(technologies, demand_share):
    from muse.quantities import capacity_to_service_demand

    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand_share
    )
    return capacity


@fixture
def _production(technologies, _capacity, demand_share):
    from muse.timeslices import QuantityType, convert_timeslice

    production = (
        _capacity * technologies.fixed_outputs * technologies.utilization_factor
    )
    production = convert_timeslice(
        production, demand_share.timeslice, QuantityType.EXTENSIVE
    )
    return production


def test_fixtures(technologies, _prices, _capacity, _production):
    """Validating that the fixtures have appropriate dimensions."""
    assert set(technologies.dims) == {"commodity", "region", "technology", "year"}
    assert set(_prices.dims) == {"commodity", "region", "timeslice", "year"}
    assert set(_capacity.dims) == {"asset", "region", "technology", "year"}
    assert set(_production.dims) == {
        "asset",
        "commodity",
        "region",
        "technology",
        "timeslice",
        "year",
    }


def test_net_present_value(technologies, _prices, _capacity, _production, year=2030):
    from muse.costs import net_present_value

    result = net_present_value(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "region", "technology", "timeslice", "year"}


def test_net_present_cost(technologies, _prices, _capacity, _production, year=2030):
    from muse.costs import net_present_cost

    result = net_present_cost(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "region", "technology", "timeslice", "year"}


def test_equivalent_annual_cost(
    technologies, _prices, _capacity, _production, year=2030
):
    from muse.costs import equivalent_annual_cost

    result = equivalent_annual_cost(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "region", "technology", "timeslice", "year"}


def test_lifetime_levelized_cost_of_energy(
    technologies, _prices, _capacity, _production, year=2030
):
    from muse.costs import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(
        technologies, _prices, _capacity, _production, year
    )
    assert set(result.dims) == {"asset", "region", "technology", "timeslice", "year"}


def test_annual_levelized_cost_of_energy(technologies, _prices):
    from muse.costs import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(technologies, _prices)
    assert set(result.dims) == {"region", "technology", "timeslice", "year"}


def test_supply_cost(_production, _prices, technologies):
    from muse.costs import annual_levelized_cost_of_energy, supply_cost

    lcoe = annual_levelized_cost_of_energy(technologies, _prices)
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {
        "commodity",
        "region",
        "technology",
        "timeslice",
        "year",
    }


def test_capital_recovery_factor(technologies):
    from muse.costs import capital_recovery_factor

    result = capital_recovery_factor(technologies)
    assert set(result.dims) == {"region", "technology", "year"}
