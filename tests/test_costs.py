from pytest import fixture


@fixture
def _prices(market):
    prices = market.prices
    assert set(prices.dims) == {"commodity", "region", "year", "timeslice"}
    return prices


@fixture
def _capacity(technologies, demand_share):
    from muse.quantities import capacity_to_service_demand

    assert set(technologies.dims) == {"region", "year", "technology", "commodity"}
    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand_share
    )
    assert set(capacity.dims) == {"asset", "region", "year", "technology"}
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
    assert set(production.dims) == {
        "asset",
        "timeslice",
        "commodity",
        "region",
        "year",
        "technology",
    }
    return production


def test_net_present_value(technologies, _prices, _capacity, _production, year=2030):
    from muse.costs import net_present_value

    result = net_present_value(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "timeslice", "region", "year", "technology"}


def test_net_present_cost(technologies, _prices, _capacity, _production, year=2030):
    from muse.costs import net_present_cost

    result = net_present_cost(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "timeslice", "region", "year", "technology"}


def test_equivalent_annual_cost(
    technologies, _prices, _capacity, _production, year=2030
):
    from muse.costs import equivalent_annual_cost

    result = equivalent_annual_cost(technologies, _prices, _capacity, _production, year)
    assert set(result.dims) == {"asset", "timeslice", "region", "year", "technology"}


def test_lifetime_levelized_cost_of_energy(
    technologies, _prices, _capacity, _production, year=2030
):
    from muse.costs import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(
        technologies, _prices, _capacity, _production, year
    )
    assert set(result.dims) == {"asset", "timeslice", "region", "year", "technology"}


def test_annual_levelized_cost_of_energy(technologies, _prices):
    from muse.costs import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(technologies, _prices)
    assert set(result.dims) == {"timeslice", "region", "year", "technology"}


def test_supply_cost(_production, _prices, technologies):
    from muse.costs import annual_levelized_cost_of_energy, supply_cost

    lcoe = annual_levelized_cost_of_energy(technologies, _prices)
    result = supply_cost(_production, lcoe)
    assert set(result.dims) == {
        "timeslice",
        "region",
        "year",
        "technology",
        "commodity",
    }
