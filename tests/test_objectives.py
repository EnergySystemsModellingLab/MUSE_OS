from pytest import fixture, mark

YEAR = 2030


@fixture
def _demand(demand_share):
    return demand_share


@fixture
def _technologies(technologies, demand_share):
    from muse.utilities import broadcast_over_assets

    techs = technologies.sel(year=YEAR).rename(technology="replacement")
    return broadcast_over_assets(techs, demand_share)


@fixture
def _prices(market, demand_share):
    from muse.utilities import broadcast_over_assets

    prices = market.prices.sel(year=YEAR)
    return broadcast_over_assets(prices, demand_share, installed_as_year=False)


def test_fixtures(_technologies, _demand, _prices):
    """Validating that the fixtures have appropriate dimensions."""
    assert set(_technologies.dims) == {"asset", "commodity", "replacement"}
    assert set(_demand.dims) == {"asset", "commodity", "timeslice"}
    assert set(_prices.dims) == {"asset", "commodity", "timeslice"}


@mark.usefixtures("save_registries")
def test_objective_registration():
    from muse.objectives import OBJECTIVES, register_objective

    @register_objective
    def a_objective(*args, **kwargs):
        pass

    assert "a_objective" in OBJECTIVES
    assert OBJECTIVES["a_objective"] is a_objective

    @register_objective(name="something")
    def b_objective(*args, **kwargs):
        pass

    assert "something" in OBJECTIVES
    assert OBJECTIVES["something"] is b_objective


@mark.usefixtures("save_registries")
def test_computing_objectives(_technologies, _demand, _prices):
    from muse.objectives import factory, register_objective

    @register_objective
    def first(technologies, demand, switch=True, *args, **kwargs):
        from xarray import broadcast, full_like

        value = 1 if switch else 2
        result = full_like(
            broadcast(technologies["replacement"], demand["asset"])[0],
            value,
            dtype=float,
        )
        return result

    @register_objective
    def second(technologies, demand, assets=None, *args, **kwargs):
        from xarray import broadcast, full_like

        result = full_like(
            broadcast(technologies["replacement"], demand["asset"])[0], 5, dtype=float
        )
        result[{"asset": assets}] = 3
        return result

    # Test first objective with/without switch
    objectives = factory("first")(
        technologies=_technologies, demand=_demand, prices=_prices, switch=True
    )
    assert set(objectives.data_vars) == {"first"}
    assert (objectives.first == 1).all()
    objectives = factory("first")(
        technologies=_technologies, demand=_demand, prices=_prices, switch=False
    )
    assert (objectives.first == 2).all()

    # Test multiple objectives
    objectives = factory(["first", "second"])(
        technologies=_technologies,
        demand=_demand,
        prices=_prices,
        switch=False,
        assets=0,
    )
    assert set(objectives.data_vars) == {"first", "second"}
    assert (objectives.first == 2).all()
    if len(objectives.asset) > 0:
        assert (objectives.second.isel(asset=0) == 3).all()
    if len(objectives.asset) > 1:
        assert (objectives.second.isel(asset=1) == 5).all()


def test_comfort(_technologies, _demand):
    from muse.objectives import comfort

    _technologies["comfort"] = add_var(_technologies, "replacement")
    result = comfort(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_efficiency(_technologies, _demand):
    from muse.objectives import efficiency

    _technologies["efficiency"] = add_var(_technologies, "replacement")
    result = efficiency(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_capacity_to_service_demand(_technologies, _demand):
    from muse.objectives import capacity_to_service_demand

    result = capacity_to_service_demand(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_capacity_in_use(_technologies, _demand):
    from muse.objectives import capacity_in_use

    result = capacity_in_use(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_consumption(_technologies, _demand, _prices):
    from muse.objectives import consumption

    result = consumption(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_fixed_costs(_technologies, _demand):
    from muse.objectives import fixed_costs

    result = fixed_costs(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_capital_costs(_technologies, _demand):
    from muse.objectives import capital_costs

    result = capital_costs(_technologies, _demand)
    assert set(result.dims) == {"replacement", "asset"}


def test_emission_cost(_technologies, _demand, _prices):
    from muse.objectives import emission_cost

    result = emission_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_fuel_consumption_cost(_technologies, _demand, _prices):
    from muse.objectives import fuel_consumption_cost

    result = fuel_consumption_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_annual_levelized_cost_of_energy(_technologies, _demand, _prices):
    from muse.objectives import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset"}


def test_lifetime_levelized_cost_of_energy(_technologies, _demand, _prices):
    from muse.objectives import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset"}


def test_net_present_value(_technologies, _demand, _prices):
    from muse.objectives import net_present_value

    result = net_present_value(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset"}


def test_net_present_cost(_technologies, _demand, _prices):
    from muse.objectives import net_present_cost

    result = net_present_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset"}


def test_equivalent_annual_cost(_technologies, _demand, _prices):
    from muse.objectives import equivalent_annual_cost

    result = equivalent_annual_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset"}


def add_var(coordinates, *dims, factor=100.0):
    from numpy.random import rand

    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))
