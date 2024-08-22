from pytest import fixture, mark


@fixture
def _demand(demand_share, search_space):
    reduced_demand = demand_share.sel(
        {
            k: search_space[k]
            for k in set(demand_share.dims).intersection(search_space.dims)
        }
    )
    reduced_demand["year"] = 2030
    return reduced_demand


@fixture
def _technologies(technologies, retro_agent, search_space):
    techs = retro_agent.filter_input(
        technologies,
        technology=search_space.replacement,
        year=retro_agent.forecast_year,
    ).drop_vars("technology")
    return techs


@fixture
def _prices(retro_agent, agent_market):
    prices = retro_agent.filter_input(agent_market.prices)
    return prices


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
def test_computing_objectives(_technologies, _demand):
    from muse.objectives import factory, register_objective

    @register_objective
    def first(technologies, switch=True, *args, **kwargs):
        from xarray import full_like

        value = 1 if switch else 2
        result = full_like(technologies["replacement"], value, dtype=float)
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
    objectives = factory("first")(technologies=_technologies, switch=True)
    assert set(objectives.data_vars) == {"first"}
    assert (objectives.first == 1).all()
    objectives = factory("first")(technologies=_technologies, switch=False)
    assert (objectives.first == 2).all()

    # Test multiple objectives
    objectives = factory(["first", "second"])(
        technologies=_technologies, demand=_demand, switch=False, assets=0
    )
    assert set(objectives.data_vars) == {"first", "second"}
    assert (objectives.first == 2).all()
    if len(objectives.asset) > 0:
        assert (objectives.second.isel(asset=0) == 3).all()
    if len(objectives.asset) > 1:
        assert (objectives.second.isel(asset=1) == 5).all()


def test_comfort(_technologies):
    from muse.objectives import comfort

    _technologies["comfort"] = add_var(_technologies, "replacement")
    result = comfort(_technologies)
    assert set(result.dims) == {"replacement"}


def test_efficiency(_technologies):
    from muse.objectives import efficiency

    _technologies["efficiency"] = add_var(_technologies, "replacement")
    result = efficiency(_technologies)
    assert set(result.dims) == {"replacement"}


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


def test_capital_costs(_technologies):
    from muse.objectives import capital_costs

    _technologies["scaling_size"] = add_var(_technologies, "replacement")
    result = capital_costs(_technologies)
    assert set(result.dims) == {"replacement"}


def test_emission_cost(_technologies, _demand, _prices):
    from muse.objectives import emission_cost

    result = emission_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_fuel_consumption(_technologies, _demand, _prices):
    from muse.objectives import fuel_consumption_cost

    result = fuel_consumption_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_annual_levelized_cost_of_energy(_technologies, _demand, _prices):
    from muse.objectives import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement"}


def test_lifetime_levelized_cost_of_energy(_technologies, _demand, _prices):
    from muse.objectives import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_net_present_value(_technologies, _demand, _prices):
    from muse.objectives import net_present_value

    result = net_present_value(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_net_present_cost(_technologies, _demand, _prices):
    from muse.objectives import net_present_cost

    result = net_present_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_equivalent_annual_cost(_technologies, _demand, _prices):
    from muse.objectives import equivalent_annual_cost

    result = equivalent_annual_cost(_technologies, _demand, _prices)
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def add_var(coordinates, *dims, factor=100.0):
    from numpy.random import rand

    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))
