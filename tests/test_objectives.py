from pytest import approx, mark
from xarray import DataArray


def add_var(coordinates, *dims, factor=100.0):
    from numpy.random import rand

    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


@mark.usefixtures("save_registries")
def test_objective_registration():
    from muse.objectives import OBJECTIVES, register_objective

    @register_objective
    def a_objective(*args, **kwargs):
        pass

    assert "a_objective" in OBJECTIVES
    assert OBJECTIVES["a_objective"] is a_objective

    @register_objective(name="something")
    def b_objective(search_space: DataArray):
        pass

    assert "something" in OBJECTIVES
    assert OBJECTIVES["something"] is b_objective


@mark.usefixtures("save_registries")
def test_computing_objectives(demand_share, search_space):
    from muse.objectives import factory, register_objective

    @register_objective
    def first(retro_agent, demand_share, search_space, switch=True, assets=None):
        return (1 if switch else 2) * (search_space == search_space)

    @register_objective
    def second(retro_agent, demand_share, search_space, switch=True, assets=None):
        from xarray import DataArray
        from numpy import full

        shape = len(search_space.asset), len(search_space.replacement)
        result = DataArray(full(shape, 5), dims=search_space.coords)
        result[{"asset": assets}] = 3
        return result

    objectives = factory("first")(None, demand_share, search_space)
    assert set(objectives.data_vars) == {"first"}
    assert (objectives.first == 1).all()

    objectives = factory("first")(None, demand_share, search_space, False)
    assert (objectives.first == 2).all()

    objectives = factory(["first", "second"])(
        None, demand_share, search_space, False, 0
    )
    assert set(objectives.data_vars) == {"first", "second"}
    assert (objectives.first == 2).all()
    if len(objectives.asset) > 0:
        assert (objectives.second.isel(asset=0) == 3).all()
    if len(objectives.asset) > 1:
        assert (objectives.second.isel(asset=1) == 5).all()


def test_comfort(search_space, demand_share, technologies, retro_agent):
    from muse.objectives import comfort

    technologies["comfort"] = add_var(technologies, "technology")

    expected = technologies.comfort.sel(technology=search_space.replacement)
    actual = comfort(retro_agent, demand_share, search_space, technologies)
    assert set(actual.dims) == {"replacement"}
    assert actual.values == approx(expected.values)


def test_capital_costs(demand_share, search_space, technologies, retro_agent):
    from muse.objectives import capital_costs

    technologies["cap_par"] = add_var(technologies, "technology", "region", "year")
    technologies["cap_exp"] = add_var(technologies, "technology", "region", "year")
    technologies["scaling_size"] = add_var(technologies, "technology", "region", "year")

    # exp == 0
    technologies.cap_exp.loc[
        {"region": retro_agent.region, "technology": search_space.replacement}
    ] = 0
    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    assert set(actual.dims) == {"replacement"}
    expected = technologies.cap_par.sel(
        technology=search_space.replacement, region=retro_agent.region
    ).interp(year=retro_agent.year, method=retro_agent.interpolation)
    assert actual.values == approx(expected.values)

    # exp == 1
    technologies.cap_exp.loc[
        {"region": retro_agent.region, "technology": search_space.replacement}
    ] = 1
    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    assert set(actual.dims) == {"replacement"}
    expected = technologies.cap_par * technologies.scaling_size
    expected = expected.sel(
        technology=search_space.replacement, region=retro_agent.region
    ).interp(year=retro_agent.year, method=retro_agent.interpolation)
    assert actual.values == approx(expected.values)

    # exp == numbers
    technologies["scaling_size"] = add_var(technologies, "technology", "region", "year")
    expected = technologies.cap_par * technologies.scaling_size ** technologies.cap_exp
    expected = expected.sel(
        technology=search_space.replacement, region=retro_agent.region
    ).interp(year=retro_agent.year, method=retro_agent.interpolation)
    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    assert set(actual.dims) == {"replacement"}
    assert actual.values == approx(expected.values)


def test_emission_cost(
    demand_share, search_space, technologies, retro_agent, agent_market
):
    from muse.objectives import emission_cost
    from muse.commodities import is_enduse, is_pollutant
    from xarray import broadcast

    fouts = technologies.fixed_outputs.sel(
        commodity=is_pollutant(technologies.comm_usage)
    )
    envs = is_pollutant(technologies.comm_usage.sel(commodity=agent_market.commodity))
    prices = (
        agent_market.prices.sel(commodity=envs)
        .interp(year=retro_agent.forecast_year, method="linear")
        .drop_vars("year")
    )

    expected = (
        (
            demand_share.sel(
                commodity=is_enduse(
                    technologies.comm_usage.sel(commodity=demand_share.commodity)
                ),
                asset=search_space.asset,
            ).sum("commodity")
            * (prices * fouts).sum("commodity")
        )
        .sum("timeslice")
        .sel(
            region=retro_agent.region,
            technology=search_space.replacement,
            year=retro_agent.year,
        )
    )

    actual = emission_cost(
        retro_agent, demand_share, search_space, technologies, agent_market
    )
    assert {"asset", "replacement"}.issuperset(actual.dims)

    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)


def test_capacity_fulfilling_demand(
    search_space, demand_share, technologies, retro_agent, agent_market
):
    from muse.objectives import capacity_to_service_demand, fixed_costs

    # capacity
    outs = technologies.fixed_outputs.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.year,
        commodity=demand_share.commodity,
    )

    max_demand = (demand_share.where(outs > 0, 0) / outs.where(outs > 0, 1)).max(
        ("commodity", "timeslice")
    )
    max_hours = agent_market.represent_hours.max() / agent_market.represent_hours.sum()

    ufac = technologies.utilization_factor.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.year,
    )

    capacity = (max_demand / ufac / max_hours).sel(asset=search_space.asset)
    actual = capacity_to_service_demand(
        retro_agent, demand_share, search_space, technologies, agent_market
    )

    assert actual.dims == capacity.dims
    assert actual.values == approx(capacity.values)

    # fixed costs
    # TODO Move to a separate test
    technologies["fix_par"] = add_var(technologies, "technology", "region", "year")
    technologies["fix_exp"] = add_var(technologies, "technology", "region", "year")
    fpar = technologies.fix_par.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.year,
    )
    fexp = technologies.fix_exp.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.year,
    )
    expected = fpar * capacity ** fexp

    actual = fixed_costs(
        retro_agent, demand_share, search_space, technologies, agent_market
    )
    assert actual.values == approx(expected.values)
    assert set(actual.dims) == set(expected.dims)


def test_fuel_consumption(
    demand_share, search_space, technologies, retro_agent, agent_market
):
    from muse.quantities import consumption
    from muse.objectives import fuel_consumption_cost
    from muse.commodities import is_fuel
    from xarray import broadcast

    actual = fuel_consumption_cost(
        retro_agent, demand_share, search_space, technologies, agent_market
    )
    assert {"asset", "replacement"}.issuperset(actual.dims)

    demand = (
        demand_share.sel(asset=search_space.asset)
        .where(search_space, 0)
        .rename(replacement="technology")
    )
    cons = consumption(
        production=demand,
        technologies=technologies.sel(year=retro_agent.year, region=retro_agent.region),
        prices=agent_market.prices.sel(region=retro_agent.region).interp(
            year=retro_agent.forecast_year, method="linear"
        ),
    )
    fuels = is_fuel(technologies.comm_usage)
    prices = agent_market.prices.sel(commodity=fuels, region=retro_agent.region).interp(
        year=retro_agent.forecast_year, method="linear"
    )

    dims = list(set((cons * prices).dims) - {"asset", "technology"})
    expected = (cons * prices).sum(dims).sel(technology=search_space.replacement)
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)


def test_net_present_value(
    demand_share, search_space, technologies, retro_agent, agent_market
):
    """Test the net present value objective.

    It is essentially the same maths but filtering the inputs using the "sel" method
    rather than the agent.filter_input method, as well as changing the other in which
    things are added together.

    Indeed, this is more compact and elegant than the actual implementation...
    """
    from muse.objectives import (
        net_present_value,
        capacity_to_service_demand,
        discount_factor,
        fuel_consumption_cost,
        fixed_costs,
    )
    from muse.commodities import is_material, is_enduse, is_pollutant

    actual = net_present_value(
        retro_agent, demand_share, search_space, technologies, agent_market
    )

    tech = technologies.sel(
        technology=search_space.replacement,
        region=retro_agent.region,
        year=retro_agent.year,
    )

    nyears = tech.technical_life.astype(int)
    interest_rate = tech.interest_rate
    cap_par = tech.cap_par
    cap_exp = tech.cap_exp
    var_par = tech.var_par
    var_exp = tech.var_exp
    fixed_outputs = tech.fixed_outputs
    utilization_factor = tech.utilization_factor

    # All years the simulation is running and the prices
    all_years = range(retro_agent.year, retro_agent.year + nyears.values.max())
    prices = agent_market.prices.sel(region=retro_agent.region).interp(year=all_years)

    # Evolution of rates with time
    rates = discount_factor(
        years=prices.year - retro_agent.year + 1,
        interest_rate=interest_rate,
        mask=prices.year <= retro_agent.year + nyears,
    )

    # The individual prices
    prices_environmental = prices.sel(commodity=is_pollutant(technologies.comm_usage))
    prices_material = prices.sel(commodity=is_material(technologies.comm_usage))
    prices_non_env = prices.sel(commodity=is_enduse(technologies.comm_usage))

    # Capacity
    capacity = capacity_to_service_demand(
        retro_agent, demand_share, search_space, technologies, agent_market
    )

    # raw revenues --> Make the NPV more positive
    # This production is the absolute maximum production, given the capacity
    production = capacity * fixed_outputs * utilization_factor
    raw_revenues = (production * prices_non_env * rates).sum(("commodity", "year"))

    # Hours ratio
    hours_ratio = agent_market.represent_hours / agent_market.represent_hours.sum()

    # raw costs --> make the NPV more negative
    # Cost of installed capacity
    installed_capacity_costs = hours_ratio * cap_par * capacity ** cap_exp

    # Fuel/energy costs
    fuel_cost = fuel_consumption_cost(
        retro_agent, demand_share, search_space, technologies, agent_market
    )
    fuel_consumption_costs = (fuel_cost * rates).sum("year") * hours_ratio

    # Cost related to environmental products
    environmental_costs = (production * prices_environmental * rates).sum(
        ("commodity", "year")
    )

    # Cost related to material other than fuel/energy and environmentals
    material_costs = (production * prices_material * rates).sum(("commodity", "year"))

    # Fixed and Variable costs
    non_env_production = production.sel(commodity=is_enduse(technologies.comm_usage))
    fix_costs = (
        rates
        * hours_ratio
        * fixed_costs(
            retro_agent, demand_share, search_space, technologies, agent_market
        )
    ).sum("year")
    variable_costs = (
        rates * hours_ratio * var_par * non_env_production ** var_exp
    ).sum(("commodity", "year"))
    fixed_and_variable_costs = fix_costs + variable_costs

    assert set(installed_capacity_costs.dims) == set(fuel_consumption_costs.dims)
    assert set(environmental_costs.dims) == set(fuel_consumption_costs.dims)
    assert set(material_costs.dims) == set(fuel_consumption_costs.dims)
    assert set(fixed_and_variable_costs.dims) == set(fuel_consumption_costs.dims)
    raw_costs = (
        installed_capacity_costs
        + fuel_consumption_costs
        + environmental_costs
        + material_costs
        + fixed_and_variable_costs
    )

    assert set(raw_revenues.dims) == set(raw_costs.dims)
    expected = (raw_revenues - raw_costs).sum("timeslice")

    assert {"replacement", "asset"}.issuperset(actual.dims)
    assert actual.values == approx(expected.values, rel=5e-5)
