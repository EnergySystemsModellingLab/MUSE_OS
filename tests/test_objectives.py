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
    import numpy as np

    technologies["cap_par"] = add_var(technologies, "technology", "region", "year")
    technologies["cap_exp"] = add_var(technologies, "technology", "region", "year")
    technologies["scaling_size"] = add_var(technologies, "technology", "region", "year")
    minyear = technologies.year.values.min()
    maxyear = technologies.year.values.max()
    years = np.linspace(minyear, maxyear, retro_agent.forecast).astype(int)
    technologies = technologies.interp(year=years, method=retro_agent.interpolation)
    # exp == 0
    technologies.cap_exp.loc[
        {"region": retro_agent.region, "technology": search_space.replacement}
    ] = 0
    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    actual = actual.sum("timeslice")
    assert set(actual.dims) == {"replacement"}
    expected = technologies.cap_par.sel(
        technology=search_space.replacement,
        region=retro_agent.region,
        year=retro_agent.forecast_year,
    )

    assert actual.values == approx(expected.values)

    # exp == 1
    technologies.cap_exp.loc[
        {"region": retro_agent.region, "technology": search_space.replacement}
    ] = 1
    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    actual = actual.sum("timeslice")
    assert set(actual.dims) == {"replacement"}
    expected = technologies.cap_par * technologies.scaling_size
    expected = expected.sel(
        technology=search_space.replacement,
        region=retro_agent.region,
        year=retro_agent.forecast_year,
    )

    assert actual.values == approx(expected.values)

    # exp == numbers
    technologies["scaling_size"] = add_var(technologies, "technology", "region", "year")
    expected = technologies.cap_par * technologies.scaling_size ** technologies.cap_exp
    expected = expected.sel(
        technology=search_space.replacement,
        region=retro_agent.region,
        year=retro_agent.forecast_year,
    )

    actual = capital_costs(retro_agent, demand_share, search_space, technologies)
    actual = actual.sum("timeslice")
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
        .interp(year=retro_agent.forecast_year)
        .sel(region=retro_agent.region, technology=search_space.replacement)
    )

    actual = emission_cost(
        retro_agent, demand_share, search_space, technologies, agent_market
    ).sum("timeslice")
    assert {"asset", "replacement"}.issuperset(actual.dims)

    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)


def test_capacity_fulfilling_demand(
    search_space, demand_share, technologies, retro_agent, agent_market
):
    from muse.objectives import capacity_to_service_demand, fixed_costs
    import numpy as np

    minyear = technologies.year.values.min()
    maxyear = technologies.year.values.max()
    years = np.linspace(minyear, maxyear, retro_agent.forecast).astype(int)
    technologies = technologies.interp(year=years, method=retro_agent.interpolation)

    # capacity
    outs = technologies.fixed_outputs.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.forecast_year,
        commodity=demand_share.commodity,
    )

    max_demand = (demand_share.where(outs > 0, 0) / outs.where(outs > 0, 1)).max(
        ("commodity", "timeslice")
    )
    max_hours = agent_market.represent_hours.max() / agent_market.represent_hours.sum()

    ufac = technologies.utilization_factor.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.forecast_year,
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
        year=retro_agent.forecast_year,
    )
    fexp = technologies.fix_exp.sel(
        region=retro_agent.region,
        technology=search_space.replacement,
        year=retro_agent.forecast_year,
    )
    expected = fpar * capacity ** fexp

    actual = fixed_costs(
        retro_agent, demand_share, search_space, technologies, agent_market
    ).sum("timeslice")
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
    ).sum("timeslice")
    assert {"asset", "replacement"}.issuperset(actual.dims)

    demand = (
        demand_share.sel(asset=search_space.asset)
        .where(search_space, 0)
        .rename(replacement="technology")
    )
    cons = consumption(
        production=demand,
        technologies=technologies.sel(region=retro_agent.region).interp(
            year=retro_agent.forecast_year, method="linear"
        ),
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
.
    """
    import xarray
    from muse.objectives import (
        net_present_value,
        capacity_to_service_demand,
        discount_factor,
    )
    from muse.quantities import consumption
    from muse.commodities import is_material, is_enduse, is_fuel, is_pollutant

    #    tech = technologies.interp(year=retro_agent.forecast_year, method="linear")
    tech = technologies.interp(
        year=range(agent_market.year.values.min(), agent_market.year.values.max()),
        method="linear",
    )
    actual = net_present_value(
        retro_agent, demand_share, search_space, tech, agent_market,
    ).sum("timeslice")
    tech = tech.sel(
        technology=search_space.replacement,
        region=retro_agent.region,
        year=retro_agent.forecast_year,
    )

    nyears = tech.technical_life.astype(int)
    years = range(
        retro_agent.year,
        max(retro_agent.year + nyears.values.max(), retro_agent.forecast_year + 1),
    )
    # mask in discoutn rate could give error if different dimension
    # used if all_ears is array
    # ifxarray used, dimension differences d not tirgger errors
    all_years = xarray.DataArray(years, coords={"year": years}, dims="year")
    interest_rate = tech.interest_rate
    cap_par = tech.cap_par
    cap_exp = tech.cap_exp
    var_par = tech.var_par
    var_exp = tech.var_exp
    fix_par = tech.fix_par
    fix_exp = tech.fix_exp
    fixed_outputs = tech.fixed_outputs
    utilization_factor = tech.utilization_factor

    # All years the simulation is running and the prices
    prices = agent_market.prices.interp(year=all_years)

    # Evolution of rates with time
    rates = discount_factor(
        years=all_years - retro_agent.year + 1,
        interest_rate=interest_rate,
        mask=all_years <= retro_agent.year + nyears,
    )
    print("test")

    # The individual prices
    prices_environmental = prices.sel(
        commodity=is_pollutant(technologies.comm_usage), region=retro_agent.region
    ).ffill("year")
    prices_material = prices.sel(
        commodity=is_material(technologies.comm_usage), region=retro_agent.region
    ).ffill("year")
    prices_non_env = prices.sel(
        commodity=is_enduse(technologies.comm_usage), region=retro_agent.region
    ).ffill("year")
    prices_fuel = prices.sel(
        commodity=is_fuel(technologies.comm_usage), region=retro_agent.region
    ).ffill("year")
    # Capacity
    capacity = capacity_to_service_demand(
        retro_agent, demand_share, search_space, technologies, agent_market
    )

    # Hours ratio
    hours_ratio = agent_market.represent_hours / agent_market.represent_hours.sum()

    # raw revenues --> Make the NPV more positive
    # This production is the absolute maximum production, given the capacity
    production = hours_ratio * capacity * fixed_outputs * utilization_factor
    raw_revenues = (production * prices_non_env * rates).sum(("commodity", "year"))

    # raw costs --> make the NPV more negative
    # Cost of installed capacity
    installed_capacity_costs = hours_ratio * cap_par * capacity ** cap_exp

    # Fuel/energy costs
    fuel = consumption(
        technologies=tech,
        production=production,
        prices=prices.sel(region=retro_agent.region),
    ).sel(commodity=is_fuel(tech.comm_usage))
    fuel_consumption_costs = (
        (fuel * prices_fuel * rates).sum(("commodity", "year"))
    ).drop_vars("technology")

    # Cost related to environmental products
    environmental_costs = (production * prices_environmental * rates).sum(
        ("commodity", "year")
    )

    # Cost related to material other than fuel/energy and environmentals
    material_costs = (production * prices_material * rates).sum(("commodity", "year"))

    # Fixed and Variable costs
    non_env_production = production.sel(commodity=is_enduse(tech.comm_usage)).sum(
        "commodity"
    )
    fix_costs = ((rates * hours_ratio * fix_par * capacity ** fix_exp)).sum("year")

    print(
        all_years,
        fuel.sum(),
        fuel.replacement,
        tech.technical_life,
        environmental_costs.sum(),
    )
    variable_costs = (rates * var_par * (non_env_production ** var_exp)).sum("year")
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
    assert actual.values == approx(expected.values, rel=1e-2)
