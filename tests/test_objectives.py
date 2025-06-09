from numpy.random import rand
from pytest import fixture, mark

YEAR = 2030


@fixture
def objective_data(technologies, market, demand_share):
    """Creates the complete dataset needed for objective calculations.

    The transformation follows these steps:
    1. Extract year-specific data from technologies and market
    2. Transform technology data to asset level with replacement dimension
    3. Transform price data to asset level
    4. Add any additional variables needed for specific objectives

    Returns:
        dict: Contains all necessary data for objective calculations:
            - technologies: Technology parameters with replacement dimension
            - prices: Prices relevant to each asset
            - demand: Demand share data
    """
    from muse.utilities import broadcast_over_assets

    # Step 1: Extract year-specific data
    tech_year = technologies.sel(year=YEAR).rename(technology="replacement")
    prices_year = market.prices.sel(year=YEAR)

    # Step 2 & 3: Transform to asset level
    tech_assets = broadcast_over_assets(tech_year, demand_share)
    prices_assets = broadcast_over_assets(
        prices_year, demand_share, installed_as_year=False
    )

    # Step 4: Add computed variables needed by some objectives
    tech_assets["comfort"] = _add_var(tech_assets, "replacement")
    tech_assets["efficiency"] = _add_var(tech_assets, "replacement")
    tech_assets["scaling_size"] = _add_var(tech_assets, "replacement")

    return {
        "technologies": tech_assets,
        "prices": prices_assets,
        "demand": demand_share,
    }


def _add_var(coordinates, *dims, factor=100.0):
    """Helper function to add random variables with specified dimensions."""
    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


def test_fixtures(objective_data):
    """Validating that the fixture data has appropriate dimensions."""
    assert set(objective_data["technologies"].dims) == {
        "asset",
        "commodity",
        "replacement",
    }
    assert set(objective_data["demand"].dims) == {"asset", "commodity", "timeslice"}
    assert set(objective_data["prices"].dims) == {"asset", "commodity", "timeslice"}


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
def test_computing_objectives(objective_data):
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
        technologies=objective_data["technologies"],
        demand=objective_data["demand"],
        prices=objective_data["prices"],
        switch=True,
    )
    assert set(objectives.data_vars) == {"first"}
    assert (objectives.first == 1).all()

    objectives = factory("first")(
        technologies=objective_data["technologies"],
        demand=objective_data["demand"],
        prices=objective_data["prices"],
        switch=False,
    )
    assert (objectives.first == 2).all()

    # Test multiple objectives
    objectives = factory(["first", "second"])(
        technologies=objective_data["technologies"],
        demand=objective_data["demand"],
        prices=objective_data["prices"],
        switch=False,
        assets=0,
    )
    assert set(objectives.data_vars) == {"first", "second"}
    assert (objectives.first == 2).all()
    if len(objectives.asset) > 0:
        assert (objectives.second.isel(asset=0) == 3).all()
    if len(objectives.asset) > 1:
        assert (objectives.second.isel(asset=1) == 5).all()


def test_comfort(objective_data):
    from muse.objectives import comfort

    result = comfort(objective_data["technologies"], objective_data["demand"])
    assert set(result.dims) == {"replacement", "asset"}


def test_efficiency(objective_data):
    from muse.objectives import efficiency

    result = efficiency(objective_data["technologies"], objective_data["demand"])
    assert set(result.dims) == {"replacement", "asset"}


def test_capacity_to_service_demand(objective_data):
    from muse.objectives import capacity_to_service_demand

    result = capacity_to_service_demand(
        objective_data["technologies"], objective_data["demand"]
    )
    assert set(result.dims) == {"replacement", "asset"}


def test_capacity_in_use(objective_data):
    from muse.objectives import capacity_in_use

    result = capacity_in_use(objective_data["technologies"], objective_data["demand"])
    assert set(result.dims) == {"replacement", "asset"}


def test_consumption(objective_data):
    from muse.objectives import consumption

    result = consumption(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_fixed_costs(objective_data):
    from muse.objectives import fixed_costs

    result = fixed_costs(objective_data["technologies"], objective_data["demand"])
    assert set(result.dims) == {"replacement", "asset"}


def test_capital_costs(objective_data):
    from muse.objectives import capital_costs

    result = capital_costs(objective_data["technologies"], objective_data["demand"])
    assert set(result.dims) == {"replacement", "asset"}


def test_emission_cost(objective_data):
    from muse.objectives import emission_cost

    result = emission_cost(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_fuel_consumption_cost(objective_data):
    from muse.objectives import fuel_consumption_cost

    result = fuel_consumption_cost(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset", "timeslice"}


def test_annual_levelized_cost_of_energy(objective_data):
    from muse.objectives import annual_levelized_cost_of_energy

    result = annual_levelized_cost_of_energy(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset"}


def test_lifetime_levelized_cost_of_energy(objective_data):
    from muse.objectives import lifetime_levelized_cost_of_energy

    result = lifetime_levelized_cost_of_energy(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset"}


def test_net_present_value(objective_data):
    from muse.objectives import net_present_value

    result = net_present_value(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset"}


def test_net_present_cost(objective_data):
    from muse.objectives import net_present_cost

    result = net_present_cost(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset"}


def test_equivalent_annual_cost(objective_data):
    from muse.objectives import equivalent_annual_cost

    result = equivalent_annual_cost(
        objective_data["technologies"],
        objective_data["demand"],
        objective_data["prices"],
    )
    assert set(result.dims) == {"replacement", "asset"}
