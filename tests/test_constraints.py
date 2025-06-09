from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, fixture, mark, raises

from muse import examples
from muse.constraints import (
    demand,
    demand_limiting_capacity,
    lp_costs,
    max_capacity_expansion,
    max_production,
)
from muse.quantities import maximum_production
from muse.timeslices import drop_timeslice
from muse.utilities import broadcast_over_assets, interpolate_capacity, reduce_assets

CURRENT_YEAR = 2020
INVESTMENT_YEAR = 2025


@fixture(params=["timeslice_as_list", "timeslice_as_multindex"])
def constraint_data(request):
    """Creates the complete dataset needed for constraint testing.

    The transformation follows these steps:
    1. Load residential sector data
    2. Extract technologies and assets
    3. Calculate capacity and market demand
    4. Create search space and costs
    5. Generate all constraint-specific data

    Returns:
        dict: Contains all necessary data for constraint testing:
            - technologies: Technology parameters
            - assets: Asset data
            - capacity: Interpolated capacity data
            - market_demand: Calculated market demand
            - search_space: Search space for investments
            - costs: Cost data for each option
            - commodities: List of relevant commodities
            - lp_costs: Linear programming cost data
            - constraints: Dict of all constraint matrices
    """
    # Step 1: Load residential sector data
    residential = examples.sector("residential", model="medium")

    # Step 2: Extract technologies and assets
    technologies = residential.technologies.squeeze("region").sel(year=INVESTMENT_YEAR)
    assets = next(a.assets for a in residential.agents)

    # Step 3: Calculate capacity and market demand
    capacity = interpolate_capacity(
        reduce_assets(assets.capacity, coords=("technology", "region")),
        year=[CURRENT_YEAR, INVESTMENT_YEAR],
    )

    # Create initial market demand as 80% of maximum production
    market_demand = 0.8 * maximum_production(
        broadcast_over_assets(technologies, assets),
        assets.capacity,
    ).sel(year=INVESTMENT_YEAR).groupby("technology").sum("asset").rename(
        technology="asset"
    )
    # Remove un-demanded commodities
    market_demand = market_demand.sel(
        commodity=(market_demand > 0).any(dim=["timeslice", "asset"])
    )

    # Step 4: Create search space and costs
    search_space = examples.search_space("residential", "medium")
    search_space = search_space.sel(asset=assets.technology.values)
    shape = search_space.shape
    costs = search_space * np.arange(np.prod(shape)).reshape(shape)

    # Get list of commodities
    commodities = list(market_demand.commodity.values)

    # Step 5: Generate constraints
    constraints = {
        "max_production": max_production(
            market_demand, capacity, search_space, technologies
        ),
        "demand": demand(market_demand, capacity, search_space, technologies),
        "max_capacity_expansion": max_capacity_expansion(
            market_demand, capacity, search_space, technologies
        ),
        "demand_limiting_capacity": demand_limiting_capacity(
            market_demand, capacity, search_space, technologies
        ),
    }

    # Testing two different timeslicing formats
    if request.param == "timeslice_as_multindex":
        constraints = {key: _as_list(cs) for key, cs in constraints.items()}

    # Step 6: Generate lp costs
    lp_cost_data = lp_costs(costs, commodities=commodities)

    return {
        "technologies": technologies,
        "capacity": capacity,
        "market_demand": market_demand,
        "search_space": search_space,
        "costs": costs,
        "commodities": commodities,
        "lp_costs": lp_cost_data,
        "constraints": constraints,
    }


def _as_list(data: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    """Helper function to convert timeslice data to list format."""
    if "timeslice" in data.dims:
        data = data.copy(deep=False)
        index = pd.MultiIndex.from_tuples(
            data.get_index("timeslice"), names=("month", "day", "hour")
        )
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, "timeslice")
        data = drop_timeslice(data).assign_coords(mindex_coords)
    return data


def test_fixtures(constraint_data):
    """Validating that the fixture data has appropriate dimensions."""
    assert set(constraint_data["technologies"].dims) == {"technology", "commodity"}
    assert set(constraint_data["search_space"].dims) == {"asset", "replacement"}
    assert set(constraint_data["costs"].dims) == {"asset", "replacement"}
    assert set(constraint_data["capacity"].dims) == {"asset", "year"}
    assert set(constraint_data["market_demand"].dims) == {
        "asset",
        "commodity",
        "timeslice",
    }


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


def test_constraints_dimensions(constraint_data):
    """Test dimensions of all constraint matrices."""
    constraints = constraint_data["constraints"]

    # Max production constraint
    max_prod_dims = {"asset", "commodity", "replacement", "timeslice"}
    assert set(constraints["max_production"].capacity.dims) == max_prod_dims
    assert set(constraints["max_production"].production.dims) == max_prod_dims
    assert set(constraints["max_production"].b.dims) == max_prod_dims

    # Demand constraint
    assert set(constraints["demand"].capacity.dims) == set()
    assert set(constraints["demand"].production.dims) == set()
    assert set(constraints["demand"].b.dims) == {"asset", "commodity", "timeslice"}

    # Demand limiting capacity constraint
    assert set(constraints["demand_limiting_capacity"].capacity.dims) == {
        "asset",
        "commodity",
        "replacement",
    }
    assert set(constraints["demand_limiting_capacity"].production.dims) == set()
    assert set(constraints["demand_limiting_capacity"].b.dims) == {"asset", "commodity"}

    # Max capacity expansion constraint
    assert set(constraints["max_capacity_expansion"].capacity.dims) == set()
    assert set(constraints["max_capacity_expansion"].production.dims) == set()
    assert set(constraints["max_capacity_expansion"].b.dims) == {"replacement"}


def test_lp_constraints_matrix_b_is_scalar(constraint_data):
    """B is a scalar - output should be equivalent to a single row matrix."""
    from muse.constraints import lp_constraint_matrix

    constraint = constraint_data["constraints"]["max_production"]
    lpcosts = constraint_data["lp_costs"]

    for attr in ["capacity", "production"]:
        lpconstraint = lp_constraint_matrix(
            xr.DataArray(1), getattr(constraint, attr), getattr(lpcosts, attr)
        )
        expected_value = -1 if attr == "capacity" else 1
        assert lpconstraint.values == approx(expected_value)
        assert set(lpconstraint.dims) == {
            f"d({x})" for x in getattr(lpcosts, attr).dims
        }


def test_max_production_constraint_diagonal(constraint_data):
    """Test production side of max capacity production is diagonal."""
    from muse.constraints import lp_constraint_matrix

    constraint = constraint_data["constraints"]["max_production"]
    lpcosts = constraint_data["lp_costs"]

    # Test capacity constraints
    result = lp_constraint_matrix(constraint.b, constraint.capacity, lpcosts.capacity)
    decision_dims = {f"d({x})" for x in lpcosts.capacity.dims}
    constraint_dims = {
        f"c({x})" for x in set(lpcosts.production.dims).union(constraint.b.dims)
    }
    assert set(result.dims) == decision_dims.union(constraint_dims)

    # Test production constraints
    result = lp_constraint_matrix(
        constraint.b, constraint.production, lpcosts.production
    )
    decision_dims = {f"d({x})" for x in lpcosts.production.dims}
    assert set(result.dims) == decision_dims.union(constraint_dims)

    # Verify diagonal matrix
    result = result.reset_index("d(timeslice)", drop=True).assign_coords(
        {"d(timeslice)": result["d(timeslice)"].values}
    )
    stacked = result.stack(d=sorted(decision_dims), c=sorted(constraint_dims))
    assert stacked.shape[0] == stacked.shape[1]
    assert stacked.values == approx(np.eye(stacked.shape[0]))


def test_lp_constraint(constraint_data):
    from muse.constraints import lp_constraint

    constraint = constraint_data["constraints"]["max_production"]
    lpcosts = constraint_data["lp_costs"]

    result = lp_constraint(constraint, lpcosts)
    constraint_dims = {
        f"c({x})" for x in set(lpcosts.production.dims).union(constraint.b.dims)
    }

    # Test capacity constraints
    decision_dims = {f"d({x})" for x in lpcosts.capacity.dims}
    assert set(result.capacity.dims) == decision_dims.union(constraint_dims)

    # Test production constraints
    decision_dims = {f"d({x})" for x in lpcosts.production.dims}
    assert set(result.production.dims) == decision_dims.union(constraint_dims)
    stacked = result.production.stack(
        d=sorted(decision_dims), c=sorted(constraint_dims)
    )
    assert stacked.shape[0] == stacked.shape[1]
    assert stacked.values == approx(np.eye(stacked.shape[0]))

    assert set(result.b.dims) == constraint_dims
    assert result.b.values == approx(0)


def test_to_scipy_adapter_maxprod(constraint_data):
    """Test scipy adapter with max production constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        constraint_data["costs"],
        constraints=[constraint_data["constraints"]["max_production"]],
        commodities=constraint_data["commodities"],
    )
    lpcosts = constraint_data["lp_costs"]

    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_ub is not None
    assert adapter.b_ub is not None
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1
    assert adapter.b_ub.ndim == 1
    assert adapter.A_ub.ndim == 2

    capsize = lpcosts.capacity.size  # number of capacity decision variables
    prodsize = lpcosts.production.size  # number of production decision variables
    bsize = adapter.b_ub.size  # number of constraints

    assert adapter.c.size == capsize + prodsize
    assert adapter.A_ub.shape[0] == bsize
    assert adapter.A_ub.shape[1] == capsize + prodsize

    assert (
        bsize
        == lpcosts.commodity.size
        * lpcosts.timeslice.size
        * lpcosts.asset.size
        * lpcosts.replacement.size
    )
    assert adapter.b_ub == approx(0)
    assert adapter.A_ub[:, capsize:] == approx(np.eye(prodsize))


def test_to_scipy_adapter_demand(constraint_data):
    """Test scipy adapter with demand constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        constraint_data["costs"],
        constraints=[constraint_data["constraints"]["demand"]],
        commodities=constraint_data["commodities"],
    )
    lpcosts = constraint_data["lp_costs"]

    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_ub is not None
    assert adapter.b_ub is not None
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1
    assert adapter.b_ub.ndim == 1
    assert adapter.A_ub.ndim == 2

    capsize = lpcosts.capacity.size  # number of capacity decision variables
    prodsize = lpcosts.production.size  # number of production decision variables
    bsize = adapter.b_ub.size  # number of constraints

    assert adapter.c.size == capsize + prodsize
    assert adapter.A_ub.shape[0] == bsize
    assert adapter.A_ub.shape[1] == capsize + prodsize

    assert bsize == lpcosts.commodity.size * lpcosts.timeslice.size * lpcosts.asset.size
    assert adapter.A_ub[:, :capsize] == approx(0)
    assert set(adapter.A_ub[:, capsize:].flatten()) == {0.0, -1.0}


def test_to_scipy_adapter_max_capacity_expansion(constraint_data):
    """Test scipy adapter with max capacity expansion constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        constraint_data["costs"],
        constraints=[constraint_data["constraints"]["max_capacity_expansion"]],
        commodities=constraint_data["commodities"],
    )
    lpcosts = constraint_data["lp_costs"]

    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_ub is not None
    assert adapter.b_ub is not None
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1
    assert adapter.b_ub.ndim == 1
    assert adapter.A_ub.ndim == 2

    capsize = lpcosts.capacity.size  # number of capacity decision variables
    prodsize = lpcosts.production.size  # number of production decision variables
    bsize = adapter.b_ub.size  # number of constraints

    assert adapter.c.size == capsize + prodsize
    assert adapter.A_ub.shape[0] == bsize
    assert adapter.A_ub.shape[1] == capsize + prodsize

    assert bsize == lpcosts.replacement.size
    assert adapter.A_ub[:, capsize:] == approx(0)
    assert set(adapter.A_ub[:, :capsize].flatten()) == {0.0, 1.0}


def test_scipy_adapter_no_constraint(constraint_data):
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        constraint_data["costs"],
        constraints=[],
        commodities=constraint_data["commodities"],
    )
    lpcosts = constraint_data["lp_costs"]

    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert all(
        getattr(adapter, attr) is None for attr in ["A_ub", "b_ub", "A_eq", "b_eq"]
    )
    assert adapter.c.ndim == 1
    assert adapter.c.size == lpcosts.capacity.size + lpcosts.production.size


def test_back_to_muse_quantities(constraint_data):
    from muse.constraints import ScipyAdapter

    lpcosts = constraint_data["lp_costs"]
    data = ScipyAdapter._unified_dataset(lpcosts)

    # Test capacity
    lpquantity = ScipyAdapter._selected_quantity(data, "capacity")
    assert set(lpquantity.dims) == {"d(asset)", "d(replacement)"}
    copy = ScipyAdapter._back_to_muse_quantity(
        lpquantity.costs.values, xr.zeros_like(lpquantity.costs)
    )
    assert (copy == lpcosts.capacity).all()

    # Test production
    lpquantity = ScipyAdapter._selected_quantity(data, "production")
    assert set(lpquantity.dims) == {
        "d(asset)",
        "d(replacement)",
        "d(timeslice)",
        "d(commodity)",
    }
    copy = ScipyAdapter._back_to_muse_quantity(
        lpquantity.costs.values, xr.zeros_like(lpquantity.costs)
    )
    assert (copy == lpcosts.production).all()


def test_back_to_muse_all(constraint_data):
    from muse.constraints import ScipyAdapter

    lpcosts = constraint_data["lp_costs"]
    data = ScipyAdapter._unified_dataset(lpcosts)
    lpcapacity = ScipyAdapter._selected_quantity(data, "capacity")
    lpproduction = ScipyAdapter._selected_quantity(data, "production")

    x = np.concatenate(
        (
            lpcosts.capacity.transpose(
                *[u[2:-1] for u in lpcapacity.dims]
            ).values.flatten(),
            lpcosts.production.transpose(
                *[u[2:-1] for u in lpproduction.dims]
            ).values.flatten(),
        )
    )

    copy = ScipyAdapter._back_to_muse(
        x, xr.zeros_like(lpcapacity.costs), xr.zeros_like(lpproduction.costs)
    )
    assert copy.capacity.size + copy.production.size == x.size
    assert (copy.capacity == lpcosts.capacity).all()
    assert (copy.production == lpcosts.production).all()


def test_scipy_adapter_standard_constraints(constraint_data):
    from muse.constraints import ScipyAdapter

    constraints = [
        constraint_data["constraints"]["max_production"],
        constraint_data["constraints"]["max_capacity_expansion"],
        constraint_data["constraints"]["demand"],
        constraint_data["constraints"]["demand_limiting_capacity"],
    ]

    adapter = ScipyAdapter.factory(
        constraint_data["costs"],
        constraints,
        commodities=constraint_data["commodities"],
    )

    n_constraints = adapter.b_ub.size
    n_decision_vars = adapter.c.size

    assert (
        n_decision_vars
        == constraint_data["costs"].size + constraints[0].production.size
    )
    assert adapter.b_eq is None
    assert adapter.A_eq is None
    assert adapter.A_ub.shape == (n_constraints, n_decision_vars)
    assert n_constraints == sum(c.b.size for c in constraints)


def test_scipy_solver(constraint_data):
    """Test the scipy solver for demand matching."""
    from muse.investments import scipy_match_demand

    constraints = [
        constraint_data["constraints"]["max_production"],
        constraint_data["constraints"]["max_capacity_expansion"],
        constraint_data["constraints"]["demand"],
        constraint_data["constraints"]["demand_limiting_capacity"],
    ]

    solution = scipy_match_demand(
        costs=constraint_data["costs"],
        search_space=xr.ones_like(constraint_data["costs"]),
        technologies=constraint_data["technologies"],
        constraints=constraints,
        commodities=constraint_data["commodities"],
    )
    assert isinstance(solution, xr.DataArray)
    assert set(solution.dims) == {"asset", "replacement"}


def test_minimum_service(constraint_data):
    from muse.constraints import minimum_service

    # Test with no minimum service factor
    assert (
        minimum_service(
            constraint_data["market_demand"],
            constraint_data["capacity"],
            constraint_data["search_space"],
            constraint_data["technologies"],
        )
        is None
    )

    # Test with minimum service factor
    technologies = constraint_data["technologies"].copy()
    technologies["minimum_service_factor"] = 0.4 * xr.ones_like(
        technologies.technology, dtype=float
    )
    min_service = minimum_service(
        constraint_data["market_demand"],
        constraint_data["capacity"],
        constraint_data["search_space"],
        technologies,
    )
    assert isinstance(min_service, xr.Dataset)


def test_max_capacity_expansion_properties(constraint_data):
    max_capacity_expansion = constraint_data["constraints"]["max_capacity_expansion"]
    assert (max_capacity_expansion.capacity == 1).all()
    assert max_capacity_expansion.production == 0
    assert max_capacity_expansion.b.dims == ("replacement",)
    assert max_capacity_expansion.b.shape == (4,)
    assert (
        max_capacity_expansion.replacement
        == ["estove", "gasboiler", "gasstove", "heatpump"]
    ).all()


def test_max_capacity_expansion_no_limits(constraint_data):
    from muse.constraints import max_capacity_expansion

    techs = constraint_data["technologies"].drop_vars(
        ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
    )
    assert (
        max_capacity_expansion(
            constraint_data["market_demand"],
            constraint_data["capacity"],
            constraint_data["search_space"],
            techs,
        )
        is None
    )


def test_max_capacity_expansion_seed(constraint_data):
    from muse.constraints import max_capacity_expansion

    seed = 10
    technologies = constraint_data["technologies"].copy()
    technologies["growth_seed"] = seed

    # Test different capacity scenarios
    scenarios = [0, seed, 2 * seed]
    results = []
    for cap in scenarios:
        capacity = constraint_data["capacity"].copy()
        capacity.sel(year=2020)[:] = cap
        results.append(
            max_capacity_expansion(
                constraint_data["market_demand"],
                capacity,
                constraint_data["search_space"],
                technologies,
            )
        )

    # Zero capacity should match seed capacity
    assert results[0].b.values == approx(results[1].b.values)
    # Higher capacity should differ
    assert results[0].b.values != approx(results[2].b.values)


def test_max_capacity_expansion_infinite_limits(constraint_data):
    from muse.constraints import max_capacity_expansion

    technologies = constraint_data["technologies"].copy()
    for limit in [
        "max_capacity_addition",
        "max_capacity_growth",
        "total_capacity_limit",
    ]:
        technologies[limit] = np.inf
    with raises(ValueError):
        max_capacity_expansion(
            constraint_data["market_demand"],
            constraint_data["capacity"],
            constraint_data["search_space"],
            technologies,
        )


def test_max_production(constraint_data):
    assert (constraint_data["constraints"]["max_production"].capacity <= 0).all()


def test_demand_limiting_capacity(constraint_data):
    demand_limiting_capacity = constraint_data["constraints"][
        "demand_limiting_capacity"
    ]
    max_production = constraint_data["constraints"]["max_production"]
    demand_constraint = constraint_data["constraints"]["demand"]

    # Test capacity values
    expected_capacity = (
        -max_production.capacity.max("timeslice").values
        if "timeslice" in max_production.capacity.dims
        else -max_production.capacity.values
    )
    assert demand_limiting_capacity.capacity.values == approx(expected_capacity)

    # Test production and b values
    assert demand_limiting_capacity.production == 0
    expected_b = (
        demand_constraint.b.max("timeslice").values
        if "timeslice" in demand_constraint.b.dims
        else demand_constraint.b.values
    )
    assert demand_limiting_capacity.b.values == approx(expected_b)
