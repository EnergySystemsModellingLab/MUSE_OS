from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, fixture, raises

from muse import examples
from muse.constraints import (
    demand,
    demand_limiting_capacity,
    max_capacity_expansion,
    max_production,
    minimum_service,
)
from muse.lp_adapter import (
    ScipyAdapter,
    back_to_muse_quantity,
    lp_constraint_matrix,
    selected_quantity,
    unified_dataset,
)
from muse.timeslices import drop_timeslice
from muse.utilities import broadcast_over_assets, interpolate_capacity, reduce_assets

CURRENT_YEAR = 2020
INVESTMENT_YEAR = 2025


@fixture
def model_data():
    """Model data required for the constraints.

    Returns:
        dict: Contains all necessary data for building constraints:
            - technologies: Technology data for a single year
            - capacity: Capacity data for assets in the current and investment year
            - demand: Demand for the investment year
            - search_space: Search space for the assets
    """
    from muse.quantities import maximum_production

    # Load residential sector data
    residential = examples.sector("residential", model="medium")

    # Extract technologies and assets
    technologies = residential.technologies.squeeze("region").sel(year=INVESTMENT_YEAR)
    assets = next(a.assets for a in residential.agents)

    # Add nonzero minimum service factor data to allow calculation of the constraint
    technologies["minimum_service_factor"] = 0.1 * xr.ones_like(
        technologies.technology, dtype=float
    )

    # Calculate capacity
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

    # Create search space
    search_space = examples.search_space("residential", "medium")
    search_space = search_space.sel(asset=assets.technology.values)

    # Return dictionary of data
    return {
        "technologies": technologies,
        "capacity": capacity,
        "demand": market_demand,
        "search_space": search_space,
    }


@fixture(params=["timeslice_as_list", "timeslice_as_multindex"])
def constraints(request, model_data):
    """Default set of constraints for testing."""
    constraints = {
        "max_production": max_production(**model_data),
        "demand": demand(**model_data),
        "max_capacity_expansion": max_capacity_expansion(**model_data),
        "demand_limiting_capacity": demand_limiting_capacity(**model_data),
        "minimum_service": minimum_service(**model_data),
    }

    # Testing two different ways of handling timeslices
    if request.param == "timeslice_as_multindex":
        constraints = {key: _as_list(cs) for key, cs in constraints.items()}
    return constraints


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


def test_model_data(model_data):
    """Validating that the model data has appropriate dimensions."""
    assert set(model_data["technologies"].dims) == {"technology", "commodity"}
    assert set(model_data["search_space"].dims) == {"asset", "replacement"}
    assert set(model_data["capacity"].dims) == {"asset", "year"}
    assert set(model_data["demand"].dims) == {
        "asset",
        "commodity",
        "timeslice",
    }


def test_constraints_dimensions(constraints):
    """Test dimensions of all constraint matrices."""
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

    # Minimum service constraint
    assert set(constraints["minimum_service"].capacity.dims) == max_prod_dims
    assert set(constraints["minimum_service"].production.dims) == max_prod_dims
    assert set(constraints["minimum_service"].b.dims) == max_prod_dims


def test_max_capacity_expansion(constraints):
    """Checking basic properties of the max capacity expansion constraint."""
    max_capacity_expansion = constraints["max_capacity_expansion"]
    assert (max_capacity_expansion.capacity == 1).all()
    assert max_capacity_expansion.production == 0
    assert max_capacity_expansion.b.dims == ("replacement",)
    assert max_capacity_expansion.b.shape == (4,)
    assert (
        max_capacity_expansion.replacement
        == ["estove", "gasboiler", "gasstove", "heatpump"]
    ).all()


def test_max_production(constraints):
    """Checking basic properties of the max production constraint."""
    assert (constraints["max_production"].capacity <= 0).all()


def test_demand_limiting_capacity(constraints):
    """Checking basic properties of the demand limiting capacity constraint."""
    demand_limiting_capacity = constraints["demand_limiting_capacity"]
    max_production = constraints["max_production"]
    demand_constraint = constraints["demand"]

    # Test capacity values
    expected_capacity = (
        -max_production.capacity.max("timeslice")
        if "timeslice" in max_production.capacity.dims
        else -max_production.capacity
    )
    assert abs(demand_limiting_capacity.capacity - expected_capacity).sum() < 1e-5

    # Test production and b values
    assert demand_limiting_capacity.production == 0
    expected_b = (
        demand_constraint.b.max("timeslice")
        if "timeslice" in demand_constraint.b.dims
        else demand_constraint.b
    )
    assert abs(demand_limiting_capacity.b - expected_b).sum() < 1e-5


def test_max_capacity_expansion_no_limits(model_data):
    """Checking that the constraint is None when no limits are set."""
    technologies = model_data["technologies"].drop_vars(
        ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
    )
    assert (
        max_capacity_expansion(**{**model_data, "technologies": technologies}) is None
    )


def test_max_capacity_expansion_infinite_limits(model_data):
    """Checking that error is raised when infinite limits are set."""
    technologies = model_data["technologies"].copy()
    for limit in [
        "max_capacity_addition",
        "max_capacity_growth",
        "total_capacity_limit",
    ]:
        technologies[limit] = np.inf
    with raises(ValueError):
        max_capacity_expansion(**{**model_data, "technologies": technologies})


def test_max_capacity_expansion_seed(model_data):
    """Sanity checks for the seed parameter of the max capacity expansion constraint."""
    seed = 10
    technologies = model_data["technologies"].copy()
    technologies["growth_seed"] = seed

    # Test different capacity scenarios
    scenarios = [0, seed, 2 * seed]
    results = []
    for cap in scenarios:
        capacity = model_data["capacity"].copy()
        capacity.sel(year=2020)[:] = cap
        results.append(
            max_capacity_expansion(
                **{
                    **model_data,
                    "technologies": technologies,
                    "capacity": capacity,
                }
            )
        )

    # Zero capacity should match seed capacity
    assert results[0].b.values == approx(results[1].b.values)
    # Higher capacity should differ
    assert results[0].b.values != approx(results[2].b.values)


def test_no_minimum_service(model_data):
    """Checking that the constraint is None when minimum service factor is set to 0."""
    technologies = model_data["technologies"].copy()
    technologies["minimum_service_factor"] = 0
    assert minimum_service(**{**model_data, "technologies": technologies}) is None


@fixture
def lp_inputs(model_data):
    """Inputs to the lp adapter, in addition to the constraints."""
    # Make up capacity costs data
    shape = model_data["search_space"].shape
    costs = model_data["search_space"] * np.arange(np.prod(shape)).reshape(shape)

    # List of commodities
    commodities = list(model_data["demand"].commodity.values)

    return {
        "capacity_costs": costs,
        "commodities": commodities,
    }


@fixture
def lpcosts(lp_inputs):
    """Benchmark lpcosts dataset to test against."""
    from muse.lp_adapter import lp_costs

    return lp_costs(**lp_inputs)


def test_lp_constraints_matrix_b_is_scalar(lpcosts, constraints):
    """B is a scalar - output should be equivalent to a single row matrix."""
    constraint = constraints["max_production"]

    for attr in ["capacity", "production"]:
        lpconstraint = lp_constraint_matrix(
            xr.DataArray(1), getattr(constraint, attr), getattr(lpcosts, attr)
        )
        expected_value = -1 if attr == "capacity" else 1
        assert lpconstraint.values == approx(expected_value)
        assert set(lpconstraint.dims) == {
            f"d({x})" for x in getattr(lpcosts, attr).dims
        }


def test_max_production_constraint_diagonal(lpcosts, constraints):
    """Test production side of max capacity production is diagonal."""
    constraint = constraints["max_production"]

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


def test_lp_constraint(lpcosts, constraints):
    from muse.lp_adapter import lp_constraint

    constraint = constraints["max_production"]

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


def test_to_scipy_adapter_maxprod(lp_inputs, lpcosts, constraints):
    """Test scipy adapter with max production constraint."""
    adapter = ScipyAdapter.from_muse_data(
        **lp_inputs,
        constraints=[constraints["max_production"]],
    )

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


def test_to_scipy_adapter_demand(lp_inputs, lpcosts, constraints):
    """Test scipy adapter with demand constraint."""
    adapter = ScipyAdapter.from_muse_data(
        **lp_inputs,
        constraints=[constraints["demand"]],
    )

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


def test_to_scipy_adapter_max_capacity_expansion(lp_inputs, lpcosts, constraints):
    """Test scipy adapter with max capacity expansion constraint."""
    adapter = ScipyAdapter.from_muse_data(
        **lp_inputs,
        constraints=[constraints["max_capacity_expansion"]],
    )

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


def test_scipy_adapter_no_constraint(lp_inputs, lpcosts):
    adapter = ScipyAdapter.from_muse_data(
        **lp_inputs,
        constraints=[],
    )

    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert all(
        getattr(adapter, attr) is None for attr in ["A_ub", "b_ub", "A_eq", "b_eq"]
    )
    assert adapter.c.ndim == 1
    assert adapter.c.size == lpcosts.capacity.size + lpcosts.production.size


def test_back_to_muse_quantities(lpcosts):
    data = unified_dataset(lpcosts)

    # Test capacity
    lpquantity = selected_quantity(data, "capacity")
    assert set(lpquantity.dims) == {"d(asset)", "d(replacement)"}
    copy = back_to_muse_quantity(
        lpquantity.costs.values, xr.zeros_like(lpquantity.costs)
    )
    assert (copy == lpcosts.capacity).all()

    # Test production
    lpquantity = selected_quantity(data, "production")
    assert set(lpquantity.dims) == {
        "d(asset)",
        "d(replacement)",
        "d(timeslice)",
        "d(commodity)",
    }
    copy = back_to_muse_quantity(
        lpquantity.costs.values, xr.zeros_like(lpquantity.costs)
    )
    assert (copy == lpcosts.production).all()


def test_back_to_muse_all(lpcosts, lp_inputs):
    data = unified_dataset(lpcosts)
    lpcapacity = selected_quantity(data, "capacity")
    lpproduction = selected_quantity(data, "production")

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

    adapter = ScipyAdapter.from_muse_data(**lp_inputs, constraints=[])
    copy = adapter.to_muse(x)
    assert copy.capacity.size + copy.production.size == x.size
    assert (copy.capacity == lpcosts.capacity).all()
    assert (copy.production == lpcosts.production).all()


def test_scipy_adapter_standard_constraints(lp_inputs, constraints):
    adapter = ScipyAdapter.from_muse_data(
        **lp_inputs, constraints=list(constraints.values())
    )

    n_constraints = adapter.b_ub.size
    n_decision_vars = adapter.c.size
    maxprod_constraint = constraints["max_production"]

    assert (
        n_decision_vars
        == lp_inputs["capacity_costs"].size + maxprod_constraint.production.size
    )
    assert adapter.b_eq is None
    assert adapter.A_eq is None
    assert adapter.A_ub.shape == (n_constraints, n_decision_vars)
    assert n_constraints == sum(c.b.size for c in constraints.values())


def test_scipy_solver(model_data, lp_inputs, constraints):
    """Test the scipy solver for demand matching."""
    from muse.investments import scipy_match_demand

    solution = scipy_match_demand(
        costs=lp_inputs["capacity_costs"],
        commodities=lp_inputs["commodities"],
        search_space=model_data["search_space"],
        technologies=model_data["technologies"],
        constraints=list(constraints.values()),
    )
    assert isinstance(solution, xr.DataArray)
    assert set(solution.dims) == {"asset", "replacement"}
