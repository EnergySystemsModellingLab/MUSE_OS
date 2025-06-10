from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, fixture, raises

from muse.timeslices import drop_timeslice
from muse.utilities import interpolate_capacity, reduce_assets

CURRENT_YEAR = 2020
INVESTMENT_YEAR = 2025


@fixture
def model():
    return "medium"


@fixture
def residential(model):
    from muse import examples

    return examples.sector("residential", model=model)


@fixture
def technologies(residential):
    return residential.technologies.squeeze("region").sel(year=INVESTMENT_YEAR)


@fixture
def search_space(model, assets):
    from muse import examples

    space = examples.search_space("residential", model)
    return space.sel(asset=assets.technology.values)


@fixture
def costs(search_space):
    shape = search_space.shape
    return search_space * np.arange(np.prod(shape)).reshape(shape)


@fixture
def assets(residential):
    return next(a.assets for a in residential.agents)


@fixture
def capacity(assets):
    return interpolate_capacity(
        reduce_assets(assets.capacity, coords=("technology", "region")),
        year=[CURRENT_YEAR, INVESTMENT_YEAR],
    )


@fixture
def market_demand(assets, technologies):
    from muse.quantities import maximum_production
    from muse.utilities import broadcast_over_assets

    # Set demand just below the maximum production of existing assets
    res = 0.8 * maximum_production(
        broadcast_over_assets(technologies, assets),
        assets.capacity,
    ).sel(year=INVESTMENT_YEAR).groupby("technology").sum("asset").rename(
        technology="asset"
    )

    # Remove un-demanded commodities
    res = res.sel(commodity=(res > 0).any(dim=["timeslice", "asset"]))
    return res


@fixture
def commodities(market_demand):
    return list(market_demand.commodity.values)


def test_fixtures(technologies, search_space, costs, assets, capacity, market_demand):
    assert set(technologies.dims) == {"technology", "commodity"}
    assert set(search_space.dims) == {"asset", "replacement"}
    assert set(costs.dims) == {"asset", "replacement"}
    assert set(assets.dims) == {"asset", "year"}
    assert set(capacity.dims) == {"asset", "year"}
    assert set(market_demand.dims) == {"asset", "commodity", "timeslice"}


@fixture
def lpcosts(costs, commodities):
    from muse.constraints import lp_costs

    return lp_costs(costs, commodities=commodities)


@fixture
def max_production(market_demand, capacity, search_space, technologies):
    from muse.constraints import max_production

    return max_production(market_demand, capacity, search_space, technologies)


@fixture
def demand_constraint(market_demand, capacity, search_space, technologies):
    from muse.constraints import demand

    return demand(market_demand, capacity, search_space, technologies)


@fixture
def max_capacity_expansion(market_demand, capacity, search_space, technologies):
    from muse.constraints import max_capacity_expansion

    return max_capacity_expansion(market_demand, capacity, search_space, technologies)


@fixture
def demand_limiting_capacity(market_demand, capacity, search_space, technologies):
    from muse.constraints import demand_limiting_capacity

    return demand_limiting_capacity(market_demand, capacity, search_space, technologies)


@fixture
def constraint(max_production):
    return max_production


@fixture(params=["timeslice_as_list", "timeslice_as_multindex"])
def constraints(
    request,
    max_production,
    demand_constraint,
    demand_limiting_capacity,
    max_capacity_expansion,
):
    constraints = [
        max_production,
        demand_limiting_capacity,
        demand_constraint,
        max_capacity_expansion,
    ]
    if request.param == "timeslice_as_multindex":
        constraints = [_as_list(cs) for cs in constraints]
    return constraints


def test_constraints_dimensions(
    max_production, demand_constraint, demand_limiting_capacity, max_capacity_expansion
):
    # Max production constraint
    max_prod_dims = {"asset", "commodity", "replacement", "timeslice"}
    assert set(max_production.capacity.dims) == max_prod_dims
    assert set(max_production.production.dims) == max_prod_dims
    assert set(max_production.b.dims) == max_prod_dims

    # Demand constraint
    assert set(demand_constraint.capacity.dims) == set()
    assert set(demand_constraint.production.dims) == set()
    assert set(demand_constraint.b.dims) == {"asset", "commodity", "timeslice"}

    # Demand limiting capacity constraint
    assert set(demand_limiting_capacity.capacity.dims) == {
        "asset",
        "commodity",
        "replacement",
    }
    assert set(demand_limiting_capacity.production.dims) == set()
    assert set(demand_limiting_capacity.b.dims) == {"asset", "commodity"}

    # Max capacity expansion constraint
    assert set(max_capacity_expansion.capacity.dims) == set()
    assert set(max_capacity_expansion.production.dims) == set()
    assert set(max_capacity_expansion.b.dims) == {"replacement"}


def test_lp_constraints_matrix_b_is_scalar(constraint, lpcosts):
    """B is a scalar - output should be equivalent to a single row matrix."""
    from muse.constraints import lp_constraint_matrix

    for attr in ["capacity", "production"]:
        lpconstraint = lp_constraint_matrix(
            xr.DataArray(1), getattr(constraint, attr), getattr(lpcosts, attr)
        )
        expected_value = -1 if attr == "capacity" else 1
        assert lpconstraint.values == approx(expected_value)
        assert set(lpconstraint.dims) == {
            f"d({x})" for x in getattr(lpcosts, attr).dims
        }


def test_max_production_constraint_diagonal(constraint, lpcosts):
    """Test production side of max capacity production is diagonal."""
    from muse.constraints import lp_constraint_matrix

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


def test_lp_constraint(constraint, lpcosts):
    from muse.constraints import lp_constraint

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


def test_to_scipy_adapter_maxprod(costs, max_production, commodities, lpcosts):
    """Test scipy adapter with max production constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        costs, constraints=[max_production], commodities=commodities
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


def test_to_scipy_adapter_demand(costs, demand_constraint, commodities, lpcosts):
    """Test scipy adapter with demand constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        costs, constraints=[demand_constraint], commodities=commodities
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


def test_to_scipy_adapter_max_capacity_expansion(
    costs, max_capacity_expansion, commodities, lpcosts
):
    """Test scipy adapter with max capacity expansion constraint."""
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(
        costs, constraints=[max_capacity_expansion], commodities=commodities
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


def test_scipy_adapter_no_constraint(costs, commodities, lpcosts):
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(costs, constraints=[], commodities=commodities)
    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert all(
        getattr(adapter, attr) is None for attr in ["A_ub", "b_ub", "A_eq", "b_eq"]
    )
    assert adapter.c.ndim == 1
    assert adapter.c.size == lpcosts.capacity.size + lpcosts.production.size


def test_back_to_muse_quantities(lpcosts):
    from muse.constraints import ScipyAdapter

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


def test_back_to_muse_all(lpcosts):
    from muse.constraints import ScipyAdapter

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


def test_scipy_adapter_back_to_muse(costs, constraints, commodities, lpcosts):
    """Test converting back from scipy adapter format to MUSE format."""
    from muse.constraints import ScipyAdapter

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

    adapter = ScipyAdapter.factory(costs, constraints, commodities=commodities)
    assert (adapter.to_muse(x).capacity == lpcosts.capacity).all()
    assert (adapter.to_muse(x).production == lpcosts.production).all()


def _as_list(data: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    if "timeslice" in data.dims:
        data = data.copy(deep=False)
        index = pd.MultiIndex.from_tuples(
            data.get_index("timeslice"), names=("month", "day", "hour")
        )
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, "timeslice")
        data = drop_timeslice(data).assign_coords(mindex_coords)
    return data


def test_scipy_adapter_standard_constraints(costs, constraints, commodities):
    from muse.constraints import ScipyAdapter

    adapter = ScipyAdapter.factory(costs, constraints, commodities=commodities)
    constraint_map = {cs.name: cs for cs in constraints}
    maxprod = constraint_map["max_production"]
    maxcapa = constraint_map["max capacity expansion"]
    demand = constraint_map["demand"]
    dlc = constraint_map["demand_limiting_capacity"]

    n_constraints = adapter.b_ub.size
    n_decision_vars = adapter.c.size

    assert n_decision_vars == costs.size + maxprod.production.size
    assert adapter.b_eq is None
    assert adapter.A_eq is None
    assert adapter.A_ub.shape == (n_constraints, n_decision_vars)
    assert n_constraints == sum(c.b.size for c in [demand, maxprod, maxcapa, dlc])


def test_scipy_solver(technologies, costs, constraints, commodities):
    """Test the scipy solver for demand matching."""
    from muse.investments import scipy_match_demand

    solution = scipy_match_demand(
        costs=costs,
        search_space=xr.ones_like(costs),
        technologies=technologies,
        constraints=constraints,
        commodities=commodities,
    )
    assert isinstance(solution, xr.DataArray)
    assert set(solution.dims) == {"asset", "replacement"}


def test_minimum_service(
    market_demand, capacity, search_space, technologies, constraints
):
    from muse.constraints import minimum_service

    # Test with no minimum service factor
    assert minimum_service(market_demand, capacity, search_space, technologies) is None

    # Test with minimum service factor
    technologies["minimum_service_factor"] = 0.4 * xr.ones_like(
        technologies.technology, dtype=float
    )
    min_service = minimum_service(market_demand, capacity, search_space, technologies)
    constraints.append(min_service)
    assert isinstance(min_service, xr.Dataset)


def test_max_capacity_expansion_properties(max_capacity_expansion):
    assert (max_capacity_expansion.capacity == 1).all()
    assert max_capacity_expansion.production == 0
    assert max_capacity_expansion.b.dims == ("replacement",)
    assert max_capacity_expansion.b.shape == (4,)
    assert (
        max_capacity_expansion.replacement
        == ["estove", "gasboiler", "gasstove", "heatpump"]
    ).all()


def test_max_capacity_expansion_no_limits(
    market_demand, capacity, search_space, technologies
):
    from muse.constraints import max_capacity_expansion

    techs = technologies.drop_vars(
        ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
    )
    assert max_capacity_expansion(market_demand, capacity, search_space, techs) is None


def test_max_capacity_expansion_seed(
    market_demand, capacity, search_space, technologies
):
    from muse.constraints import max_capacity_expansion

    seed = 10
    technologies["growth_seed"] = seed

    # Test different capacity scenarios
    scenarios = [0, seed, 2 * seed]
    results = []
    for cap in scenarios:
        capacity.sel(year=2020)[:] = cap
        results.append(
            max_capacity_expansion(market_demand, capacity, search_space, technologies)
        )

    # Zero capacity should match seed capacity
    assert results[0].b.values == approx(results[1].b.values)
    # Higher capacity should differ
    assert results[0].b.values != approx(results[2].b.values)


def test_max_capacity_expansion_infinite_limits(
    market_demand, capacity, search_space, technologies
):
    from muse.constraints import max_capacity_expansion

    for limit in [
        "max_capacity_addition",
        "max_capacity_growth",
        "total_capacity_limit",
    ]:
        technologies[limit] = np.inf
    with raises(ValueError):
        max_capacity_expansion(market_demand, capacity, search_space, technologies)


def test_max_production(max_production):
    assert (max_production.capacity <= 0).all()


def test_demand_limiting_capacity(
    demand_limiting_capacity, max_production, demand_constraint
):
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
