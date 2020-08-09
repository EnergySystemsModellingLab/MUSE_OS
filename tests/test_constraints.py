from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, fixture, mark


@fixture
def model():
    return "medium"


@fixture
def residential(model):
    from muse import examples

    return examples.sector("residential", model=model)


@fixture(params=["timeslice_as_list", "timeslice_as_multindex"])
def timeslices(market, request):
    timeslice = market.timeslice
    if request.param == "timeslice_as_multindex":
        timeslice = _as_list(timeslice)
    return timeslice


@fixture
def technologies(residential):
    return residential.technologies.squeeze("region")


@fixture
def market(model):
    from muse import examples

    return examples.residential_market(model)


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
def lpcosts(technologies, market, costs):
    from muse.constraints import lp_costs

    return lp_costs(
        technologies.interp(year=market.year.min() + 5).drop_vars("year"),
        costs=costs,
        timeslices=market.timeslice,
    )


@fixture
def assets(residential):
    return next(a.assets for a in residential.agents if a.category == "retrofit")


@fixture
def market_demand(assets, technologies, market):
    from muse.quantities import maximum_production
    from muse.timeslices import convert_timeslice

    return 0.8 * maximum_production(
        technologies.interp(year=2025),
        convert_timeslice(
            assets.capacity.sel(year=2025).groupby("technology").sum("asset"), market
        ),
    ).rename(technology="asset")


@fixture
def max_production(market_demand, assets, search_space, market, technologies):
    from muse.constraints import max_production

    return max_production(market_demand, assets, search_space, market, technologies)


@fixture
def constraint(max_production):
    return max_production


@fixture
def demand_constraint(market_demand, assets, search_space, market, technologies):
    from muse.constraints import demand

    return demand(market_demand, assets, search_space, market, technologies)


@fixture
def max_capacity_expansion(market_demand, assets, search_space, market, technologies):
    from muse.constraints import max_capacity_expansion

    return max_capacity_expansion(
        market_demand, assets, search_space, market, technologies
    )


@fixture(params=["timeslice_as_list", "timeslice_as_multindex"])
def constraints(request, market_demand, assets, search_space, market, technologies):
    from muse import constraints as cs

    constraints = [
        cs.max_production(market_demand, assets, search_space, market, technologies),
        cs.demand(market_demand, assets, search_space, market, technologies),
        cs.max_capacity_expansion(
            market_demand, assets, search_space, market, technologies
        ),
    ]
    if request.param == "timeslice_as_multindex":
        constraints = [_as_list(cs) for cs in constraints]
    return constraints


def test_lp_constraints_matrix_b_is_scalar(constraint, lpcosts):
    """b is a scalar.

    When ``b`` is a scalar, the output should be equivalent to a single row matrix, or a
    single vector with only decision variables.
    """
    from muse.constraints import lp_constraint_matrix

    lpconstraint = lp_constraint_matrix(
        xr.DataArray(1), constraint.capacity, lpcosts.capacity
    )
    assert lpconstraint.values == approx(-1)
    assert set(lpconstraint.dims) == {f"d({x})" for x in lpcosts.capacity.dims}

    lpconstraint = lp_constraint_matrix(
        xr.DataArray(1), constraint.production, lpcosts.production
    )
    assert lpconstraint.values == approx(1)
    assert set(lpconstraint.dims) == {f"d({x})" for x in lpcosts.production.dims}


def test_max_production_constraint_diagonal(constraint, lpcosts):
    """production side of max capacity production is diagonal.

    The production for each timeslice, region, asset, and replacement technology should
    not outstrip the assigned for the asset and replacement technology. Hence, the
    production side of the constraint is the identity with a -1 factor. The capacity
    side is diagonal, but the values reflect the max-production for each timeslices,
    commodity and technology.
    """
    from muse.constraints import lp_constraint_matrix

    result = lp_constraint_matrix(constraint.b, constraint.capacity, lpcosts.capacity)
    decision_dims = {f"d({x})" for x in lpcosts.capacity.dims}
    constraint_dims = {
        f"c({x})" for x in set(lpcosts.production.dims).union(constraint.b.dims)
    }
    assert set(result.dims) == decision_dims.union(constraint_dims)

    result = lp_constraint_matrix(
        constraint.b, constraint.production, lpcosts.production
    )
    decision_dims = {f"d({x})" for x in lpcosts.production.dims}
    assert set(result.dims) == decision_dims.union(constraint_dims)
    stacked = result.stack(d=sorted(decision_dims), c=sorted(constraint_dims))
    assert stacked.shape[0] == stacked.shape[1]
    assert stacked.values == approx(np.eye(stacked.shape[0]))


def test_lp_constraint(constraint, lpcosts):
    from muse.constraints import lp_constraint

    result = lp_constraint(constraint, lpcosts)
    decision_dims = {f"d({x})" for x in lpcosts.capacity.dims}
    constraint_dims = {
        f"c({x})" for x in set(lpcosts.production.dims).union(constraint.b.dims)
    }
    assert set(result.capacity.dims) == decision_dims.union(constraint_dims)

    decision_dims = {f"d({x})" for x in lpcosts.production.dims}
    assert set(result.production.dims) == decision_dims.union(constraint_dims)
    stacked = result.production.stack(
        d=sorted(decision_dims), c=sorted(constraint_dims)
    )
    assert stacked.shape[0] == stacked.shape[1]
    assert stacked.values == approx(np.eye(stacked.shape[0]))

    assert set(result.b.dims) == constraint_dims
    assert result.b.values == approx(0)


def test_to_scipy_adapter_maxprod(technologies, costs, max_production, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    adapter = ScipyAdapter.factory(technologies, costs, timeslices, max_production)
    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1
    assert adapter.b_ub.ndim == 1
    assert adapter.A_ub.ndim == 2
    assert adapter.b_ub.size == adapter.A_ub.shape[0]
    assert adapter.c.size == adapter.A_ub.shape[1]

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert adapter.c.size == capsize + prodsize
    assert adapter.b_ub.size == prodsize
    assert adapter.b_ub == approx(0)
    assert adapter.A_ub[:, capsize:] == approx(np.eye(prodsize))


def test_to_scipy_adapter_demand(technologies, costs, demand_constraint, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    adapter = ScipyAdapter.factory(technologies, costs, timeslices, demand_constraint)
    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_ub is not None
    assert adapter.b_ub is not None
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1
    assert adapter.b_ub.ndim == 1
    assert adapter.A_ub.ndim == 2
    assert adapter.b_ub.size == adapter.A_ub.shape[0]
    assert adapter.c.size == adapter.A_ub.shape[1]

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert adapter.c.size == capsize + prodsize
    assert (
        adapter.b_ub.size
        == lpcosts.commodity.size * lpcosts.timeslice.size * lpcosts.asset.size
    )
    assert adapter.A_ub[:, :capsize] == approx(0)
    assert adapter.A_ub[:, capsize:].sum(axis=1) == approx(-lpcosts.replacement.size)
    assert set(adapter.A_ub[:, capsize:].flatten()) == {0.0, -1.0}


def test_to_scipy_adapter_max_capacity_expansion(
    technologies, costs, max_capacity_expansion, timeslices
):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    adapter = ScipyAdapter.factory(
        technologies, costs, timeslices, max_capacity_expansion
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
    assert adapter.b_ub.size == adapter.A_ub.shape[0]
    assert adapter.c.size == adapter.A_ub.shape[1]
    assert adapter.c.ndim == 1

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert adapter.c.size == capsize + prodsize
    assert adapter.b_ub.size == lpcosts.replacement.size
    assert adapter.A_ub[:, capsize:] == approx(0)
    assert adapter.A_ub[:, :capsize].sum(axis=1) == approx(lpcosts.asset.size)
    assert set(adapter.A_ub[:, :capsize].flatten()) == {0.0, 1.0}


def test_to_scipy_adapter_no_constraint(technologies, costs, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    adapter = ScipyAdapter.factory(technologies, costs, timeslices)
    assert set(adapter.kwargs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert adapter.bounds == (0, np.inf)
    assert adapter.A_ub is None
    assert adapter.b_ub is None
    assert adapter.A_eq is None
    assert adapter.b_eq is None
    assert adapter.c.ndim == 1

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert adapter.c.size == capsize + prodsize


@mark.parametrize("quantity", ["capacity", "production"])
def test_back_to_muse_quantity(quantity, technologies, costs, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    lpcosts = lp_costs(technologies, costs, timeslices)
    data = ScipyAdapter._unified_dataset(technologies, lpcosts)
    lpquantity = ScipyAdapter._stacked_quantity(data, quantity)
    assert set(lpquantity.dims) == {"decision"}

    decision = lpquantity.get_index("decision")
    assert len(set(decision)) == len(decision)
    lpquantity.costs[:] = range(lpquantity.costs.size)
    assignment = {k: lpquantity.isel(decision=i) for i, k in enumerate(decision)}

    copy = ScipyAdapter._back_to_muse_quantity(
        lpquantity.costs.values, xr.zeros_like(lpquantity.costs)
    )
    assert copy.size == lpquantity.costs.size
    assert copy.size == len(assignment)

    for coordinates, expected in assignment.items():
        location = {k[2:-1]: c for k, c in zip(decision.names, coordinates)}
        assert copy.sel(location) == expected


def test_back_to_muse(technologies, costs, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    lpcosts = lp_costs(technologies, costs, timeslices)
    data = ScipyAdapter._unified_dataset(technologies, lpcosts)

    lpcapacity = ScipyAdapter._stacked_quantity(data, "capacity")
    assert set(lpcapacity.dims) == {"decision"}
    decision = lpcapacity.get_index("decision")
    assert len(set(decision)) == len(decision)
    lpcapacity.costs[:] = range(lpcapacity.costs.size)

    lpproduction = ScipyAdapter._stacked_quantity(data, "production")
    assert set(lpproduction.dims) == {"decision"}
    decision = lpproduction.get_index("decision")
    assert len(set(decision)) == len(decision)
    lpproduction.costs[:] = range(lpproduction.costs.size)

    x = np.concatenate((lpcapacity.costs.values, lpproduction.costs.values))

    copy = ScipyAdapter._back_to_muse(
        x, xr.zeros_like(lpcapacity.costs), xr.zeros_like(lpproduction.costs)
    )
    assert copy.capacity.size + copy.production.size == x.size

    decision = lpcapacity.get_index("decision")
    assignment = {k: lpcapacity.isel(decision=i) for i, k in enumerate(decision)}
    for coordinates, expected in assignment.items():
        location = {k[2:-1]: c for k, c in zip(decision.names, coordinates)}
        assert copy.capacity.sel(location) == expected

    decision = lpproduction.get_index("decision")
    assignment = {k: lpproduction.isel(decision=i) for i, k in enumerate(decision)}
    for coordinates, expected in assignment.items():
        location = {k[2:-1]: c for k, c in zip(decision.names, coordinates)}
        assert copy.production.sel(location) == expected


def test_scipy_adapter_back_to_muse(technologies, costs, timeslices):
    from muse.constraints import ScipyAdapter, lp_costs

    technologies = technologies.interp(year=2025)

    lpcosts = lp_costs(technologies, costs, timeslices)
    data = ScipyAdapter._unified_dataset(technologies, lpcosts)

    lpcapacity = ScipyAdapter._stacked_quantity(data, "capacity")
    assert set(lpcapacity.dims) == {"decision"}
    decision = lpcapacity.get_index("decision")
    assert len(set(decision)) == len(decision)
    lpcapacity.costs[:] = range(lpcapacity.costs.size)

    lpproduction = ScipyAdapter._stacked_quantity(data, "production")
    assert set(lpproduction.dims) == {"decision"}
    decision = lpproduction.get_index("decision")
    assert len(set(decision)) == len(decision)
    lpproduction.costs[:] = range(lpproduction.costs.size)

    x = np.concatenate((lpcapacity.costs.values, lpproduction.costs.values))

    copy = ScipyAdapter._back_to_muse(
        x, xr.zeros_like(lpcapacity.costs), xr.zeros_like(lpproduction.costs)
    )

    adapter = ScipyAdapter.factory(technologies, costs, timeslices)
    assert (adapter.to_muse(x).capacity == copy.capacity).all()
    assert (adapter.to_muse(x).production == copy.production).all()


def _as_list(data: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
    if "timeslice" in data.dims:
        data = data.copy(deep=False)
        data["timeslice"] = pd.MultiIndex.from_tuples(
            data.get_index("timeslice"), names=("month", "day", "hour")
        )
    return data


def test_scipy_adapter_standard_constraints(
    technologies, costs, constraints, timeslices
):
    from muse.constraints import ScipyAdapter

    technologies = technologies.interp(year=2025)

    adapter = ScipyAdapter.factory(technologies, costs, timeslices, *constraints)
    maxprod = next(cs for cs in constraints if cs.name == "max_production")
    maxcapa = next(cs for cs in constraints if cs.name == "max capacity expansion")
    demand = next(cs for cs in constraints if cs.name == "demand")
    assert adapter.c.size == costs.size + maxprod.production.size
    assert adapter.b_eq is None
    assert adapter.A_eq is None
    assert adapter.A_ub.shape == (adapter.b_ub.size, adapter.c.size)
    assert adapter.b_ub.size == demand.b.size + maxprod.b.size + maxcapa.b.size


def test_scipy_solver(technologies, costs, constraints):
    from muse.investments import scipy_match_demand

    solution = scipy_match_demand(
        costs=costs,
        search_space=xr.ones_like(costs),
        technologies=technologies,
        constraints=constraints,
        year=2025,
    )
    assert isinstance(solution, xr.DataArray)
    assert set(solution.dims) == {"asset", "replacement"}


def test_minimum_service(
    market_demand, assets, search_space, market, technologies, costs, constraints
):
    from muse.constraints import minimum_service
    from muse.investments import scipy_match_demand

    minimum_service_constraint = minimum_service(
        market_demand, assets, search_space, market, technologies
    )

    # test it is none (when appropriate)
    assert minimum_service_constraint is None

    # use this constraint (and others) to find a solution
    solution = scipy_match_demand(
        costs=costs,
        search_space=search_space,
        technologies=technologies,
        constraints=constraints,
        year=2025,
    )

    # add the column to technologies
    minimum_service_factor = 0.4 * xr.ones_like(technologies.technology, dtype=float)
    technologies["minimum_service_factor"] = minimum_service_factor

    # append minimum_service_constraint to constraints
    minimum_service_constraint = minimum_service(
        market_demand, assets, search_space, market, technologies
    )
    constraints.append(minimum_service_constraint)

    # test that it is no longer none
    assert isinstance(minimum_service_constraint, xr.Dataset)

    # test solution using new constraint is different from first solution
    minserv_solution = scipy_match_demand(
        costs=costs,
        search_space=search_space,
        technologies=technologies,
        constraints=constraints,
        year=2025,
    )

    assert np.allclose(minserv_solution, solution) is False
