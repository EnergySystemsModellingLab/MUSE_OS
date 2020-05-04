from pytest import fixture, approx

import numpy as np
import xarray as xr


@fixture
def model():
    return "medium"


@fixture
def residential(model):
    from muse import examples

    return examples.sector("residential", model=model)


@fixture
def timeslices(market):
    return market.timeslice


@fixture
def technologies(residential):
    return residential.technologies


@fixture
def market(model):
    from muse import examples

    return examples.residential_market(model)


@fixture
def search_space(model):
    from muse import examples

    return examples.search_space("residential", model)


@fixture
def costs(technologies, search_space, market):
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
def max_production(assets, search_space, market, technologies):
    from muse.constraints import max_production

    return max_production(assets, search_space, market, technologies)


@fixture
def constraint(max_production):
    return max_production


@fixture
def demand(assets, search_space, market, technologies):
    from muse.constraints import demand

    return demand(assets, search_space, market, technologies)


@fixture
def max_capacity_expansion(assets, search_space, market, technologies):
    from muse.constraints import max_capacity_expansion

    return max_capacity_expansion(assets, search_space, market, technologies)


@fixture
def constraints(assets, search_space, market, technologies):
    from muse import constraints as cs

    return [
        cs.max_production(assets, search_space, market, technologies),
        cs.demand(assets, search_space, market, technologies),
        cs.max_capacity_expansion(assets, search_space, market, technologies),
    ]


def test_lp_constraints_matrix_b_is_scalar(constraint, lpcosts):
    """b is a scalar.

    When ``b`` is a scalar, the output should be equivalent to a single row matrix, or a
    single vector with only decision variables.
    """
    from muse.constraints import lp_constraint_matrix

    lpconstraint = lp_constraint_matrix(
        xr.DataArray(1), constraint.capacity, lpcosts.capacity
    )
    assert lpconstraint.values == approx(1)
    assert set(lpconstraint.dims) == {f"d({x})" for x in lpcosts.capacity.dims}

    lpconstraint = lp_constraint_matrix(
        xr.DataArray(1), constraint.production, lpcosts.production
    )
    assert lpconstraint.values == approx(-1)
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
    assert stacked.values == approx(-np.eye(stacked.shape[0]))


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
    assert stacked.values == approx(-np.eye(stacked.shape[0]))

    assert set(result.b.dims) == constraint_dims
    assert result.b.values == approx(0)


def test_scipy_adapter_maxprod(technologies, costs, max_production, timeslices):
    from muse.constraints import scipy_adapter, lp_costs

    technologies = technologies.interp(year=2025)

    inputs = scipy_adapter(technologies, costs, timeslices, max_production)
    assert set(inputs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert inputs["bounds"] == (0, None)
    assert inputs["A_eq"] is None
    assert inputs["b_eq"] is None
    assert inputs["c"].ndim == 1
    assert inputs["b_ub"].ndim == 1
    assert inputs["A_ub"].ndim == 2
    assert inputs["b_ub"].size == inputs["A_ub"].shape[0]
    assert inputs["c"].size == inputs["A_ub"].shape[1]

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert inputs["c"].size == capsize + prodsize
    assert inputs["b_ub"].size == prodsize
    assert inputs["b_ub"] == approx(0)
    assert inputs["A_ub"][:, capsize:] == approx(-np.eye(prodsize))


def test_scipy_adapter_demand(technologies, costs, demand, timeslices):
    from muse.constraints import scipy_adapter, lp_costs

    technologies = technologies.interp(year=2025)

    inputs = scipy_adapter(technologies, costs, timeslices, demand)
    assert set(inputs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert inputs["bounds"] == (0, None)
    assert inputs["A_ub"] is None
    assert inputs["b_ub"] is None
    assert inputs["A_eq"] is not None
    assert inputs["b_eq"] is not None
    assert inputs["c"].ndim == 1
    assert inputs["b_eq"].ndim == 1
    assert inputs["A_eq"].ndim == 2
    assert inputs["b_eq"].size == inputs["A_eq"].shape[0]
    assert inputs["c"].size == inputs["A_eq"].shape[1]

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert inputs["c"].size == capsize + prodsize
    assert inputs["b_eq"].size == lpcosts.commodity.size * lpcosts.timeslice.size
    assert inputs["A_eq"][:, :capsize] == approx(0)
    assert inputs["A_eq"][:, capsize:].sum(axis=1) == approx(
        lpcosts.asset.size * lpcosts.replacement.size
    )
    assert set(inputs["A_eq"][:, capsize:].flatten()) == {0.0, 1.0}


def test_scipy_adapter_max_capacity_expansion(
    technologies, costs, max_capacity_expansion, timeslices
):
    from muse.constraints import scipy_adapter, lp_costs

    technologies = technologies.interp(year=2025)

    inputs = scipy_adapter(technologies, costs, timeslices, max_capacity_expansion)
    assert set(inputs) == {"c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds"}
    assert inputs["bounds"] == (0, None)
    assert inputs["A_ub"] is not None
    assert inputs["b_ub"] is not None
    assert inputs["A_eq"] is None
    assert inputs["b_eq"] is None
    assert inputs["c"].ndim == 1
    assert inputs["b_ub"].ndim == 1
    assert inputs["A_ub"].ndim == 2
    assert inputs["b_ub"].size == inputs["A_ub"].shape[0]
    assert inputs["c"].size == inputs["A_ub"].shape[1]
    assert inputs["c"].ndim == 1

    lpcosts = lp_costs(technologies, costs, timeslices)
    capsize = lpcosts.capacity.size
    prodsize = lpcosts.production.size
    assert inputs["c"].size == capsize + prodsize
    assert inputs["b_ub"].size == lpcosts.replacement.size
    assert inputs["A_ub"][:, capsize:] == approx(0)
    assert inputs["A_ub"][:, :capsize].sum(axis=1) == approx(lpcosts.asset.size)
    assert set(inputs["A_ub"][:, :capsize].flatten()) == {0.0, 1.0}
