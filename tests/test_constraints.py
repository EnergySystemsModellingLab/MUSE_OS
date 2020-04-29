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
def lpcosts(technologies, search_space, market):
    from muse.constraints import lp_costs

    shape = search_space.shape
    return lp_costs(
        technologies.interp(year=market.year.min() + 5).drop_vars("year"),
        costs=search_space * np.arange(np.prod(shape)).reshape(shape),
        timeslices=market.timeslice,
    )


@fixture
def assets(residential):
    return next(a.assets for a in residential.agents if a.category == "retrofit")


@fixture
def constraint(assets, search_space, market, technologies):
    from muse.constraints import max_production as cons

    return cons(assets, search_space, market, technologies)


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
