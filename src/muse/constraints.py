"""Investment constraints.

Constraints on investements ensure that investements match some given criteria. For
instance, the constraints could ensure that only so much of a new asset can be built
every year.

Functions to compute constraints should be registered via the decorator
:py:meth:`~muse.constraints.register_constraints`. This registration step makes it
possible for constraints to be declared in the TOML file.

Generally, LP solvers accept linear constraint defined as:

.. math::

    A x \\leq b

with :math:`A` a matrix, :math:`x` the decision variables, and :math:`b` a vector.
However, these quantities are dimensionless. They do no have timeslices, assets, or
replacement technologies, or any other dimensions that users have set-up in their model.
The crux is to translates from MUSE's data-structures to a consistent dimensionless
format.

In MUSE, users can register constraints functions that return fully dimensional
quantities. The matrix operator is split over the capacity decision variables and the
production decision variables:

.. math::

    A_c .* x_c + A_p .* x_p \\leq b

The operator :math:`.*` means the standard elementwise multiplication of xarray,
including automatic broadcasting (adding missing dimensions by repeating the smaller
matrix along the missing dimension).  Constraint functions return the three quantities
:math:`A_c`, :math:`A_p`, and :math:`b`. These three quantities will often not have the
same dimension. E.g. one might include timeslices where another might not. The
transformation from :math:`A_c`, :math:`A_p`, :math:`b` to :math:`A` and :math:`b`
happens as described below.

- :math:`b` remains the same. It defines the rows of :math:`A`.
- :math:`x_c` and :math:`x_p` are concatenated one on top of the other and define the
  columns of :math:`A`.
- :math:`A` is split into a left submatrix for capacities and a right submatrix for
  production, following the concatenation of :math:`x_c` and :math:`x_p`
- Any dimension in :math:`A_c .* x_c` (:math:`A_p .* x_p`) that is also in :math:`b`
  defines diagonal entries into the left (right) submatrix of :math:`A`.
- Any dimension in :math:`A_c .* x_c` (:math:`A_p .* x_b`) and missing from
  :math:`b` is reduce by summation over a row in the left (right) submatrix of
  :math:`A`. In other words, those dimension do become part of a standard tensor
  reduction or matrix multiplication.

There are two additional rules. However, they are likely to be the result of an
inefficient defininition of :math:`A_c`, :math:`A_p` and :math:`b`.

- Any dimension in :math:`A_c` (:math:`A_b`) that is neither in :math:`b` nor in
  :math:`x_c` (:math:`x_p`) is reduced by summation before consideration for the
  elementwise multiplication. For instance, if :math:`d` is such a dimension, present
  only in :math:`A_c`, then the problem becomes :math:`(\\sum_d A_c) .* x_c + A_p .* x_p
  \\leq b`.
- Any dimension missing from :math:`A_c .* x_c` (:math:`A_p .* x_p`) and present in
  :math:`b` is added by repeating the resulting row in :math:`A`.

Constraints are registered using the decorator
:py:meth:`~muse.constraints.register_constraints`. The decorated functions must follow
the following signature:

.. code-block:: python

    @register_constraints
    def constraints(
        demand: xr.DataArray,
        assets: xr.Dataset,
        search_space: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset,
        year: Optional[int] = None,
        **kwargs,
    ) -> Constraint:
        pass

demand:
    The demand for the sectors products. In practice it is a demand share obtained in
    :py:mod:`~muse.demand_share`. It is a data-array with dimensions including `asset`,
    `commodity`, `timeslice`.
assets:
    The capacity of the assets owned by the agent.
search_space:
    A matrix `asset` vs `replacement` technology defining which replacement technologies
    will be considered for each existing asset.
market:
    The market as obtained from the MCA.
technologies:
    Technodata characterizing the competing technologies.
year:
    current year.
``**kwargs``:
    Any other parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Text,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator

CAPACITY_DIMS = "asset", "replacement", "region"
"""Default dimensions for capacity decision variables."""
PRODUCT_DIMS = "commodity", "timeslice", "region"
"""Default dimensions for product decision variables."""


class ConstraintKind(Enum):
    EQUALITY = auto()
    UPPER_BOUND = auto()
    LOWER_BOUND = auto()


Constraint = xr.Dataset
"""An investment constraint :math:`A * x ~ b`

Where :math:`~` is one of :math:`=,\\leq,\\geq`.

A constraint should contain a data-array `b` corresponding to right-hand-side vector
of the contraint. It should also contain a data-array `capacity` corresponding to the
left-hand-side matrix operator which will be applied to the capacity-related decision
variables.  It should contain a similar matrix `production` corresponding to
the left-hand-side matrix operator which will be applied to the production-related
decision variables. Should any of these three objects be missing, they default to the
scalar 0. Finally, the constraint should contain an attribute `kind` of type
:py:class:`ConstraintKind` defining the operation. If it is missing, it defaults to an
upper bound constraint.
"""


CONSTRAINT_SIGNATURE = Callable[
    [xr.DataArray, xr.Dataset, xr.DataArray, xr.Dataset, xr.Dataset, KwArg()],
    Optional[Constraint],
]
"""Basic signature for functions producing constraints.

.. note::

    A constraint can return `None`, in which case it is ignored. This makes it simple to
    add cosntraints that are only used if some condition is met, e.g. minimum service
    conditions are defined in the technodata.
"""
CONSTRAINTS: MutableMapping[Text, CONSTRAINT_SIGNATURE] = {}
"""Registry of constraint functions."""


@registrator(registry=CONSTRAINTS)
def register_constraints(function: CONSTRAINT_SIGNATURE) -> CONSTRAINT_SIGNATURE:
    """Registers a constraint with MUSE.

    See :py:mod:`muse.constraints`.
    """
    from functools import wraps

    @wraps(function)
    def decorated(
        demand: xr.DataArray,
        assets: xr.Dataset,
        search_space: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset,
        **kwargs,
    ) -> Optional[Constraint]:
        """Computes and standardizes a constraint."""
        constraint = function(  # type: ignore
            demand, assets, search_space, market, technologies, **kwargs
        )
        if constraint is not None:
            if "kind" not in constraint.attrs:
                constraint.attrs["kind"] = ConstraintKind.UPPER_BOUND
            if (
                "capacity" not in constraint.data_vars
                and "production" not in constraint.data_vars
            ):
                raise RuntimeError("Invalid constraint format")
            if "capacity" not in constraint.data_vars:
                constraint["capacity"] = 0
            if "production" not in constraint.data_vars:
                constraint["production"] = 0
            if "b" not in constraint.data_vars:
                constraint["b"] = 0
            if "name" not in constraint.data_vars and "name" not in constraint.attrs:
                constraint.attrs["name"] = function.__name__

            # ensure that the constraint and the search space match
            dims = [d for d in constraint.dims if d in search_space.dims]
            constraint = constraint.sel({k: search_space[k] for k in dims})

        return constraint

    return decorated


def factory(
    settings: Optional[
        Union[Text, Mapping, Sequence[Text], Sequence[Union[Text, Mapping]]]
    ] = None
) -> Callable:
    """Creates a list of constraints from standard settings.

    The standard settings can be a string naming the constraint, a dictionary including
    at least "name", or a list of strings and dictionaries.
    """
    from functools import partial

    if settings is None:
        settings = (
            "max_production",
            "max_capacity_expansion",
            "demand",
            "search_space",
            "minimum_service",
        )

    def normalize(x) -> MutableMapping:
        return dict(name=x) if isinstance(x, Text) else x

    if isinstance(settings, (Text, Mapping)):
        settings = cast(Union[Sequence[Text], Sequence[Mapping]], [settings])
    parameters = [normalize(x) for x in settings]
    names = [x.pop("name") for x in parameters]

    constraint_closures = [
        partial(CONSTRAINTS[name], **param) for name, param in zip(names, parameters)
    ]

    def constraints(
        demand: xr.DataArray,
        assets: xr.Dataset,
        search_space: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset,
        year: Optional[int] = None,
    ) -> List[Constraint]:
        if year is None:
            year = int(market.year.min())
        constraints = [
            function(demand, assets, search_space, market, technologies, year=year)
            for function in constraint_closures
        ]
        return [constraint for constraint in constraints if constraint is not None]

    return constraints


@register_constraints
def max_capacity_expansion(
    demand: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: Optional[int] = None,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    r"""Max-capacity addition, max-capacity growth, and capacity limits constraints.

    Limits by how much the capacity of each technology owned by an agent can grow in
    a given year. This is a constraint on the agent's ability to invest in a
    technology.

    Let :math:`L_t^r(y)` be the total capacity limit for a given year, technology,
    and region. :math:`G_t^r(y)` is the maximum growth. And :math:`W_t^r(y)` is
    the maximum additional capacity. :math:`y=y_0` is the current year and
    :math:`y=y_1` is the year marking the end of the investment period.

    Let :math:`\mathcal{A}^{i, r}_{t, \iota}(y)` be the current assets, before
    invesment, and let :math:`\Delta\mathcal{A}^{i,r}_t` be the future investements.
    The the constraint on agent :math:`i` are given as:

    .. math::

        L_t^r(y_0) - \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_1)
            \geq \Delta\mathcal{A}^{i,r}_t

        (y_1 - y_0 + 1) G_t^r(y_0) \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_0)
            - \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_1)
            \geq \Delta\mathcal{A}^{i,r}_t

        (y_1 - y_0)W_t^r(y_0) \geq  \Delta\mathcal{A}^{i,r}_t

    The three constraints are combined into a single one which is returned as the
    maximum capacity expansion, :math:`\Gamma_t^{r, i}`. The maximum capacity
    expansion cannot impose negative investments:
    Maximum capacity addition:

        .. math::

            \Gamma_t^{r, i} \geq 0
    """
    from muse.utilities import filter_input

    if year is None:
        year = int(market.year.min())
    forecast_year = forecast + year

    kwargs = dict(technology=search_space.replacement, year=year)
    if "region" in assets and "region" in technologies.dims:
        kwargs["region"] = assets.region
    techs = filter_input(
        technologies[
            ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
        ],
        **kwargs,
    )
    assert isinstance(techs, xr.Dataset)

    capacity = (
        assets.capacity.groupby("technology")
        .sum("asset")
        .interp(year=[year, forecast_year], method=interpolation)
        .rename(technology=search_space.replacement.name)
        .reindex_like(search_space.replacement, fill_value=0)
    )

    add_cap = techs.max_capacity_addition * forecast

    limit = techs.total_capacity_limit
    forecasted = capacity.sel(year=forecast_year, drop=True)
    total_cap = (limit - forecasted).clip(min=0).rename("total_cap")

    max_growth = techs.max_capacity_growth
    initial = capacity.sel(year=year, drop=True)
    growth_cap = initial * (max_growth * forecast + 1) - forecasted

    zero_cap = add_cap.where(add_cap < total_cap, total_cap)
    with_growth = zero_cap.where(zero_cap < growth_cap, growth_cap)
    constraint = with_growth.where(initial > 0, zero_cap)
    return xr.Dataset(
        dict(b=constraint, capacity=1),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND, name="max capacity expansion"),
    )


@register_constraints
def demand(
    demand: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: Optional[int] = None,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    """Constraints production to meet demand."""
    from muse.commodities import is_enduse

    enduse = technologies.commodity.sel(commodity=is_enduse(technologies.comm_usage))
    b = demand.sel(commodity=demand.commodity.isin(enduse))
    assert "year" not in b.dims
    return xr.Dataset(
        dict(b=b, production=1), attrs=dict(kind=ConstraintKind.LOWER_BOUND)
    )


@register_constraints
def search_space(
    demand: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: Optional[int] = None,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Optional[Constraint]:
    """Removes disabled technologies."""
    b = ~search_space.astype(bool)
    b = b.sel(asset=b.any("replacement"))
    if b.size == 0:
        return None
    return xr.Dataset(dict(b=b, capacity=1), attrs=dict(kind=ConstraintKind.EQUALITY))


@register_constraints
def max_production(
    demand: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: Optional[int] = None,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    """Constructs constraint between capacity and maximum production.

    Constrains the production decision variable by the maximum production for a given
    capacity.

    Example:

        >>> from muse import examples
        >>> from muse.constraints import max_production
        >>> technologies = examples.technodata("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> search_space = examples.search_space("residential", "medium")
        >>> assets = None  # not used in max_production
        >>> demand = None
        >>> maxprod = max_production(demand, assets, search_space, market, technologies)
    """
    from xarray import zeros_like, ones_like
    from muse.commodities import is_enduse
    from muse.timeslices import convert_timeslice, QuantityType

    if year is None:
        year = market.year.min()
    commodities = technologies.commodity.sel(
        commodity=is_enduse(technologies.comm_usage)
    )
    kwargs = dict(technology=search_space.replacement, year=year, commodity=commodities)
    if getattr(assets, "region", None) is not None and "region" in technologies.dims:
        kwargs["region"] = assets.region
    techs = technologies[["fixed_outputs", "utilization_factor"]].sel(**kwargs)
    capacity = convert_timeslice(
        techs.fixed_outputs * techs.utilization_factor,
        market.timeslice,
        QuantityType.EXTENSIVE,
    ).expand_dims(asset=search_space.asset)
    production = ones_like(capacity)
    b = zeros_like(production)
    return xr.Dataset(
        dict(capacity=cast(xr.DataArray, -capacity), production=production, b=b),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND),
    )


@register_constraints
def minimum_service(
    demand: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: Optional[int] = None,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    """ Constructs constraint between capacity and minimum service. """
    from xarray import zeros_like, ones_like
    from muse.commodities import is_enduse
    from muse.timeslices import convert_timeslice, QuantityType

    if "minimum_service_factor" not in technologies.data_vars:
        return None
    if np.all(technologies["minimum_service_factor"] == 0):
        return None
    if year is None:
        year = market.year.min()
    commodities = technologies.commodity.sel(
        commodity=is_enduse(technologies.comm_usage)
    )
    kwargs = dict(technology=search_space.replacement, year=year, commodity=commodities)
    if getattr(assets, "region", None) is not None and "region" in technologies.dims:
        kwargs["region"] = assets.region
    techs = technologies[
        ["fixed_outputs", "utilization_factor", "minimum_service_factor"]
    ].sel(**kwargs)
    capacity = convert_timeslice(
        techs.fixed_outputs * techs.utilization_factor * techs.minimum_service_factor,
        market.timeslice,
        QuantityType.EXTENSIVE,
    ).expand_dims(asset=search_space.asset)
    production = ones_like(capacity)
    b = zeros_like(production)
    return xr.Dataset(
        dict(capacity=cast(xr.DataArray, -capacity), production=production, b=b),
        attrs=dict(kind=ConstraintKind.LOWER_BOUND),
    )


def lp_costs(
    technologies: xr.Dataset, costs: xr.DataArray, timeslices: xr.DataArray
) -> xr.Dataset:
    """Creates costs for solving with scipy's LP solver.

    Example:

        We can now construct example inputs to the funtion from the sample model. The
        costs will be a matrix where each assets has a candidate replacement technology.

        >>> from muse import examples
        >>> technologies = examples.technodata("residential", model="medium")
        >>> search_space = examples.search_space("residential", model="medium")
        >>> timeslices = examples.sector("residential", model="medium").timeslices
        >>> costs = (
        ...     search_space
        ...     * np.arange(np.prod(search_space.shape)).reshape(search_space.shape)
        ... )

        The function returns the LP vector split along capacity and production
        variables.

        >>> from muse.constraints import lp_costs
        >>> lpcosts = lp_costs(
        ...     technologies.sel(year=2020, region="USA"), costs, timeslices
        ... )
        >>> assert "capacity" in lpcosts.data_vars
        >>> assert "production" in lpcosts.data_vars

        The capacity costs correspond exactly to the input costs:

        >>> assert (costs == lpcosts.capacity).all()

        The production is zero in this context. It does not enter the cost function of
        the LP problem:

        >>> assert (lpcosts.production == 0).all()

        They should correspond to a data-array with dimensions ``(asset, replacement)``
        (and possibly ``region`` as well).

        >>> lpcosts.capacity.dims
        ('asset', 'replacement')

        The production costs are zero by default. However, the production expands over
        not only the dimensions of the capacity, but also the ``timeslice`` during
        which production occurs and the ``commodity`` produced.

        >>> lpcosts.production.dims
        ('timeslice', 'asset', 'replacement', 'commodity')
    """
    from xarray import zeros_like
    from muse.commodities import is_enduse
    from muse.timeslices import convert_timeslice

    assert "year" not in technologies.dims

    production = zeros_like(
        convert_timeslice(costs, timeslices)
        * technologies.fixed_outputs.sel(
            commodity=is_enduse(technologies.comm_usage),
            technology=technologies.technology.isin(costs.replacement),
        ).rename(technology="replacement")
    )
    for dim in production.dims:
        if isinstance(production.get_index(dim), pd.MultiIndex):
            production[dim] = pd.Index(production.get_index(dim), tupleize_cols=False)
    return xr.Dataset(dict(capacity=costs, production=production))


def merge_lp(
    costs: xr.Dataset, *constraints: Constraint
) -> Tuple[xr.Dataset, List[Constraint]]:
    """Unify coordinate systems of costs and constraints.

    In practice, this function brings costs and constraints into a single xr.Dataset and
    then splits things up again. This ensures the dimensions are not only compatible,
    but also such that that their order in memory is the same.
    """
    from xarray import merge

    data = merge(
        [costs]
        + [
            constraint.rename(
                b=f"b{i}", capacity=f"capacity{i}", production=f"production{i}"
            )
            for i, constraint in enumerate(constraints)
        ]
    )

    unified_costs = cast(xr.Dataset, data[["capacity", "production"]])
    unified_constraints = [
        xr.Dataset(
            {
                "capacity": data[f"capacity{i}"],
                "production": data[f"production{i}"],
                "b": data[f"b{i}"],
            },
            attrs=constraint.attrs,
        )
        for i, constraint in enumerate(constraints)
    ]

    return unified_costs, unified_constraints


def lp_constraint(constraint: Constraint, lpcosts: xr.Dataset) -> Constraint:
    """Transforms the constraint to LP data.

    The goal is to create from ``lpcosts.capacity``, ``constraint.capacity``, and
    ``constraint.b`` a 2d-matrix ``constraint`` vs ``decision variables``.

    #. The dimensions of ``constraint.b`` are the constraint dimensions. They are
        renamed ``"c(xxx)"``.
    #. The dimensions of ``lpcosts`` are the decision-variable dimensions. They are
        renamed ``"d(xxx)"``.
    #. ``set(b.dims).intersection(lpcosts.xxx.dims)`` are diagonal
        in constraint dimensions and decision variables dimension, with ``xxx`` the
        capacity or the production
    #. ``set(constraint.xxx.dims) - set(lpcosts.xxx.dims) - set(b.dims)`` are reduced by
        summation, with ``xxx`` the capacity or the production
    #. ``set(lpcosts.xxx.dims) - set(constraint.xxx.dims) - set(b.dims)`` are added for
        expansion, with ``xxx`` the capacity or the production

    See :py:func:`muse.constraints.lp_constraint_matrix` for a more detailed explanation
    of the transformations applied here.
    """
    constraint = constraint.copy(deep=False)
    for dim in constraint.dims:
        if isinstance(constraint.get_index(dim), pd.MultiIndex):
            constraint[dim] = pd.Index(constraint.get_index(dim), tupleize_cols=False)
    b = constraint.b.drop_vars(set(constraint.b.coords) - set(constraint.b.dims))
    b = b.rename({k: f"c({k})" for k in b.dims})
    capacity = lp_constraint_matrix(constraint.b, constraint.capacity, lpcosts.capacity)
    capacity = capacity.drop_vars(set(capacity.coords) - set(capacity.dims))
    production = lp_constraint_matrix(
        constraint.b, constraint.production, lpcosts.production
    )
    production = production.drop_vars(set(production.coords) - set(production.dims))
    return xr.Dataset(
        {"b": b, "capacity": capacity, "production": production}, attrs=constraint.attrs
    )


def lp_constraint_matrix(
    b: xr.DataArray, constraint: xr.DataArray, lpcosts: xr.DataArray
):
    """Transforms one constraint block into an lp matrix.

   The goal is to create from ``lpcosts``, ``constraint``, and ``b`` a 2d-matrix of
   constraints vs decision variables.

    #. The dimensions of ``b`` are the constraint dimensions. They are renamed
        ``"c(xxx)"``.
    #. The dimensions of ``lpcosts`` are the decision-variable dimensions. They are
        renamed ``"d(xxx)"``.
    #. ``set(b.dims).intersection(lpcosts.dims)`` are diagonal
        in constraint dimensions and decision variables dimension
    #. ``set(constraint.dims) - set(lpcosts.dims) - set(b.dims)`` are reduced by
        summation
    #. ``set(lpcosts.dims) - set(constraint.dims) - set(b.dims)`` are added for
        expansion
    #. ``set(b.dims) - set(constraint.dims) - set(lpcosts.dims)`` are added for
        expansion. Such dimensions only make sense if they consist of one point.

    The result is the constraint matrix, expanded, reduced and diagonalized for the
    conditions above.

    Example:

        Lets first setup a constraint and a cost matrix:

        >>> from muse import examples
        >>> from muse import constraints as cs
        >>> res = examples.sector("residential", model="medium")
        >>> technologies = res.technologies
        >>> market = examples.residential_market("medium")
        >>> search = examples.search_space("residential", model="medium")
        >>> assets = next(a.assets for a in res.agents if a.category == "retrofit")
        >>> demand = None # not used in max production
        >>> constraint = cs.max_production(demand, assets, search, market, technologies)
        >>> lpcosts = cs.lp_costs(
        ...     (
        ...         technologies
        ...         .interp(year=market.year.min() + 5)
        ...         .drop_vars("year")
        ...         .sel(region=assets.region)
        ...     ),
        ...     costs=search * np.arange(np.prod(search.shape)).reshape(search.shape),
        ...     timeslices=market.timeslice,
        ... )

        For a simple example, we can first check the case where b is scalar. The result
        ought to be a single row of a matrix, or a vector with only decision variables:

        >>> from pytest import approx
        >>> result = cs.lp_constraint_matrix(
        ...     xr.DataArray(1), constraint.capacity, lpcosts.capacity
        ... )
        >>> assert result.values == approx(-1)
        >>> assert set(result.dims) == {f"d({x})" for x in lpcosts.capacity.dims}
        >>> result = cs.lp_constraint_matrix(
        ...     xr.DataArray(1), constraint.production, lpcosts.production
        ... )
        >>> assert set(result.dims) == {f"d({x})" for x in lpcosts.production.dims}
        >>> assert result.values == approx(1)

        As expected, the capacity vector is 1, whereas the production vector is -1.
        These are the values the :py:func:`~muse.constraints.max_production` is set up
        to create.

        Now, let's check the case where ``b`` is the one from the
        :py:func:`~muse.constraints.max_production` constraint. In that case, all the
        dimensions should end up as constraint dimensions: the production for each
        timeslice, region, asset, and replacement technology should not outstrip the
        capacity assigned for the asset and replacement technology.

        >>> result = cs.lp_constraint_matrix(
        ...     constraint.b, constraint.capacity, lpcosts.capacity
        ... )
        >>> decision_dims = {f"d({x})" for x in lpcosts.capacity.dims}
        >>> constraint_dims = {
        ...     f"c({x})" for x in set(lpcosts.production.dims).union(constraint.b.dims)
        ... }
        >>> assert set(result.dims) == decision_dims.union(constraint_dims)

        The :py:func:`~muse.constraints.max_production` constraint on the production
        side is the identy matrix with a factor :math:`-1`. We can easily check this
        by stacking the decision and constraint dimensions in the result:

        >>> result = cs.lp_constraint_matrix(
        ...     constraint.b, constraint.production, lpcosts.production
        ... )
        >>> decision_dims = {f"d({x})" for x in lpcosts.production.dims}
        >>> assert set(result.dims) == decision_dims.union(constraint_dims)
        >>> stacked = result.stack(d=sorted(decision_dims), c=sorted(constraint_dims))
        >>> assert stacked.shape[0] == stacked.shape[1]
        >>> assert stacked.values == approx(np.eye(stacked.shape[0]))
    """
    from numpy import eye
    from functools import reduce

    result = constraint.sum(set(constraint.dims) - set(lpcosts.dims) - set(b.dims))
    result = result.rename(
        {k: f"d({k})" for k in set(result.dims).intersection(lpcosts.dims)}
    )
    result = result.rename(
        {k: f"c({k})" for k in set(result.dims).intersection(b.dims)}
    )

    expand = set(lpcosts.dims) - set(constraint.dims) - set(b.dims)
    result = result.expand_dims({f"d({k})": lpcosts[k] for k in expand})
    expand = set(b.dims) - set(constraint.dims) - set(lpcosts.dims)
    result = result.expand_dims({f"c({k})": b[k] for k in expand})

    diag_dims = set(b.dims).intersection(lpcosts.dims)
    if diag_dims:

        def get_dimension(dim):
            if dim in b.dims:
                return b[dim].values
            if dim in lpcosts.dims:
                return lpcosts[dim].values
            return constraint[dim].values

        diagonal_submats = [
            xr.DataArray(
                eye(len(b[k])),
                coords={f"c({k})": get_dimension(k), f"d({k})": get_dimension(k)},
                dims=(f"c({k})", f"d({k})"),
            )
            for k in diag_dims
        ]
        result = result * reduce(xr.DataArray.__mul__, diagonal_submats)
    return result


@dataclass
class ScipyAdapter:
    """Creates the input for the scipy solvers.

    Example:

        Lets give a fist simple example. The constraint
        :py:func:`~muse.constraints.max_capacity_expansion` limits how much each
        capacity can be expanded in a given year.

        >>> from muse import examples
        >>> from muse.quantities import maximum_production
        >>> from muse.timeslices import convert_timeslice
        >>> from muse import constraints as cs
        >>> res = examples.sector("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> search = examples.search_space("residential", model="medium")
        >>> assets = next(a.assets for a in res.agents if a.category == "retrofit")
        >>> market_demand =  0.8 * maximum_production(
        ...     res.technologies.interp(year=2025),
        ...     convert_timeslice(
        ...         assets.capacity.sel(year=2025).groupby("technology").sum("asset"),
        ...         market.timeslice,
        ...     ),
        ... ).rename(technology="asset")
        >>> costs = search * np.arange(np.prod(search.shape)).reshape(search.shape)
        >>> constraint = cs.max_capacity_expansion(
        ...     market_demand, assets, search, market, res.technologies,
        ... )

        The constraint acts over capacity decision variables only:

        >>> assert constraint.production.data == np.array(0)
        >>> assert len(constraint.production.dims) == 0

        It is an upper bound for a straightforward sum over the capacities for a given
        technology. The matrix operator is simply the identity:

        >>> assert constraint.capacity.data == np.array(1)
        >>> assert len(constraint.capacity.dims) == 0

        And the upperbound is exanded over the replacement technologies,
        but not over the assets. Hence the assets will be summed over in the final
        constraint:

        >>> assert (constraint.b.data == np.array([[500.0]] * 4)).all()
        >>> assert set(constraint.b.dims) == {"replacement"}
        >>> assert constraint.kind == cs.ConstraintKind.UPPER_BOUND

        As shown above, it does not bind the production decision variables. Hence,
        production is zero. The matrix operator for the capacity is simply the identity.
        Hence it can be inputed as the dimensionless scalar 1. The upper bound is simply
        the maximum for replacement technology (and region, if that particular dimension
        exists in the problem).

        The lp problem then becomes:

        >>> technologies = res.technologies.interp(year=market.year.min() + 5)
        >>> inputs = cs.ScipyAdapter.factory(
        ...     technologies, costs, market.timeslice, constraint
        ... )

        The decision variables are always constrained between zero and infinity:

        >>> assert inputs.bounds == (0, np.inf)

        The problem is an upper-bound one. There are no equality constraints:

        >>> assert inputs.A_eq is None
        >>> assert inputs.b_eq is None

        The upper bound matrix and vector, and the costs are consistent in their
        dimensions:

        >>> assert inputs.c.ndim == 1
        >>> assert inputs.b_ub.ndim == 1
        >>> assert inputs.A_ub.ndim == 2
        >>> assert inputs.b_ub.size == inputs.A_ub.shape[0]
        >>> assert inputs.c.size == inputs.A_ub.shape[1]
        >>> assert inputs.c.ndim == 1

        In practice, :py:func:`~muse.constraints.lp_costs` helps us define the decision
        variables (and ``c``). We can verify that the sizes are consistent:

        >>> lpcosts = cs.lp_costs(technologies, costs, market.timeslice)
        >>> capsize = lpcosts.capacity.size
        >>> prodsize = lpcosts.production.size
        >>> assert inputs.c.size == capsize + prodsize

        The upper bound itself is over each replacement technology:

        >>> assert inputs.b_ub.size == lpcosts.replacement.size

        The production decision variables are not involved:

        >>> from pytest import approx
        >>> assert inputs.A_ub[:, capsize:] == approx(0)

        The matrix for the capacity decision variables is a sum over assets for a given
        replacement technology. Hence, each row is constituted of zeros and ones and
        sums to the number of assets:

        >>> assert inputs.A_ub[:, :capsize].sum(axis=1) == approx(lpcosts.asset.size)
        >>> assert set(inputs.A_ub[:, :capsize].flatten()) == {0.0, 1.0}
    """

    c: np.ndarray
    to_muse: Callable[[np.ndarray], xr.Dataset]
    bounds: Tuple[Optional[float], Optional[float]] = (0, np.inf)
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None

    @classmethod
    def factory(
        cls,
        technologies: xr.Dataset,
        costs: xr.DataArray,
        timeslices: pd.Index,
        *constraints: Constraint,
    ) -> ScipyAdapter:

        lpcosts = lp_costs(technologies, costs, timeslices)
        data = cls._unified_dataset(technologies, lpcosts, *constraints)
        capacities = cls._stacked_quantity(data, "capacity")
        productions = cls._stacked_quantity(data, "production")
        bs = cls._stacked_quantity(data, "b")
        kwargs = cls._to_scipy_adapter(capacities, productions, bs, *constraints)

        def to_muse(x: np.ndarray) -> xr.Dataset:
            return ScipyAdapter._back_to_muse(x, capacities.costs, productions.costs)

        return ScipyAdapter(to_muse=to_muse, **kwargs)

    @property
    def kwargs(self):
        return {
            "c": self.c,
            "A_eq": self.A_eq,
            "b_eq": self.b_eq,
            "A_ub": self.A_ub,
            "b_ub": self.b_ub,
            "bounds": self.bounds,
        }

    @staticmethod
    def _unified_dataset(
        technologies: xr.Dataset, lpcosts: xr.Dataset, *constraints: Constraint
    ) -> xr.Dataset:
        """Creates single xr.Dataset from costs and contraints."""
        from xarray import merge

        assert "year" not in technologies.dims
        data = merge(
            [lpcosts.rename({k: f"d({k})" for k in lpcosts.dims})]
            + [
                lp_constraint(constraint, lpcosts).rename(
                    b=f"b{i}", capacity=f"capacity{i}", production=f"production{i}"
                )
                for i, constraint in enumerate(constraints)
            ]
        )
        for i, constraint in enumerate(constraints):
            if constraint.kind == ConstraintKind.LOWER_BOUND:
                data[f"b{i}"] = -data[f"b{i}"]  # type: ignore
                data[f"capacity{i}"] = -data[f"capacity{i}"]  # type: ignore
                data[f"production{i}"] = -data[f"production{i}"]  # type: ignore

        return data

    @staticmethod
    def _stacked_quantity(data: xr.Dataset, name: Text) -> xr.Dataset:
        result = cast(
            xr.Dataset, data[[u for u in data.data_vars if str(u).startswith(name)]]
        )
        result = result.rename(
            {
                k: ("costs" if k == name else int(str(k).replace(name, "")))
                for k in result.data_vars
            }
        )
        if len(result) and "costs" in result.data_vars:
            result = result.stack(decision=sorted(result["costs"].dims))
        return result

    @staticmethod
    def _to_scipy_adapter(
        capacities: xr.Dataset, productions: xr.Dataset, bs: xr.Dataset, *constraints
    ):
        def extract_bA(constraints, *kinds):
            indices = [i for i in range(len(bs)) if constraints[i].kind in kinds]
            capa_constraints = [
                capacities[i]
                .stack(constraint=sorted(bs[i].dims))
                .transpose("constraint", "decision")
                .values
                for i in indices
            ]
            prod_constraints = [
                productions[i]
                .stack(constraint=sorted(bs[i].dims))
                .transpose("constraint", "decision")
                .values
                for i in indices
            ]
            if capa_constraints:
                A: Optional[np.ndarray] = np.concatenate(
                    (
                        np.concatenate(capa_constraints, axis=0),
                        np.concatenate(prod_constraints, axis=0),
                    ),
                    axis=1,
                )
                b: Optional[np.ndarray] = np.concatenate(
                    [bs[i].stack(constraint=sorted(bs[i].dims)) for i in indices],
                    axis=0,
                )
            else:
                A = None
                b = None
            return A, b

        c = np.concatenate(
            (capacities["costs"].values, productions["costs"].values), axis=0
        )
        A_ub, b_ub = extract_bA(
            constraints, ConstraintKind.UPPER_BOUND, ConstraintKind.LOWER_BOUND
        )
        A_eq, b_eq = extract_bA(constraints, ConstraintKind.EQUALITY)

        return {
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": (0, np.inf),
        }

    @staticmethod
    def _back_to_muse_quantity(
        x: np.ndarray, template: Union[xr.DataArray, xr.Dataset]
    ) -> xr.DataArray:
        result = xr.DataArray(x, coords=template.coords, dims=template.dims).unstack(
            "decision"
        )
        return result.rename({k: str(k)[2:-1] for k in result.dims})

    @staticmethod
    def _back_to_muse(
        x: np.ndarray, capacity: xr.DataArray, production: xr.DataArray
    ) -> xr.Dataset:
        capa = ScipyAdapter._back_to_muse_quantity(x[: capacity.size], capacity)
        prod = ScipyAdapter._back_to_muse_quantity(x[capacity.size :], production)
        return xr.Dataset({"capacity": capa, "production": prod})
