r"""Investment constraints.

Constraints on investments ensure that investments match some given criteria. For
instance, the constraints could ensure that only so much of a new asset can be built
every year.

Functions to compute constraints should be registered via the decorator
:py:meth:`~muse.constraints.register_constraints`. This registration step makes it
possible for constraints to be declared in the TOML file.

Generally, LP solvers accept linear constraints defined as:

.. math::

    A x \\leq b

with :math:`A` a matrix, :math:`x` the decision variables, and :math:`b` a vector.
However, these quantities are dimensionless. They do no have timeslices, assets, or
replacement technologies, or any other dimensions that users have set up in their model.
The crux is to translate from MUSE's data-structures to a consistent dimensionless
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
  :math:`b` is reduced by summation over a row in the left (right) submatrix of
  :math:`A`. In other words, those dimensions become part of a standard tensor
  reduction or matrix multiplication.

There are two additional rules. However, they are likely to be the result of an
inefficient definition of :math:`A_c`, :math:`A_p` and :math:`b`.

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
        capacity: xr.DataArray,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        **kwargs,
    ) -> Constraint:
        pass

demand:
    The demand for the sectors products in the investment year. In practice it is a
    demand share obtained in :py:mod:`~muse.demand_share`. It is a data-array with
    dimensions including `asset`, `commodity`, `timeslice`.
capacity:
    A data-array with dimensions `technology` and `year` defining the existing capacity
    of each technology in the current year and investment year.
search_space:
    A matrix `asset` vs `replacement` technology defining which replacement technologies
    will be considered for each existing asset.
technologies:
    Technodata characterizing the competing technologies in the investment year.
``**kwargs``:
    Any other parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator
from muse.timeslices import broadcast_timeslice, distribute_timeslice, drop_timeslice

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
of the constraint. It should also contain a data-array `capacity` corresponding to the
left-hand-side matrix operator which will be applied to the capacity-related decision
variables.  It should contain a similar matrix `production` corresponding to
the left-hand-side matrix operator which will be applied to the production-related
decision variables. Should any of these three objects be missing, they default to the
scalar 0. Finally, the constraint should contain an attribute `kind` of type
:py:class:`ConstraintKind` defining the operation. If it is missing, it defaults to an
upper bound constraint.
"""


CONSTRAINT_SIGNATURE = Callable[
    [xr.DataArray, xr.DataArray, xr.DataArray, xr.Dataset, KwArg(Any)],
    Optional[Constraint],
]
"""Basic signature for functions producing constraints.

.. note::

    A constraint can return `None`, in which case it is ignored. This makes it simple to
    add constraints that are only used if some condition is met, e.g. minimum service
    conditions are defined in the technodata.
"""
CONSTRAINTS: MutableMapping[str, CONSTRAINT_SIGNATURE] = {}
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
        capacity: xr.DataArray,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        **kwargs,
    ) -> Constraint | None:
        """Computes and standardizes a constraint."""
        # Check inputs
        assert "year" not in technologies.dims
        assert len(capacity.year) == 2  # current year and investment year

        # Calculate constraint
        constraint = function(  # type: ignore
            demand,
            capacity=capacity,
            search_space=search_space,
            technologies=technologies,
            **kwargs,
        )

        if constraint is not None:
            # Check constraint
            if "b" not in constraint.data_vars:
                raise RuntimeError("Constraint must contain a right-hand-side vector")
            assert not constraint.b.dims == ()
            if (
                "capacity" not in constraint.data_vars
                and "production" not in constraint.data_vars
            ):
                raise RuntimeError("Constraint must contain a left-hand-side matrix")
            if "kind" not in constraint.attrs:
                raise RuntimeError("Constraint must contain a kind attribute")

            # Standardize constraint
            if "capacity" not in constraint.data_vars:
                constraint["capacity"] = 0
            if "production" not in constraint.data_vars:
                constraint["production"] = 0
            if "name" not in constraint.data_vars and "name" not in constraint.attrs:
                constraint.attrs["name"] = function.__name__

            # ensure that the constraint and the search space match
            dims = [d for d in constraint.dims if d in search_space.dims]
            constraint = constraint.sel({k: search_space[k] for k in dims})

        return constraint

    return decorated


def factory(
    settings: str | Mapping | Sequence[str] | Sequence[str | Mapping] | None = None,
) -> Callable:
    """Creates a list of constraints from standard settings.

    The standard settings can be a string naming the constraint, a dictionary including
    at least "name", or a list of strings and dictionaries.
    """
    from functools import partial

    if not settings:
        settings = (
            "max_production",
            "max_capacity_expansion",
            "demand",
            "search_space",
            "minimum_service",
            "demand_limiting_capacity",
        )

    def normalize(x) -> MutableMapping:
        return dict(name=x) if isinstance(x, str) else x

    if isinstance(settings, (str, Mapping)):
        settings = cast(Union[Sequence[str], Sequence[Mapping]], [settings])
    parameters = [normalize(x) for x in settings]
    names = [x.pop("name") for x in parameters]

    constraint_closures = [
        partial(CONSTRAINTS[name], **param) for name, param in zip(names, parameters)
    ]

    def constraints(
        demand: xr.DataArray,
        capacity: xr.DataArray,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        **kwargs,
    ) -> list[Constraint]:
        constraints = [
            function(
                demand,
                capacity=capacity,
                search_space=search_space,
                technologies=technologies,
                **kwargs,
            )
            for function in constraint_closures
        ]
        return [constraint for constraint in constraints if constraint is not None]

    return constraints


@register_constraints
def max_capacity_expansion(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    **kwargs,
) -> Constraint | None:
    r"""Max-capacity addition, max-capacity growth, and capacity limits constraints.

    Limits by how much the capacity of each technology owned by an agent can grow in
    a given year. This is a constraint on the agent's ability to invest in a
    technology.

    Let :math:`L_t^r(y)` be the total capacity limit for a given year, technology,
    and region. :math:`G_t^r(y)` is the maximum growth. And :math:`W_t^r(y)` is
    the maximum additional capacity. :math:`y=y_0` is the current year and
    :math:`y=y_1` is the year marking the end of the investment period.

    Let :math:`\mathcal{A}^{i, r}_{t, \iota}(y)` be the current assets, before
    investment, and let :math:`\Delta\mathcal{A}^{i,r}_t` be the future investments.
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

    :math:`L_t^r(y)`, :math:`G_t^r(y)` and :math:`W_t^r(y)` default to np.inf if
    not provided (i.e. no constraint).
    If all three parameters are not provided, no constraint is applied (returns None).
    """
    # If all three parameters are missing, don't apply the constraint
    if not any(
        param in technologies
        for param in (
            "max_capacity_addition",
            "total_capacity_limit",
            "max_capacity_growth",
        )
    ):
        return None

    # case with technology and region in asset dimension
    if capacity.region.dims != ():
        names = [u for u in capacity.asset.coords if capacity[u].dims == ("asset",)]
        index = pd.MultiIndex.from_arrays(
            [capacity[u].values for u in names], names=names
        )
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, "asset")
        capacity = capacity.drop_vars(names).assign_coords(mindex_coords)
        capacity = capacity.unstack("asset", fill_value=0).rename(
            technology=search_space.replacement.name
        )
    # case with only technology in asset dimension
    else:
        capacity = cast(xr.DataArray, capacity.set_index(asset="technology")).rename(
            asset=search_space.replacement.name
        )
    capacity = capacity.reindex_like(search_space.replacement, fill_value=0)

    replacement = search_space.replacement
    replacement = replacement.drop_vars(
        [u for u in replacement.coords if u not in replacement.dims]
    )
    techs = technologies.sel(technology=replacement).drop_vars("technology")
    regions = getattr(capacity, "region", None)
    if regions is not None and "region" in technologies.dims:
        techs = techs.sel(region=regions)

    # Existing and forecasted capacity
    initial = capacity.isel(year=0, drop=True)
    forecasted = capacity.isel(year=1, drop=True)
    time_frame = int(capacity.year[1] - capacity.year[0])

    # MaxCapacityAddition constraint
    if "max_capacity_addition" in techs:
        add_cap = techs.max_capacity_addition * time_frame
    else:
        add_cap = np.inf

    # TotalCapacityLimit constraint
    if "total_capacity_limit" in techs:
        limit = techs.total_capacity_limit
        total_cap = (limit - forecasted).clip(min=0)
    else:
        total_cap = np.inf

    # MaxCapacityGrowth constraint
    if "max_capacity_growth" in techs:
        seed = techs.get("growth_seed", 1)
        seeded_initial = np.maximum(initial, seed)
        growth_cap = (
            (seeded_initial * (techs.max_capacity_growth + 1) ** time_frame)
            - forecasted
        ).clip(min=0)
    else:
        growth_cap = np.inf

    # Take the most restrictive constraint
    b = np.minimum(np.minimum(add_cap, total_cap), growth_cap)

    # np.inf values are not allowed in the final constraint - raise error
    # Will happen if user provides "inf" for all three parameters for any technology
    if np.isinf(b).any():
        inf_replacements = b.replacement[np.isinf(b)].values
        raise ValueError(
            "Capacity growth constraint cannot be infinite. "
            f"Check growth constraint parameters for technologies: {inf_replacements}"
        )

    # np.nan values are also not allowed. This shouldn't happen but just in case
    if np.isnan(b).any():
        nan_replacements = b.replacement[np.isnan(b)].values
        raise ValueError(
            "Capacity growth constraint cannot be NaN. "
            f"Check growth constraint parameters for technologies: {nan_replacements}"
        )

    if b.region.dims == ():
        capa = 1
    elif "dst_region" in b.dims:
        b = b.rename(region="src_region")
        capa = search_space.agent.region == b.src_region

    return xr.Dataset(
        dict(b=b, capacity=capa),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND, name="max capacity expansion"),
    )


@register_constraints
def demand(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    **kwargs,
) -> Constraint:
    """Constraints production to meet demand."""
    b = demand
    if "region" in b.dims and "dst_region" in technologies.dims:
        b = b.rename(region="dst_region")
    assert "year" not in b.dims
    return xr.Dataset(
        dict(b=b, production=1), attrs=dict(kind=ConstraintKind.LOWER_BOUND)
    )


@register_constraints
def search_space(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    **kwargs,
) -> Constraint | None:
    """Removes disabled technologies."""
    if search_space.all():
        return None
    capacity = cast(xr.DataArray, 1 - 2 * cast(np.ndarray, search_space))
    b = xr.zeros_like(capacity)
    return xr.Dataset(
        dict(b=b, capacity=capacity), attrs=dict(kind=ConstraintKind.UPPER_BOUND)
    )


@register_constraints
def max_production(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    *,
    timeslice_level: str | None = None,
    **kwargs,
) -> Constraint:
    """Constructs constraint between capacity and maximum production.

    Constrains the production decision variable by the maximum production for a given
    capacity.
    """
    from xarray import ones_like, zeros_like

    commodities = demand.commodity
    kwargs = dict(commodity=commodities)
    if "region" in search_space.coords and "region" in technologies.dims:
        kwargs["region"] = search_space.region
    techs = (
        technologies[["fixed_outputs", "utilization_factor"]]
        .sel(**kwargs)
        .rename(technology="replacement")
    )
    capa = distribute_timeslice(
        techs.fixed_outputs, level=timeslice_level
    ) * broadcast_timeslice(techs.utilization_factor, level=timeslice_level)
    if "asset" not in capa.dims and "asset" in search_space.dims:
        capa = capa.expand_dims(asset=search_space.asset)
    production = ones_like(capa)
    b = zeros_like(production)

    # Include maxaddition constraint in max production to match region-dst_region
    if "dst_region" in technologies.dims:
        b = b.expand_dims(dst_region=technologies.dst_region)
        capa = capa.rename(region="src_region")
        production = production.rename(region="src_region")
        maxadd = technologies.max_capacity_addition.rename(region="src_region")
        maxadd = maxadd.rename(technology="replacement")
        maxadd = maxadd.where(maxadd == 0, 0.0)
        maxadd = maxadd.where(maxadd > 0, -1.0)
        capa = capa * broadcast_timeslice(maxadd, level=timeslice_level)
        production = production * broadcast_timeslice(maxadd, level=timeslice_level)
        b = b.rename(region="src_region")

    return xr.Dataset(
        dict(capacity=-capa, production=production, b=b),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND),
    )


@register_constraints
def demand_limiting_capacity(
    demand_: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    *,
    timeslice_level: str | None = None,
    **kwargs,
) -> Constraint:
    """Limits the maximum combined capacity to match the demand.

    This is a somewhat more restrictive constraint than the max_production constraint or
    the maximum capacity expansion. In this case, the combined new capacity of all
    assets must be sufficient to meet the demand of the most demanding timeslice, and
    no more.

    Rather than coding from scratch the constraint, we can use the max_production
    constraint and the demand constraint to construct this constraint. Starting from
    the maximum production instead of the maximum capacity ensures that the constraint
    accounts for the utilization factor of the technologies.
    """
    # We start with the maximum production constraint and the demand constraint
    capacity_constraint = max_production(
        demand_,
        capacity,
        search_space,
        technologies,
        timeslice_level=timeslice_level,
    )
    demand_constraint = demand(demand_, capacity, search_space, technologies)

    # We are interested in the demand of the demand constraint and the capacity of the
    # capacity constraint.
    b = demand_constraint.b
    capacity = -capacity_constraint.capacity

    # Drop 'year' so there's no conflict with the 'year' in the capacity constraint
    if "year" in b.coords and "year" in capacity.coords:
        b = b.drop_vars("year")

    # If there are timeslices, we need to find the one where more capacity is needed to
    # meet the demand which would be a combination of a high demand and a low
    # utilization factor.
    if "timeslice" in b.dims or "timeslice" in capacity.dims:
        ratio = b / capacity
        ts_index = ratio.min("replacement").argmax("timeslice")
        b = b.isel(timeslice=ts_index)
        capacity = capacity.isel(timeslice=ts_index)

    # An adjustment is required to account for technologies that have multiple output
    # commodities
    b = modify_dlc(technologies=capacity, demand=b)

    return xr.Dataset(
        dict(capacity=capacity, b=b),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND),
    )


def modify_dlc(technologies: xr.DataArray, demand: xr.DataArray) -> xr.DataArray:
    """Modifies DLC constraint to account for techs with multiple output commodities.

    Adjusts the commodity-level DLC based on the commodity output ratios of the
    available technologies, to allow for appropriate production of side-products.

    Args:
        technologies: DataArray with dimensions "commodity" and "replacement". This
            defines the fixed commodity outputs for each potential replacement
            technology.
        demand: DataArray with dimension "commodity", which defines the demand for each
            commodity.

    Returns:
        DataArray with dimension "commodity", which defines the new demand-limiting
        capacity constraint for each commodity.

    Example:
        Let's consider a simple example of a refinery sector with two alternative
        technologies that each produce two commodities: gasoline and diesel.

        We define the technologies DataArray as follows:
        >>> import xarray as xr
        >>> technologies = xr.DataArray(
        ...     data=[[1, 5], [0.5, 1]],
        ...     dims=['replacement', 'commodity'],
        ...     coords={'replacement': ['technology1', 'technology2'],
        ...             'commodity': ['gasoline', 'diesel']},
        ... )

        technology1 produces 1 unit of gasoline and 5 units of diesel (per unit of
        activity), whereas technology2 produces 0.5 units of gasoline and 1 unit of
        diesel.

        In this scenario, let's also define the demand for gasoline and diesel as
        follows (1 unit of demand for gasoline and 0 units for diesel):
        >>> demand = xr.DataArray(
        ...     data=[1, 0],
        ...     dims=['commodity'],
        ...     coords={'commodity': ['gasoline', 'diesel']},
        ... )

        The aim of the demand-limiting capacity (DLC) constraint is to limit the
        capacity of each technology so that supply is sufficient to meet the demand for
        each commodity, and no more.

        However, in this case we have a problem. The demand for gasoline can be met by
        either technology1 or technology2 (as both produce gasoline), but doing so would
        require producing up to 5 units of diesel (if all demand was met by
        technology1), which would exceed the diesel demand (0). Therefore, to allow the
        model to meet the demand for gasoline via either technology, we must relax the
        DLC constraint on diesel (to 5 units).

        In general, for an arbitrary set of technologies and commodity demands, the
        DLC of each commodity needs to be sufficiently high to permit any technology to
        act in service of any appropriate commodity demand, and no higher.

        The first step is to calculate the commodity output ratios for each technology:
        >>> output_ratios = technologies.rename({"commodity": "commodity2"}) / technologies
        >>> output_ratios
        <xarray.DataArray (replacement: 2, commodity2: 2, commodity: 2)> Size: 64B
        array([[[1. , 0.2],
                [5. , 1. ]],
        <BLANKLINE>
               [[1. , 0.5],
                [2. , 1. ]]])
        Coordinates:
          * replacement  (replacement) <U11 88B 'technology1' 'technology2'
          * commodity2   (commodity2) <U8 64B 'gasoline' 'diesel'
          * commodity    (commodity) <U8 64B 'gasoline' 'diesel'

        We introduce the dimension "commodity2" to compare the outputs of each commodity
        against every other commodity. For example, for technology1, producing 1 unit of
        gasoline leads to 5 units of diesel, whereas producing 1 unit of diesel leads to
        0.2 units of gasoline.

        Multiplying these output ratios by the demand, we get the full outputs that each
        technology would produce whilst acting in service of each commodity-level
        demand:
        >>> outputs = output_ratios * demand
        >>> outputs
        <xarray.DataArray (replacement: 2, commodity2: 2, commodity: 2)> Size: 64B
        array([[[1., 0.],
                [5., 0.]],
        <BLANKLINE>
               [[1., 0.],
                [2., 0.]]])
        Coordinates:
          * replacement  (replacement) <U11 88B 'technology1' 'technology2'
          * commodity2   (commodity2) <U8 64B 'gasoline' 'diesel'
          * commodity    (commodity) <U8 64B 'gasoline' 'diesel'

        In this case, meeting the gasoline demand with technology1 would require
        producing 1 unit of gasoline and 5 units of diesel, whereas meeting the gasoline
        demand with technology2 would require producing 1 unit of gasoline and 2 units
        of diesel. Since there is no diesel demand, all values for commodity = "diesel"
        are zero.

        Then, taking a maximum over the "commodity" dimension, we get the maximum
        potential outputs of each technology:
        >>> max_outputs = outputs.max("commodity")
        >>> max_outputs
        <xarray.DataArray (replacement: 2, commodity2: 2)> Size: 32B
        array([[1., 5.],
               [1., 2.]])
        Coordinates:
          * replacement  (replacement) <U11 88B 'technology1' 'technology2'
          * commodity2   (commodity2) <U8 64B 'gasoline' 'diesel'

        In this case, this is just the outputs of each technology when acting in service
        of the gasoline demand.

        Finally, summing over the "replacement" dimension, we get the maximum potential
        outputs of each commodity:
        >>> dlc = max_outputs.max("replacement").rename({"commodity2": "commodity"})
        >>> dlc
        <xarray.DataArray (commodity: 2)> Size: 16B
        array([1., 5.])
        Coordinates:
          * commodity  (commodity) <U8 64B 'gasoline' 'diesel'

        In this case, we get the maximum potential production of diesel as 5 units,
        which would occur as a side-product when technology1 is acting in service of the
        gasoline demand. This becomes the new DLC constraint.

        Putting this all together:
        >>> from muse.constraints import modify_dlc
        >>> modify_dlc(technologies, demand)
        <xarray.DataArray (commodity: 2)> Size: 16B
        array([1., 5.])
        Coordinates:
          * commodity  (commodity) <U8 64B 'gasoline' 'diesel'
    """  # noqa: E501
    # Calculate commodity output ratios for each technology
    output_ratios = technologies.rename({"commodity": "commodity2"}) / technologies
    output_ratios = output_ratios.where(np.isfinite(output_ratios), 0)  # this is
    # necessary for technologies that do not produce every commodity, which would lead
    # to an "infinite" ratio between commodities

    # Calculate the full outputs of each technology acting in service of each commodity
    # demand
    outputs = output_ratios * demand

    # Maximum potential outputs for each technology
    max_outputs = outputs.max("commodity")

    # Maximum potential production of each commodity -> demand-limiting capacity
    b = max_outputs.max("replacement").rename({"commodity2": "commodity"})
    return b


@register_constraints
def minimum_service(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    *,
    timeslice_level: str | None = None,
    **kwargs,
) -> Constraint | None:
    """Constructs constraint between capacity and minimum service."""
    from xarray import ones_like, zeros_like

    if "minimum_service_factor" not in technologies.data_vars:
        return None
    if np.all(technologies["minimum_service_factor"] == 0):
        return None

    commodities = demand.commodity
    kwargs = dict(commodity=commodities)
    if "region" in search_space.coords and "region" in technologies.dims:
        kwargs["region"] = search_space.region
    techs = (
        technologies[["fixed_outputs", "minimum_service_factor"]]
        .sel(**kwargs)
        .rename(technology="replacement")
    )
    capacity = distribute_timeslice(
        techs.fixed_outputs, level=timeslice_level
    ) * broadcast_timeslice(techs.minimum_service_factor, level=timeslice_level)
    if "asset" not in capacity.dims and "asset" in search_space.dims:
        capacity = capacity.expand_dims(asset=search_space.asset)
    production = ones_like(capacity)
    b = zeros_like(production)

    return xr.Dataset(
        dict(capacity=-capacity, production=production, b=b),
        attrs=dict(kind=ConstraintKind.LOWER_BOUND),
    )


def lp_costs(
    capacity_costs: xr.DataArray,
    commodities: list[str],
    timeslice_level: str | None = None,
) -> xr.Dataset:
    """Creates dataset of costs for solving with scipy's LP solver.

    Importantly, this also defines the decision variables in the linear program.

    The costs applied to the capacity decision variables are provided. This should
    have dimensions "asset" and "replacement". In other words, capacity addition
    is solved for each replacement technology for each existing asset.

    No cost is applied to the production decision variables. Thus, the production
    component of the resulting dataset is zero, with dimensions determining the
    production decision variables. This will have dimensions "asset", "replacement",
    "commodity", and "timeslice". In other words, production is solved for each
    replacement technology for each existing asset, for each commodity, and for each
    timeslice.

    Args:
        capacity_costs: DataArray with dimensions "asset" and "replacement" defining the
            costs of adding capacity to the system.
        commodities: List of commodities to create production decision variables for.
        timeslice_level: The timeslice level of the linear problem.
    """
    assert set(capacity_costs.dims) == {"asset", "replacement"}

    # Start with capacity costs as template (defines "asset" and "replacement" dims)
    production_costs = xr.zeros_like(capacity_costs)

    # Add a "timeslice" dimension, convert multiindex to single index
    production_costs = broadcast_timeslice(production_costs, level=timeslice_level)
    production_costs = drop_timeslice(production_costs)
    production_costs["timeslice"] = pd.Index(
        production_costs.get_index("timeslice"), tupleize_cols=False
    )

    # Add a "commodity" dimension
    production_costs = production_costs.expand_dims(commodity=commodities)
    assert set(production_costs.dims) == {
        "asset",
        "replacement",
        "commodity",
        "timeslice",
    }

    # Result is dataset of provided capacity costs and zero production costs
    return xr.Dataset(dict(capacity=capacity_costs, production=production_costs))


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

    # Deal with timeslice multiindex
    if "timeslice" in constraint.dims:
        constraint = drop_timeslice(constraint)
        constraint["timeslice"] = pd.Index(
            constraint.get_index("timeslice"), tupleize_cols=False
        )

    # Rename dimensions in b
    b = constraint.b.drop_vars(set(constraint.b.coords) - set(constraint.b.dims))
    b = b.rename({k: f"c({k})" for k in b.dims})

    # Create capacity constraint matrix
    capacity = lp_constraint_matrix(constraint.b, constraint.capacity, lpcosts.capacity)
    capacity = capacity.drop_vars(set(capacity.coords) - set(capacity.dims))

    # Create production constraint matrix
    production = lp_constraint_matrix(
        constraint.b, constraint.production, lpcosts.production
    )
    production = production.drop_vars(set(production.coords) - set(production.dims))

    # Combine data
    result = xr.Dataset(
        {"b": b, "capacity": capacity, "production": production}, attrs=constraint.attrs
    )
    return result


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
         >>> from muse.utilities import reduce_assets
         >>> res = examples.sector("residential", model="medium")
         >>> market = examples.residential_market("medium")
         >>> technologies = res.technologies.sel(year=2025)
         >>> search = examples.search_space("residential", model="medium")
         >>> assets = next(a.assets for a in res.agents)
         >>> capacity = reduce_assets(assets.capacity, coords=("region", "technology"))
         >>> demand = None # not used in max production
         >>> constraint = cs.max_production(demand, capacity.sel(year=[2020, 2025]),
         ...                                search, technologies) # noqa: E501
         >>> lpcosts = cs.lp_costs(
         ...     (
         ...         technologies
         ...         .sel(region=assets.region)
         ...     ),
         ...     costs=search * np.arange(np.prod(search.shape)).reshape(search.shape),
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
         ...     f"c({x})"
         ...     for x in set(lpcosts.production.dims).union(constraint.b.dims)
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
         >>> result = result.reset_index("d(timeslice)", drop=True).assign_coords(
         ...        {"d(timeslice)": result["d(timeslice)"].values}
         ... )
         >>> stacked = result.stack(d=sorted(decision_dims), c=sorted(constraint_dims))
         >>> assert stacked.shape[0] == stacked.shape[1]
         >>> assert stacked.values == approx(np.eye(stacked.shape[0]))
    """
    from functools import reduce

    from numpy import eye

    # Sum over all dimensions that are not in the constraint or the decision variables
    result = constraint.sum(set(constraint.dims) - set(lpcosts.dims) - set(b.dims))

    # Rename dimensions for decision variables
    result = result.rename(
        {k: f"d({k})" for k in set(result.dims).intersection(lpcosts.dims)}
    )

    # Rename dimensions for constraints
    result = result.rename(
        {k: f"c({k})" for k in set(result.dims).intersection(b.dims)}
    )

    # Expand dimensions that are in the decision variables but not in the constraint
    expand = set(lpcosts.dims) - set(constraint.dims) - set(b.dims)
    result = result.expand_dims(
        {f"d({k})": lpcosts[k].rename({k: f"d({k})"}).set_index() for k in expand}
    )

    # Expand dimensions that are in the constraint but not in the decision variables
    expand = set(b.dims) - set(constraint.dims) - set(lpcosts.dims)
    result = result.expand_dims(
        {f"c({k})": b[k].rename({k: f"c({k})"}).set_index() for k in expand}
    )

    # Dimensions that are in both the decision variables and the constraint
    diag_dims = set(b.dims).intersection(lpcosts.dims)
    diag_dims = sorted(diag_dims)

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
        reduced = reduce(xr.DataArray.__mul__, diagonal_submats)
        if "d(timeslice)" in reduced.dims:
            reduced = reduced.drop_vars("d(timeslice)")
        result = result * reduced

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
        >>> from muse.utilities import reduce_assets
        >>> from muse import constraints as cs
        >>> res = examples.sector("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> technologies = res.technologies.sel(year=2025)
        >>> search = examples.search_space("residential", model="medium")
        >>> assets = next(a.assets for a in res.agents)
        >>> capacity = reduce_assets(assets.capacity, coords=("region", "technology"))
        >>> market_demand = 0.8 * maximum_production(
        ...     technologies,
        ...     assets.capacity,
        ... ).sel(year=2025)
        >>> costs = search * np.arange(np.prod(search.shape)).reshape(search.shape)
        >>> constraint = cs.max_capacity_expansion(
        ...     market_demand, capacity.sel(year=[2020, 2025]), search, technologies)

        The constraint acts over capacity decision variables only:

        >>> assert constraint.production.data == np.array(0)
        >>> assert len(constraint.production.dims) == 0

        It is an upper bound for a straightforward sum over the capacities for a given
        technology. The matrix operator is simply the identity:

        >>> assert constraint.capacity.data == np.array(1)
        >>> assert len(constraint.capacity.dims) == 0

        And the upperbound is expanded over the replacement technologies,
        but not over the assets. Hence the assets will be summed over in the final
        constraint:

        >>> assert set(constraint.b.dims) == {"replacement"}
        >>> assert constraint.kind == cs.ConstraintKind.UPPER_BOUND

        As shown above, it does not bind the production decision variables. Hence,
        production is zero. The matrix operator for the capacity is simply the identity.
        The upper bound is simply the maximum for replacement technology (and region, if
        that particular dimension exists in the problem).

        The lp problem then becomes:

        >>> technologies = res.technologies.interp(year=market.year.min() + 5)
        >>> inputs = cs.ScipyAdapter.factory(
        ...     technologies, costs, constraint
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

        >>> lpcosts = cs.lp_costs(technologies, costs)
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
    bounds: tuple[float | None, float | None] = (0, np.inf)
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None

    @classmethod
    def factory(
        cls,
        costs: xr.DataArray,
        constraints: list[Constraint],
        commodities: list[str],
        timeslice_level: str | None = None,
    ) -> ScipyAdapter:
        # Calculate costs for the linear problem
        lpcosts = lp_costs(
            capacity_costs=costs,
            commodities=commodities,
            timeslice_level=timeslice_level,
        )

        # Create dataset from costs and constraints
        data = cls._unified_dataset(lpcosts, *constraints)

        # Get capacity constraint matrix / costs
        capacities = cls._selected_quantity(data, "capacity")

        # Get production constraint matrix / costs
        productions = cls._selected_quantity(data, "production")

        # Get constraint vector
        bs = cls._selected_quantity(data, "b")

        # Prepare scipy adapter from constraints
        kwargs = cls._to_scipy_adapter(capacities, productions, bs, *constraints)

        def to_muse(x: np.ndarray) -> xr.Dataset:
            return ScipyAdapter._back_to_muse(
                x,
                capacity_template=capacities.costs,
                production_template=productions.costs,
            )

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
    def _unified_dataset(lpcosts: xr.Dataset, *constraints: Constraint) -> xr.Dataset:
        """Creates single xr.Dataset from costs and constraints."""
        from xarray import merge

        # Reformat constraints to lp format
        lp_constraints = [
            lp_constraint(constraint, lpcosts) for constraint in constraints
        ]

        # Rename variables in lp constraints
        lp_constraints = [
            constraint.rename(
                b=f"b{i}", capacity=f"capacity{i}", production=f"production{i}"
            )
            for i, constraint in enumerate(lp_constraints)
        ]

        # Rename dimensions in lpcosts
        lpcosts = lpcosts.rename({k: f"d({k})" for k in lpcosts.dims})

        # Merge data
        data = merge([lpcosts, *lp_constraints])

        # An adjustment is required for lower bound constraints
        for i, constraint in enumerate(constraints):
            if constraint.kind == ConstraintKind.LOWER_BOUND:
                data[f"b{i}"] = -data[f"b{i}"]
                data[f"capacity{i}"] = -data[f"capacity{i}"]
                data[f"production{i}"] = -data[f"production{i}"]

        # Enusure consistent ordering of dimensions
        return data.transpose(*data.dims)

    @staticmethod
    def _selected_quantity(data: xr.Dataset, name: str) -> xr.Dataset:
        # Select data for the specified quantity ("capacity", "production", or "b")
        result = cast(
            xr.Dataset, data[[u for u in data.data_vars if str(u).startswith(name)]]
        )

        # Rename variables ("costs" for the costs variable, 0/1/2 etc. for constraints)
        return result.rename(
            {
                k: ("costs" if k == name else int(str(k).replace(name, "")))
                for k in result.data_vars
            }
        )

    @staticmethod
    def _to_scipy_adapter(
        capacities: xr.Dataset, productions: xr.Dataset, bs: xr.Dataset, *constraints
    ):
        """Converts constraints to scipy format.

        The constraints are converted to a format that can be used by scipy's linear
        programming solver. The constraints are converted to a 2D matrix of constraints
        vs decision variables. The constraints are then converted to a dictionary that
        can be used by scipy's linear programming solver.

        Args:
            capacities: Dataset with decision variables for capacity constraints.
            productions: Dataset with decision variables for production constraints.
            bs: Dataset with constraints.
            *constraints: List of constraints.

        Returns:
            Dictionary with constraints in a format that can be used by scipy's linear
            programming solver.
        """

        def reshape(matrix: xr.DataArray) -> np.ndarray:
            """Convert constraints matrix to a 2D np array.

            The rows of the constraaints matrix will represent the constraints, and the
            columns will represent the decision variables.
            """
            # Before building LP we need to sort dimensions for consistency
            if list(matrix.dims) != sorted(matrix.dims):
                matrix = matrix.transpose(*sorted(matrix.dims))

            # Size of the first dimension
            # This dimension represents the number of constraints
            size = np.prod(
                [matrix[u].shape[0] for u in matrix.dims if str(u).startswith("c")]
            )

            # Reshape into a 2D array: N constraints x N decision variables
            return matrix.values.reshape((size, -1))

        def extract_bA(constraints, *kinds: ConstraintKind):
            """Extracts A and b for constraints of specified kinds."""
            # Get indices of constraints of the specified kind
            indices = [i for i in range(len(bs)) if constraints[i].kind in kinds]

            # Convert constraints matrices to 2d np arrays
            capa_constraints = [reshape(capacities[i]) for i in indices]
            prod_constraints = [reshape(productions[i]) for i in indices]

            # Convert constraints vectors to 1d
            constraints_vectors = [
                bs[i].stack(constraint=sorted(bs[i].dims)) for i in indices
            ]

            # Concatenate constraints
            if capa_constraints:
                A = np.concatenate(
                    (
                        np.concatenate(capa_constraints, axis=0),
                        np.concatenate(prod_constraints, axis=0),
                    ),
                    axis=1,
                )
                b = np.concatenate(constraints_vectors, axis=0)
            else:
                # If there are no constraints of the given kind, return None
                A = None
                b = None
            return A, b

        # Create costs vector by concatenating capacity and production costs
        c = np.concatenate(
            (
                capacities["costs"].values.flatten(),
                productions["costs"].values.flatten(),
            ),
            axis=0,
        )

        # Extract A and b for inequality constraints
        A_ub, b_ub = extract_bA(
            constraints, ConstraintKind.UPPER_BOUND, ConstraintKind.LOWER_BOUND
        )

        # Extract A and b for equality constraints
        A_eq, b_eq = extract_bA(constraints, ConstraintKind.EQUALITY)

        # Prepare scipy adapter
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
        x: np.ndarray, template: xr.DataArray | xr.Dataset
    ) -> xr.DataArray:
        """Convert a vector of decision variables to a DataArray.

        Args:
            x: 1D vector of decision variables, outputted from the scipy solver.
            template: Template for the decision variables. This may be for either
                capacity or production variables.
        """
        # First create a multidimensional dataarray based on the template
        result = xr.DataArray(
            x.reshape(template.shape), coords=template.coords, dims=template.dims
        )

        # Then rename the dimensions (e.g. "d(asset)" -> "asset")
        return result.rename({k: str(k)[2:-1] for k in result.dims})

    @staticmethod
    def _back_to_muse(
        x: np.ndarray,
        capacity_template: xr.DataArray,
        production_template: xr.DataArray,
    ) -> xr.Dataset:
        """Convert the full set of decision variables to a Dataset.

        This must have capacity variables first, followed by production variables.

        Args:
            x: 1D vector of decision variables, outputted from the scipy solver.
            capacity_template: Template for the capacity decision variables.
            production_template: Template for the production decision variables.
        """
        n_capa = capacity_template.size  # number of capacity decision variables

        capa = ScipyAdapter._back_to_muse_quantity(
            x[:n_capa], template=capacity_template
        )
        prod = ScipyAdapter._back_to_muse_quantity(
            x[n_capa:], template=production_template
        )
        return xr.Dataset({"capacity": capa, "production": prod})
