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
from muse.timeslices import drop_timeslice

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
    [xr.DataArray, xr.Dataset, xr.DataArray, xr.Dataset, xr.Dataset, KwArg(Any)],
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
        assets: xr.Dataset,
        search_space: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset,
        **kwargs,
    ) -> Constraint | None:
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
        assets: xr.Dataset,
        search_space: xr.DataArray,
        market: xr.Dataset,
        technologies: xr.Dataset,
        year: int | None = None,
    ) -> list[Constraint]:
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
    year: int | None = None,
    forecast: int | None = None,
    interpolation: str = "linear",
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
    """
    from muse.utilities import filter_input, reduce_assets

    if year is None:
        year = int(market.year.min())
    if forecast is None and len(getattr(market, "year", [])) <= 1:
        forecast = 5
    elif forecast is None:
        forecast = next(int(u) for u in sorted(market.year - year) if u > 0)
    forecast_year = year + forecast

    capacity = (
        reduce_assets(
            assets.capacity,
            coords={"technology", "region"}.intersection(assets.capacity.coords),
        )
        .interp(year=[year, forecast_year], method=interpolation)
        .ffill("year")
    )
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
    techs = filter_input(
        technologies[
            ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
        ],
        technology=replacement,
        year=year,
    ).drop_vars("technology")
    regions = getattr(capacity, "region", None)
    if regions is not None and "region" in technologies.dims:
        techs = techs.sel(region=regions)

    add_cap = techs.max_capacity_addition * forecast

    limit = techs.total_capacity_limit
    forecasted = capacity.sel(year=forecast_year, drop=True)
    total_cap = (limit - forecasted).clip(min=0).rename("total_cap")

    max_growth = techs.max_capacity_growth
    initial = capacity.sel(year=year, drop=True)

    growth_cap = initial * (max_growth * forecast + 1) - forecasted
    growth_cap = growth_cap.where(growth_cap > 0, total_cap)

    zero_cap = add_cap.where(add_cap < total_cap, total_cap)
    with_growth = zero_cap.where(zero_cap < growth_cap, growth_cap)
    b = with_growth.where(initial > 0, zero_cap)

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
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: int | None = None,
    forecast: int = 5,
    interpolation: str = "linear",
) -> Constraint:
    """Constraints production to meet demand."""
    from muse.commodities import is_enduse

    enduse = technologies.commodity.sel(commodity=is_enduse(technologies.comm_usage))
    b = demand.sel(commodity=demand.commodity.isin(enduse))
    if "region" in b.dims and "dst_region" in assets.dims:
        b = b.rename(region="dst_region")
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
    year: int | None = None,
    forecast: int = 5,
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
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: int | None = None,
) -> Constraint:
    """Constructs constraint between capacity and maximum production.

    Constrains the production decision variable by the maximum production for a given
    capacity.
    """
    from xarray import ones_like, zeros_like

    from muse.commodities import is_enduse
    from muse.timeslices import QuantityType, convert_timeslice

    if year is None:
        year = int(market.year.min())
    commodities = technologies.commodity.sel(
        commodity=is_enduse(technologies.comm_usage)
    )
    replacement = search_space.replacement
    replacement = replacement.drop_vars(
        [u for u in replacement.coords if u not in replacement.dims]
    )
    kwargs = dict(technology=replacement, year=year, commodity=commodities)
    if "region" in search_space.coords and "region" in technologies.dims:
        kwargs["region"] = search_space.region
    techs = (
        technologies[["fixed_outputs", "utilization_factor"]]
        .sel(**kwargs)
        .drop_vars("technology")
    )
    capacity = convert_timeslice(
        techs.fixed_outputs * techs.utilization_factor,
        market.timeslice,
        QuantityType.EXTENSIVE,
    )
    if "asset" not in capacity.dims and "asset" in search_space.dims:
        capacity = capacity.expand_dims(asset=search_space.asset)
    production = ones_like(capacity)
    b = zeros_like(production)
    # Include maxaddition constraint in max production to match region-dst_region
    if "dst_region" in assets.dims:
        b = b.expand_dims(dst_region=assets.dst_region)
        capacity = capacity.rename(region="src_region")
        production = production.rename(region="src_region")
        maxadd = technologies.max_capacity_addition.rename(region="src_region")
        if "year" in maxadd.dims:
            maxadd = maxadd.sel(year=year)

        maxadd = maxadd.rename(technology="replacement")
        maxadd = maxadd.where(maxadd == 0, 0.0)
        maxadd = maxadd.where(maxadd > 0, -1.0)
        capacity = capacity * maxadd
        production = production * maxadd
        b = b.rename(region="src_region")
    return xr.Dataset(
        dict(capacity=-cast(np.ndarray, capacity), production=production, b=b),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND),
    )


@register_constraints
def demand_limiting_capacity(
    demand_: xr.DataArray,
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: int | None = None,
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
        demand_, assets, search_space, market, technologies, year=year
    )
    demand_constraint = demand(
        demand_, assets, search_space, market, technologies, year=year
    )

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
        ts = ratio.timeslice.isel(
            timeslice=ratio.min("replacement").argmax("timeslice")
        )
        # We select this timeslice for each array - don't trust the indices:
        # search for the right timeslice in the array and select it.
        b = (
            b.isel(timeslice=(b.timeslice == ts).argmax("timeslice"))
            if "timeslice" in b.dims
            else b
        )
        capacity = (
            capacity.isel(timeslice=(capacity.timeslice == ts).argmax("timeslice"))
            if "timeslice" in capacity.dims
            else capacity
        )

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
    assets: xr.Dataset,
    search_space: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    year: int | None = None,
) -> Constraint | None:
    """Constructs constraint between capacity and minimum service."""
    from xarray import ones_like, zeros_like

    from muse.commodities import is_enduse
    from muse.timeslices import QuantityType, convert_timeslice

    if "minimum_service_factor" not in technologies.data_vars:
        return None
    if np.all(technologies["minimum_service_factor"] == 0):
        return None
    if year is None:
        year = int(market.year.min())
    commodities = technologies.commodity.sel(
        commodity=is_enduse(technologies.comm_usage)
    )
    replacement = search_space.replacement
    replacement = replacement.drop_vars(
        [u for u in replacement.coords if u not in replacement.dims]
    )
    kwargs = dict(technology=replacement, year=year, commodity=commodities)
    if "region" in search_space.coords and "region" in technologies.dims:
        kwargs["region"] = assets.region
    techs = (
        technologies[["fixed_outputs", "utilization_factor", "minimum_service_factor"]]
        .sel(**kwargs)
        .drop_vars("technology")
    )
    capacity = convert_timeslice(
        techs.fixed_outputs * techs.minimum_service_factor,
        market.timeslice,
        QuantityType.EXTENSIVE,
    )
    if "asset" not in capacity.dims:
        capacity = capacity.expand_dims(asset=search_space.asset)
    production = ones_like(capacity)
    b = zeros_like(production)
    return xr.Dataset(
        dict(capacity=-cast(np.ndarray, capacity), production=production, b=b),
        attrs=dict(kind=ConstraintKind.LOWER_BOUND),
    )


def lp_costs(
    technologies: xr.Dataset, costs: xr.DataArray, timeslices: xr.DataArray
) -> xr.Dataset:
    """Creates costs for solving with scipy's LP solver.

    Example:
        We can now construct example inputs to the function from the sample model. The
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
        ...     technologies.sel(year=2020, region="R1"), costs, timeslices
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

    ts_costs = convert_timeslice(costs, timeslices)
    selection = dict(
        commodity=is_enduse(technologies.comm_usage),
        technology=technologies.technology.isin(costs.replacement),
    )

    if "region" in technologies.fixed_outputs.dims and "region" in ts_costs.coords:
        selection["region"] = ts_costs.region
    fouts = technologies.fixed_outputs.sel(selection).rename(technology="replacement")

    # lpcosts.dims = Frozen({'asset': 2,
    #                   'replacement': 2,
    #                   'timeslice': 3,
    #                   'commodity': 1})
    # muse38: lpcosts.dims = Frozen({'asset': 2, ,
    #                                'commodity': 1
    #                                'replacement': 2,
    #                                'timeslice': 3})
    production = zeros_like(ts_costs * fouts)
    for dim in production.dims:
        if isinstance(production.get_index(dim), pd.MultiIndex):
            production = drop_timeslice(production)
            production[dim] = pd.Index(production.get_index(dim), tupleize_cols=False)

    return xr.Dataset(dict(capacity=costs, production=production))


def merge_lp(
    costs: xr.Dataset, *constraints: Constraint
) -> tuple[xr.Dataset, list[Constraint]]:
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
            constraint = drop_timeslice(constraint)
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
         >>> assets = next(a.assets for a in res.agents)
         >>> demand = None # not used in max production
         >>> constraint = cs.max_production(demand, assets, search, market,
         ...                                technologies) # noqa: E501
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

    result = constraint.sum(set(constraint.dims) - set(lpcosts.dims) - set(b.dims))

    result = result.rename(
        {k: f"d({k})" for k in set(result.dims).intersection(lpcosts.dims)}
    )
    result = result.rename(
        {k: f"c({k})" for k in set(result.dims).intersection(b.dims)}
    )
    expand = set(lpcosts.dims) - set(constraint.dims) - set(b.dims)

    if expand == {"timeslice", "asset", "commodity"}:
        expand = ["asset", "timeslice", "commodity"]

    result = result.expand_dims(
        {f"d({k})": lpcosts[k].rename({k: f"d({k})"}).set_index() for k in expand}
    )
    expand = set(b.dims) - set(constraint.dims) - set(lpcosts.dims)

    result = result.expand_dims(
        {f"c({k})": b[k].rename({k: f"c({k})"}).set_index() for k in expand}
    )

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
        >>> from muse.timeslices import convert_timeslice
        >>> from muse import constraints as cs
        >>> res = examples.sector("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> search = examples.search_space("residential", model="medium")
        >>> assets = next(a.assets for a in res.agents)
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

        And the upperbound is expanded over the replacement technologies,
        but not over the assets. Hence the assets will be summed over in the final
        constraint:

        >>> assert (constraint.b.data == np.array([50.0, 12.0, 12.0, 50.0 ])).all()
        >>> assert set(constraint.b.dims) == {"replacement"}
        >>> assert constraint.kind == cs.ConstraintKind.UPPER_BOUND

        As shown above, it does not bind the production decision variables. Hence,
        production is zero. The matrix operator for the capacity is simply the identity.
        Hence it can be inputted as the dimensionless scalar 1. The upper bound is
        simply the maximum for replacement technology (and region, if that particular
        dimension exists in the problem).

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
    bounds: tuple[float | None, float | None] = (0, np.inf)
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None

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

        capacities = cls._selected_quantity(data, "capacity")

        productions = cls._selected_quantity(data, "production")

        bs = cls._selected_quantity(data, "b")

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
        """Creates single xr.Dataset from costs and constraints."""
        from xarray import merge

        assert "year" not in technologies.dims

        coords = sorted([k for k in lpcosts.dims])
        lpcosts_df = lpcosts.to_dataframe().reset_index().set_index(coords)
        slpcosts = lpcosts_df.to_xarray()  # sorted lpcosts.dims

        data = merge(
            [lpcosts.rename({k: f"d({k})" for k in slpcosts.dims})]
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

        return data.transpose(*data.dims)

    @staticmethod
    def _selected_quantity(data: xr.Dataset, name: str) -> xr.Dataset:
        result = cast(
            xr.Dataset, data[[u for u in data.data_vars if str(u).startswith(name)]]
        )

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
        def reshape(matrix: xr.DataArray) -> np.ndarray:
            if list(matrix.dims) != sorted(matrix.dims):
                new_dims = sorted(matrix.dims)
                matrix = matrix.transpose(*new_dims)

            # before building LP we need to sort dimensions for consistency

            size = np.prod(
                [matrix[u].shape[0] for u in matrix.dims if str(u).startswith("c")]
            )

            return matrix.values.reshape((size, -1))

        def extract_bA(constraints, *kinds):
            indices = [i for i in range(len(bs)) if constraints[i].kind in kinds]
            capa_constraints = [reshape(capacities[i]) for i in indices]
            prod_constraints = [reshape(productions[i]) for i in indices]
            if capa_constraints:
                A: np.ndarray | None = np.concatenate(
                    (
                        np.concatenate(capa_constraints, axis=0),
                        np.concatenate(prod_constraints, axis=0),
                    ),
                    axis=1,
                )
                b: np.ndarray | None = np.concatenate(
                    [bs[i].stack(constraint=sorted(bs[i].dims)) for i in indices],
                    axis=0,
                )
            else:
                A = None
                b = None
            return A, b

        c = np.concatenate(
            (
                cast(np.ndarray, capacities["costs"].values).flatten(),
                cast(np.ndarray, productions["costs"].values).flatten(),
            ),
            axis=0,
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
        x: np.ndarray, template: xr.DataArray | xr.Dataset
    ) -> xr.DataArray:
        result = xr.DataArray(
            x.reshape(template.shape), coords=template.coords, dims=template.dims
        )
        return result.rename({k: str(k)[2:-1] for k in result.dims})

    @staticmethod
    def _back_to_muse(
        x: np.ndarray, capacity: xr.DataArray, production: xr.DataArray
    ) -> xr.Dataset:
        capa = ScipyAdapter._back_to_muse_quantity(x[: capacity.size], capacity)
        prod = ScipyAdapter._back_to_muse_quantity(x[capacity.size :], production)
        return xr.Dataset({"capacity": capa, "production": prod})
