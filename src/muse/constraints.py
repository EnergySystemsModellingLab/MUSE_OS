"""Investment constraints.

Constraints on investements ensure that investements match some given criteria. For
instance, the constraints could ensure that only so much of a new asset can be built
every year.

Functions to compute constraints should be registered via the decorator
:py:func:`register_constraints`. This registration step makes it possible for
constraints to be declared in the TOML file.
"""

from enum import Enum, auto, unique
from typing import Callable, List, Mapping, MutableMapping, Sequence, Text, Union

from xarray import DataArray, Dataset

from muse.registration import registrator

CAPACITY_DIMS = "asset", "replacement", "region"
"""Default dimensions for capacity decision variables."""
PRODUCT_DIMS = "commodity", "timeslice", "region"
"""Default dimensions for product decision variables."""


@unique
class ConstraintKind(Enum):
    EQUALITY = auto()
    UPPER_BOUND = auto()
    LOWER_BOUND = auto()


class DecisionVariable(Enum):
    """Variables over wich the constraints act."""

    PRODUCTION = auto()
    CAPACITY = auto()


Constraint = Dataset
"""An investment constraint :math:`A * x ~ b`

Where :math:`~` is one of :math:`=,\\leq,\\geq`.

A constraint *must* containing a data-array `b` corresponding to right-hand-side vector
of the contraint. It *must* also contain a data-array `A` corresponding to the
left-hand-side matrix operator. `A` may be None, in which case it is equivalent to the
identity. It *must* also contain an attribute `kind` of type :py:class:`ConstraintKind`
defining the operation.

Note that registered function :py:func:`register_constraints` will automatically convert
a data-array to a constraint where `A` is None.
"""


CONSTRAINT_SIGNATURE = Callable[[Dataset, DataArray, Dataset, Dataset], Constraint]
"""Basic signature for functions producing constraints."""
CONSTRAINTS: MutableMapping[Text, CONSTRAINT_SIGNATURE] = {}
"""Registry of constraint functions."""


@registrator(registry=CONSTRAINTS)
def register_constraints(function: CONSTRAINT_SIGNATURE) -> CONSTRAINT_SIGNATURE:
    from functools import wraps

    @wraps(function)
    def decorated(
        assets: Dataset,
        search_space: DataArray,
        market: Dataset,
        technologies: Dataset,
        **kwargs,
    ) -> Constraint:
        """Computes and standardizes a constraint."""
        constraint = function(  # type: ignore
            assets, search_space, market, technologies, **kwargs
        )
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

        return constraint

    return decorated


def factory(
    settings: Union[Text, Mapping, Sequence[Mapping]] = "max_capacity_expansion"
) -> Callable:
    if isinstance(settings, Text):
        names = [settings]
        params: List[Mapping] = [{}]
    elif isinstance(settings, Mapping):
        names = [settings["name"]]
        params = [{k: v for k, v in settings.items() if k != "name"}]

    def constraints(
        assets: Dataset,
        search_space: DataArray,
        technologies: Dataset,
        year: int,
        **kwargs,
    ) -> List[Constraint]:
        return [
            CONSTRAINTS[name](  # type: ignore
                assets, search_space, technologies, year=year, **{**param, **kwargs}
            )
            for name, param in zip(names, params)
        ]

    return constraints


@register_constraints
def max_capacity_expansion(
    assets: Dataset,
    search_space: DataArray,
    market: Dataset,
    technologies: Dataset,
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

    Example:

        >>> from muse import examples
        >>> from muse.constraints import max_capacity_expansion
        >>> res = examples.sector("residential", model="medium")
        >>> technologies = residential.technologies
        >>> market = examples.residential_market("medium")
        >>> search_space = examples.search_space("residential", model="medium")
        >>> assets = next(a.assets for a in res.agents if a.category == "retrofit")
        >>> maxcap = max_capacity_expansion(assets, search_space, market, technologies)
    """
    from muse.utilities import filter_input

    year = market.year.min()
    forecast_year = forecast + year

    techs = filter_input(
        technologies[
            ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
        ],
        technology=search_space.replacement,
        year=year,
    )
    assert isinstance(techs, Dataset)

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
    return Dataset(
        dict(b=constraint, capacity=1), attrs=dict(kind=ConstraintKind.UPPER_BOUND)
    )


@register_constraints
def demand(
    assets: Dataset,
    search_space: DataArray,
    market: Dataset,
    technologies: Dataset,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    """Constraints production to meet demand.

    Example:

        >>> from muse import examples
        >>> from muse import constraints
        >>> technologies = examples.technodata("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> search_space = None # Not used on demand
        >>> assets = None  # not used in demand
        >>> demand = constraints.demand(assets, search_space, market, technologies)
    """
    from muse.commodities import is_enduse

    b = market.consumption.sel(
        commodity=is_enduse(technologies.comm_usage.sel(commodity=market.commodity))
    )
    return Dataset(dict(b=b, production=1), attrs=dict(kind=ConstraintKind.EQUALITY))


def max_production(
    assets: Dataset,
    search_space: DataArray,
    market: Dataset,
    technologies: Dataset,
    forecast: int = 5,
    interpolation: Text = "linear",
) -> Constraint:
    """Constructs contraint between capacity and maximum production.

    Constrains the production decision variable by the maximum production for a given
    capacity.

    Example:

        >>> from muse import examples
        >>> from muse.constraints import max_production
        >>> technologies = examples.technodata("residential", model="medium")
        >>> market = examples.residential_market("medium")
        >>> search_space = examples.search_space("residential", "medium")
        >>> assets = None  # not used in max_production
        >>> maxprod = max_production(assets, search_space, market, technologies)
    """
    from xarray import zeros_like, ones_like
    from muse.commodities import is_enduse
    from muse.timeslices import convert_timeslice, QuantityType

    commodities = technologies.commodity.sel(
        commodity=is_enduse(technologies.comm_usage)
    )
    techs = technologies[["fixed_outputs", "utilization_factor"]].sel(
        year=market.year.min(),
        commodity=commodities,
        technology=search_space.replacement,
    )
    capacity = techs.fixed_outputs * techs.utilization_factor
    production = convert_timeslice(
        -ones_like(capacity), market.timeslice, QuantityType.EXTENSIVE
    )
    b = zeros_like(production)
    return Dataset(
        dict(capacity=capacity, production=production, b=b),
        attrs=dict(kind=ConstraintKind.UPPER_BOUND),
    )


def to_lp_costs(
    technologies: Dataset, costs: DataArray, timeslices: DataArray
) -> Dataset:
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

        >>> from muse.constraints import to_lp_costs
        >>> lpcosts = to_lp_costs(
        ...     technologies.sel(year=2020, region="USA"), costs, timeslices
        ... )
        >>> lpcosts
        <xarray.Dataset>
        Dimensions:      (asset: 4, commodity: 2, replacement: 4, timeslice: 6)
        Coordinates:
          * asset        (asset) object 'estove' 'gasboiler' 'gasstove' 'heatpump'
          * replacement  (replacement) object 'estove' 'gasboiler' 'gasstove' 'heatpump'
          * timeslice    (timeslice) MultiIndex
          - month        (timeslice) object 'all-year' 'all-year' ... 'all-year'
          - day          (timeslice) object 'all-week' 'all-week' ... 'all-week'
          - hour         (timeslice) object 'night' 'morning' ... 'late-peak' 'evening'
          * commodity    (commodity) object 'cook' 'heat'
            region       <U3 'USA'
            year         ... 2020
            comm_usage   (commodity) ...
        Data variables:
            capacity     (asset, replacement) int64 0 1 2 3 4 5 6 ... 10 11 12 13 14 15
            production   (timeslice, asset, replacement, commodity) float64 0.0 ... 0.0

        The capacity costs correspond exactly to the input costs:

        >>> assert (costs == lpcosts.capa_costs).all()

        They should correspond to a data-array with dimensions ``(asset, replacement)``
        (and possibly ``region`` as well).

        >>> lpcosts.capacity.dims
        ('asset', 'replacement')

        The production costs are zero by default. However, the production expands over
        not only the dimensions of the capacity, but also the ``timeslice``(s) during
        which production occurs and the ``commodity``(s) produced.

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
    return Dataset(dict(capacity=costs, production=production))


def fill_out_constraint(constraint: Constraint, costs: Dataset) -> Constraint:
    """Transforms the coordinates to LP coords.

    Example:


        >>> from muse import example
        >>> from muse.constraints import (
        ...     Constraint, ConstraintKind, DecisionVariable, to_lp_costs,
        ...     to_lp_constraint
        ... )
        >>> technologies = example.technodata("residential")
        >>> costs = xr.DataArray(
        ...     np.array(range(5, 5 + len(technology)**2)).reshape(len(technology), -1),
        ...     coords={
        ...         "asset": technology,
        ...         "replacement": technology,
        ...     },
        ...     dims=("asset", "replacement")
        ... )
        >>> x = to_lp_costs(technologies.commodity, costs)
        >>> constraint = Constraint(
        ...     b=xr.DataArray(
        ...         np.ones(len(technologies.technology)),
        ...         coords={"technology": technologies.technology},
        ...         dims="technology"
        ...     ),
        ...     A=None,
        ...     kind=ConstraintKind.UPPER_BOUND,
        ...     variable=DecisionVariable.CAPACITY,
        ... )

        >>> constraint = Constraint(
        ...     b=xr.DataArray(
        ...         range(1, len(technologies.technology) + 1),
        ...         coords={"replacement": technologies.technology.values},
        ...         dims="replacement"
        ...     ),
        ...     A=None,
        ...     kind=ConstraintKind.UPPER_BOUND,
        ...     variable=DecisionVariable.CAPACITY,
        ... )
    """
    pass
    # data = Dataset(dict(b=constraint.b, x=x))
    # if constraint.A is not None:
    #     data["A"] = constraint.A
    # transforms = dict(
    #     technology=x.replacement,
    #     **{str(k): v.values for k, v in x.coords.items() if k not in x.dims},
    # )
    # transforms = {k: v for k, v in transforms.items() if k in data.dims}
    # for initial, end in transforms.items():
    #     transform = DataArray(
    #         [t == end for t in data[initial].values],
    #         coords=dict([(initial, data[initial].values), ("x", x)]),
    #         dims=(initial, "x"),
    #     ).astype(int)
    #     data = (data * transform).sum(initial)


def to_lp_problem(
    technologies: Dataset, costs: DataArray, *constraints: Constraint
) -> Dataset:
    """Dataset with all parameters to scipy LP solver.

    Args:
        technology: Names the technologies that will become part of the decision
            variables.
        product: Names the products that will become part of the decision variables.
        cost: Cost associated with each technology. The cost of a product is zero.
        constraints: List of constraits on the decision variables.

    Example:

        Lets first construct the inputs to the funtion from the example model:

        >>> from muse.agents import example
        >>> technologies = example.technodata("residential", model="medium")
        >>> technology = technologies.technology.values
        >>> costs = xr.DataArray(
        ...     np.array(range(5, 5 + len(technology)**2)).reshape(len(technology), -1),
        ...     coords={
        ...         "asset": technology,
        ...         "replacement": technology,
        ...     },
        ...     dims=("asset", "replacement")
        ... )

        The function returns a dataset where each variable corresponds to an input for
        the ``scipy`` linear program solver:

        >>> from muse.agents.constraints import to_lp_problem
        >>> args = to_lp_problem(technologies, costs)
        >>> list(args.data_vars)
        ['c', 'bounds', 'Aeq', 'beq', 'Aub', 'bub']

        The costs ``c`` is a vector with dimensions matching the decision variables,
        i.e. capacity and production. The cost associated with production is zero, since
        this the probably models installing new capacity, rather than operation. See
        :py:func:`to_lp_costs`.

        >>> args.c
        <xarray.DataArray 'c' (x: 5)>
        array([5., 6., 7., 8., 0.])
        Coordinates:
          * x            (x) int64 0 1 2 3 4
            asset        (x) object 'gasboiler' 'gasboiler' 'heatpump' 'heatpump' None
            replacement  (x) object 'gasboiler' 'heatpump' 'gasboiler' 'heatpump' None
            product      (x) object None None None None 'heat'

        The bounds are between 0 and anything (``None``) for all decision variables, by
        default. Note that the dimensions correspond to the decision variables and the
        bounds themselves:

        >>> args.bounds
        <xarray.DataArray 'bounds' (bound: 2, x: 5)>
        array([[0, 0, 0, 0, 0],
               [None, None, None, None, None]], dtype=object)
        Coordinates:
          * x            (x) int64 0 1 2 3 4
            asset        (x) object 'gasboiler' 'gasboiler' 'heatpump' 'heatpump' None
            replacement  (x) object 'gasboiler' 'heatpump' 'gasboiler' 'heatpump' None
            product      (x) object None None None None 'heat'
          * bound        (bound) <U5 'lower' 'upper'

        By default, all other constraints are empty:

        >>> assert len(args.Aeq) == 0
        >>> assert len(args.beq) == 0
        >>> assert len(args.Aub) == 0
        >>> assert len(args.bub) == 0

        We can add a constraints. We'll start off with an upper bound on  capacity. Note
        that the input dimension is "technology". It implies a constraint across all
        assets simultaneously.

        >>> from muse.agents.constraints import (
        ...     Constraint, ConstraintKind, DecisionVariable
        ... )
        >>> from muse.agents.commodities import is_enduse
        >>> product = technologies.commodity.sel(
        ...     commodity=is_enduse(technologies.comm_usage)
        ... ).values
        >>> max_cap = Constraint(
        ...     b=xr.DataArray(
        ...         np.ones(len(technologies.technology)),
        ...         coords={"technology": technologies.technology},
        ...         dims="technology"
        ...     ),
        ...     A=None,
        ...     kind=ConstraintKind.UPPER_BOUND,
        ...     variable=DecisionVariable.CAPACITY,
        ... )
    """
    from numpy import zeros

    costs_lp = to_lp_costs(technologies.commodity, costs)

    result = Dataset()
    result["bounds"] = DataArray(
        [[0] * len(costs_lp.x), [None] * len(costs_lp.x)],
        coords=dict(bound=["lower", "upper"], x=costs_lp.x),
        dims=("bound", "x"),
    )

    result["Aeq"] = DataArray(
        zeros((0, len(result.x)), dtype=costs.dtype),
        coords=dict(constraint=("cons_eq", []), x=costs_lp.x),
        dims=("cons_eq", "x"),
    )
    result["beq"] = DataArray(
        zeros(0, dtype=costs.dtype),
        coords=dict(constraint=("cons_eq", [])),
        dims="cons_eq",
    )
    result["Aub"] = DataArray(
        zeros((0, len(result.x)), dtype=costs.dtype),
        coords=dict(constraint=("cons_ub", []), x=costs_lp.x),
        dims=("cons_ub", "x"),
    )
    result["bub"] = DataArray(
        zeros(0, dtype=costs.dtype),
        coords=dict(constraint=("cons_ub", [])),
        dims="cons_ub",
    )

    # standardized = [
    #     to_lp_constraint(constraint, x=costs_lp.x) for constraint in constraints
    # ]

    # eq_constraints = [c for c in constraints if c.kind == ConstraintKind.EQUALITY]

    return result


def standardize_quantities(
    technologies: Dataset,
    costs: DataArray,
    *constraints: Constraint,
    capacity_dims: Sequence[Text] = CAPACITY_DIMS,
    product_dims: Sequence[Text] = PRODUCT_DIMS,
) -> Dataset:
    """
    Example:

        >>> from muse.agents import example
        >>> from muse.agents.commodities import is_enduse
        >>> from muse.agents.constraints import ConstraintKind
        >>> from muse.agents.constraints import CAPACITY_DIMS as capacity_dims
        >>> from muse.agents.constraints import PRODUCT_DIMS as product_dims
        >>> technologies = example.technodata("residential")
        >>> technology = technologies.technology.values
        >>> costs = xr.DataArray(
        ...     np.array(range(5, 5 + len(technology)**2)).reshape(len(technology), -1),
        ...     coords={
        ...         "asset": technology,
        ...         "replacement": technology,
        ...     },
        ...     dims=("asset", "replacement")
        ... )
        >>> product = technologies.commodity.sel(
        ...     commodity=is_enduse(technologies.comm_usage)
        ... ).values

        >>> b = xr.DataArray(
        ...    np.arange(len(technologies.technology)),
        ...    coords={"technology": technologies.technology},
        ...    dims="technology"
        ... )
        >>> max_cap = Dataset(
        ...     dict(b=b, A=None), attrs=dict(kind=ConstraintKind.UPPER_BOUND)
        ... )
        >>> constraints = [max_cap]

    """
    from xarray import merge

    data = merge(
        [costs.rename("c")]
        + [
            constraint.rename(b=f"b_{i}", A=f"A_{i}").pipe(
                lambda x: x.rename(technology="replacement")
                if "technology" in x.dims
                else x
            )
            for i, constraint in enumerate(constraints)
        ]
    )
    data = data.stack(
        capacity=list(set(capacity_dims).intersection(data.dims)),
        production=list(set(product_dims).intersection(data.dims)),
    )
    return data
