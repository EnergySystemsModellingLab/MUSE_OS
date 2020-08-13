"""Investment decision.

An investment determines which technologies to invest given a metric to
determine preferred technologies, a corresponding search space of technologies,
and the demand to fulfill.

Investments should be registered via the decorator `register_investment`. The
registration makes it possible to call investments dynamically through
`compute_investment`, by specifying the name of the investment. It is part of
MUSE's plugin platform.

Investments are not expected to modify any of their arguments. They should all
have the following signature:

.. code-block:: python

    @register_investment
    def investment(
        costs: DataArray,
        search_space: DataArray,
        technologies: Dataset,
        constraints: List[Constraint],
        year: int,
        **kwargs
    ) -> DataArray:
        pass

Arguments:
    costs: specifies for each `asset` which `replacement` technology should be invested
        in preferentially. This should be an integer or floating point array with
        dimensions `asset` and `replacement`.
    search_space: an `asset` by `replacement` matrix defining allowed and disallowed
        replacement technologies for each asset
    technologies: a dataset containing all constant data characterizing the
        technologies.
    constraints: a list of constraints as defined in :py:mod:`~muse.constraints`.
    year: the current year.

Returns:
    A data array with dimensions `asset` and `technology` specifying the amount
    of newly invested capacity.
"""
__all__ = [
    "adhoc_match_demand",
    "cliff_retirement_profile",
    "register_investment",
    "INVESTMENT_SIGNATURE",
]
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Text,
    Union,
    cast,
)

import numpy as np
from mypy_extensions import KwArg
from xarray import DataArray, Dataset

from muse.constraints import Constraint
from muse.registration import registrator

INVESTMENT_SIGNATURE = Callable[
    [DataArray, DataArray, Dataset, List[Constraint], KwArg(Any)], DataArray
]
"""Investment signature. """

INVESTMENTS: MutableMapping[Text, INVESTMENT_SIGNATURE] = {}
"""Dictionary of investment functions."""


@registrator(registry=INVESTMENTS, loglevel="info")
def register_investment(function: INVESTMENT_SIGNATURE) -> INVESTMENT_SIGNATURE:
    """Decorator to register a function as an investment."""
    from functools import wraps

    @wraps(function)
    def decorated(
        costs: DataArray,
        search_space: DataArray,
        technologies: Dataset,
        constraints: List[Constraint],
        **kwargs,
    ) -> DataArray:
        result = function(costs, search_space, technologies, constraints, **kwargs)
        return result.rename("investment")

    return decorated


def factory(settings: Optional[Union[Text, Mapping]] = None) -> Callable:
    from typing import Dict

    if settings is None:
        name = "match_demand"
        params: Dict = {}
    elif isinstance(settings, Text):
        name = settings
        params = {}
    else:
        name = settings["name"]
        params = {k: v for k, v in settings.items() if k != "name"}

    top = params.get("timeslice_op", "max")
    if isinstance(top, Text):
        if top.lower() == "max":

            def timeslice_op(x: DataArray) -> DataArray:
                from muse.timeslices import convert_timeslice

                return (x / convert_timeslice(DataArray(1), x)).max("timeslice")

        elif top.lower() == "sum":

            def timeslice_op(x: DataArray) -> DataArray:
                return x.sum("timeslice")

        else:
            raise ValueError(f"Unknown timeslice transform {top}")

        params["timeslice_op"] = timeslice_op

    investment = INVESTMENTS[name]

    def compute_investment(
        search: Dataset, technologies: Dataset, constraints: List[Constraint], **kwargs
    ) -> DataArray:
        """Computes investment needed to fulfill demand.

        The return is a data array with two dimensions: (asset, replacement).
        """
        from numpy import zeros

        if any(u == 0 for u in search.decision.shape):
            return DataArray(
                zeros((len(search.asset), len(search.replacement))),
                coords={"asset": search.asset, "replacement": search.replacement},
                dims=("asset", "replacement"),
            )

        return investment(
            search.decision,
            search.search_space,
            technologies,
            constraints,
            **params,
            **kwargs,
        ).rename("investment")

    return compute_investment


def cliff_retirement_profile(
    technical_life: DataArray,
    current_year: int = 0,
    protected: int = 0,
    interpolation: Text = "linear",
    **kwargs,
) -> DataArray:
    """Cliff-like retirement profile from current year.

    Computes the retirement profile of all technologies in ``technical_life``.
    Assets with a technical life smaller than the input time-period should automatically
    be renewed.

    Hence, if ``technical_life <= protected``, then effectively, the technical life is
    rewritten as ``technical_life * n`` with ``n = int(protected // technical_life) +
    1``.

    We could just return an array where each year is repesented. Instead, to save
    memory, we return a compact view of the same where years where no change happens are
    removed.

    Arguments:
        technical_life: lifetimes for each technology
        current_year: current year
        protected: The technologies are assumed to be renewed between years
            `current_year` and `current_year + protected`
        **kwargs: arguments by which to filter technical_life, if any.

    Returns:
        A boolean DataArray where each each element along the year dimension is
        true if the technology is still not retired for the given year.
    """
    from xarray import DataArray
    from muse.utilities import avoid_repetitions

    if kwargs:
        technical_life = technical_life.sel(**kwargs)
    if "year" in technical_life.dims:
        technical_life = technical_life.interp(year=current_year, method=interpolation)
    technical_life = (1 + protected // technical_life) * technical_life  # type:ignore

    if len(technical_life) > 0:
        max_year = int(current_year + technical_life.max())
    else:
        max_year = int(current_year + protected)
    allyears = DataArray(
        range(current_year, max_year + 1),
        dims="year",
        coords={"year": range(current_year, max_year + 1)},
    )
    profile = allyears < (current_year + technical_life)  # type: ignore

    # now we minimize the number of years needed to represent the profile fully
    # this is done by removing the central year of any three repeating year, ensuring
    # the removed year can be recovered by a linear interpolation.
    goodyears = avoid_repetitions(profile.astype(int))

    return profile.sel(year=goodyears).astype(bool)


class LinearProblemError(RuntimeError):
    """Error returned for infeasible LP problems."""

    def __init__(self, *args):
        super().__init__(*args)


@register_investment(name=["adhoc"])
def adhoc_match_demand(
    costs: DataArray,
    search_space: DataArray,
    technologies: Dataset,
    constraints: List[Constraint],
    year: int,
    timeslice_op: Optional[Callable[[DataArray], DataArray]] = None,
) -> DataArray:
    from muse.demand_matching import demand_matching
    from muse.quantities import maximum_production, capacity_in_use
    from muse.timeslices import convert_timeslice, QuantityType

    demand = next((c for c in constraints if c.name == "demand")).b
    max_capacity = next(
        (c for c in constraints if c.name == "max capacity expansion")
    ).b
    max_prod = maximum_production(
        technologies,
        max_capacity,
        year=year,
        technology=costs.replacement,
        commodity=demand.commodity,
    ).drop_vars("technology")
    if "timeslice" in demand.dims and "timeslice" not in max_prod.dims:
        max_prod = convert_timeslice(max_prod, demand, QuantityType.EXTENSIVE)

    # Push disabled techs to last rank.
    # Any production assigned to them by the demand-matching algorithm will be removed.
    minobj = costs.min()
    maxobj = costs.where(search_space, minobj).max("replacement") + 1
    decision = costs.where(search_space, maxobj)

    production = demand_matching(
        demand.sel(asset=demand.asset.isin(search_space.asset)), decision, max_prod
    ).where(search_space, 0)

    capacity = capacity_in_use(
        production, technologies, year=year, technology=production.replacement
    ).drop_vars("technology")
    if "timeslice" in capacity.dims and timeslice_op is not None:
        capacity = timeslice_op(capacity)
    return capacity.rename("investment")


@register_investment(name=["scipy", "match_demand"])
def scipy_match_demand(
    costs: DataArray,
    search_space: DataArray,
    technologies: Dataset,
    constraints: List[Constraint],
    year: Optional[int] = None,
    timeslice_op: Optional[Callable[[DataArray], DataArray]] = None,
    **options,
) -> DataArray:
    from muse.constraints import ScipyAdapter
    from scipy.optimize import linprog
    from logging import getLogger

    if "timeslice" in costs.dims and timeslice_op is not None:
        costs = timeslice_op(costs)
    if "year" in technologies.dims and year is None:
        raise ValueError("Missing year argument")
    elif "year" in technologies.dims:
        techs = technologies.interp(year=year).drop_vars("year")
    else:
        techs = technologies
    timeslice = next((cs.timeslice for cs in constraints if "timeslice" in cs.dims))
    adapter = ScipyAdapter.factory(
        techs, -cast(np.ndarray, costs), timeslice, *constraints
    )
    res = linprog(**adapter.kwargs, options=dict(disp=True))
    if not res.success:
        getLogger(__name__).critical(res.message)
        raise LinearProblemError("LP system could not be solved", res)

    solution = cast(Callable[[np.ndarray], Dataset], adapter.to_muse)(res.x)
    return solution.capacity


@register_investment(name=["cvxopt"])
def cvxopt_match_demand(
    costs: DataArray,
    search_space: DataArray,
    technologies: Dataset,
    constraints: List[Constraint],
    year: Optional[int] = None,
    timeslice_op: Optional[Callable[[DataArray], DataArray]] = None,
    **options,
) -> DataArray:
    from muse.constraints import ScipyAdapter
    from logging import getLogger

    if "year" in technologies.dims and year is None:
        raise ValueError("Missing year argument")
    elif "year" in technologies.dims:
        techs = technologies.interp(year=year).drop_vars("year")
    else:
        techs = technologies

    def default_to_scipy():
        return scipy_match_demand(
            costs, search_space, techs, constraints, timeslice_op=timeslice_op
        )

    try:
        from cvxopt import matrix, solvers
    except ImportError:
        msg = (
            "cvxopt is not installed\n"
            "It can be installed with `pip install cvxopt`\n"
            "Using the scipy linear solver instead."
        )
        getLogger(__name__).critical(msg)
        return default_to_scipy()

    if "timeslice" in costs.dims and timeslice_op is not None:
        costs = timeslice_op(costs)
    timeslice = next((cs.timeslice for cs in constraints if "timeslice" in cs.dims))
    adapter = ScipyAdapter.factory(
        techs, -cast(np.ndarray, costs), timeslice, *constraints
    )
    G = np.zeros((0, adapter.c.size)) if adapter.A_ub is None else adapter.A_ub
    h = np.zeros((0,)) if adapter.b_ub is None else adapter.b_ub
    if adapter.bounds[0] != -np.inf and adapter.bounds[0] is not None:
        G = np.concatenate((G, -np.eye(adapter.c.size)))
        h = np.concatenate((h, np.full(adapter.c.size, adapter.bounds[0])))
    if adapter.bounds[1] != np.inf and adapter.bounds[1] is not None:
        G = np.concatenate((G, np.eye(adapter.c.size)))
        h = np.concatenate((h, np.full(adapter.c.size, adapter.bounds[1])))

    args = [adapter.c, G, h]
    if adapter.A_eq is not None:
        args += [adapter.A_eq, adapter.b_eq]
    res = solvers.lp(*map(matrix, args), **options)
    if res["status"] != "optimal":
        getLogger(__name__).info(res["status"])
    if res["x"] is None:
        getLogger(__name__).critical("infeasible system")
        raise LinearProblemError("Infeasible system", res)

    solution = cast(Callable[[np.ndarray], Dataset], adapter.to_muse)(list(res["x"]))
    return solution.capacity
