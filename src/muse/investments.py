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
        costs: xr.DataArray,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        constraints: List[Constraint],
        year: int,
        **kwargs
    ) -> xr.DataArray:
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
    "INVESTMENT_SIGNATURE",
    "adhoc_match_demand",
    "cliff_retirement_profile",
    "register_investment",
]
from collections.abc import Mapping, MutableMapping
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import numpy as np
import xarray as xr
from mypy_extensions import KwArg

from muse.constraints import Constraint
from muse.errors import GrowthOfCapacityTooConstrained
from muse.outputs.cache import cache_quantity
from muse.registration import registrator
from muse.timeslices import timeslice_max

INVESTMENT_SIGNATURE = Callable[
    [xr.DataArray, xr.DataArray, xr.Dataset, list[Constraint], KwArg(Any)],
    Union[xr.DataArray, xr.Dataset],
]
"""Investment signature. """

INVESTMENTS: MutableMapping[str, INVESTMENT_SIGNATURE] = {}
"""Dictionary of investment functions."""


@registrator(registry=INVESTMENTS, loglevel="info")
def register_investment(function: INVESTMENT_SIGNATURE) -> INVESTMENT_SIGNATURE:
    """Decorator to register a function as an investment.

    The output of the function can be a DataArray, with the invested capacity, or a
    Dataset. In this case, it must contain a DataArray named "capacity" and, optionally,
    a DataArray named "production". Only the invested capacity DataArray is returned to
    the calling function.
    """
    from functools import wraps

    @wraps(function)
    def decorated(
        costs: xr.DataArray,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        constraints: list[Constraint],
        **kwargs,
    ) -> xr.DataArray:
        result = function(costs, search_space, technologies, constraints, **kwargs)

        if isinstance(result, xr.Dataset):
            investment = result["capacity"].rename("investment")
            if "production" in result:
                cache_quantity(production=result["production"])
        else:
            investment = result.rename("investment")

        cache_quantity(capacity=investment)

        return investment

    return decorated


def factory(settings: Optional[Union[str, Mapping]] = None) -> Callable:
    if settings is None:
        name = "match_demand"
        params: dict = {}
    elif isinstance(settings, str):
        name = settings
        params = {}
    else:
        name = settings["name"]
        params = {k: v for k, v in settings.items() if k != "name"}

    investment = INVESTMENTS[name]

    def compute_investment(
        search: xr.Dataset,
        technologies: xr.Dataset,
        constraints: list[Constraint],
        **kwargs,
    ) -> xr.DataArray:
        """Computes investment needed to fulfill demand.

        The return is a data array with two dimensions: (asset, replacement).
        """
        from numpy import zeros

        # Skip the investment step if no assets or replacements are available
        if any(u == 0 for u in search.decision.shape):
            return xr.DataArray(
                zeros((len(search.asset), len(search.replacement))),
                coords={"asset": search.asset, "replacement": search.replacement},
                dims=("asset", "replacement"),
            )

        # Otherwise, compute the investment
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
    technical_life: xr.DataArray,
    investment_year: int,
    interpolation: str = "linear",
    **kwargs,
) -> xr.DataArray:
    """Cliff-like retirement profile from current year.

    Computes the retirement profile of all technologies in ``technical_life``.
    Assets with a technical life smaller than the input time-period should automatically
    be renewed.

    We could just return an array where each year is represented. Instead, to save
    memory, we return a compact view of the same where years where no change happens are
    removed.

    Arguments:
        technical_life: lifetimes for each technology
        investment_year: The year in which the investment is made
        interpolation: Interpolation type
        **kwargs: arguments by which to filter technical_life, if any.

    Returns:
        A boolean DataArray where each each element along the year dimension is
        true if the technology is still not retired for the given year.
    """
    from muse.utilities import avoid_repetitions

    if kwargs:
        technical_life = technical_life.sel(**kwargs)
    if "year" in technical_life.dims:
        technical_life = technical_life.interp(
            year=investment_year, method=interpolation
        )

    # Create profile across all years
    if len(technical_life) > 0:
        max_year = int(investment_year + technical_life.max())
    else:
        max_year = investment_year
    allyears = xr.DataArray(
        range(investment_year, max_year + 1),
        dims="year",
        coords={"year": range(investment_year, max_year + 1)},
    )
    profile = allyears < (investment_year + technical_life)  # type: ignore

    # Minimize the number of years needed to represent the profile fully
    # This is done by removing the central year of any three repeating years, ensuring
    # the removed year can be recovered by linear interpolation.
    goodyears = avoid_repetitions(profile.astype(int))
    return profile.sel(year=goodyears).astype(bool)


class LinearProblemError(RuntimeError):
    """Error returned for infeasible LP problems."""

    def __init__(self, *args):
        super().__init__(*args)


@register_investment(name=["adhoc"])
def adhoc_match_demand(
    costs: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    constraints: list[Constraint],
    year: int,
    timeslice_level: Optional[str] = None,
) -> xr.DataArray:
    from muse.demand_matching import demand_matching
    from muse.quantities import capacity_in_use, maximum_production

    demand = next(c for c in constraints if c.name == "demand").b

    max_capacity = next(c for c in constraints if c.name == "max capacity expansion").b
    max_prod = maximum_production(
        technologies,
        max_capacity,
        year=year,
        technology=costs.replacement,
        commodity=demand.commodity,
        timeslice_level=timeslice_level,
    ).drop_vars("technology")

    # Push disabled techs to last rank.
    # Any production assigned to them by the demand-matching algorithm will be removed.
    minobj = costs.min()
    maxobj = costs.where(search_space, minobj).max("replacement") + 1

    decision = costs.where(search_space, maxobj)

    production = demand_matching(
        demand.sel(asset=demand.asset.isin(search_space.asset)),
        decision,
        max_prod,
    ).where(search_space, 0)

    capacity = capacity_in_use(
        production,
        technologies,
        year=year,
        technology=production.replacement,
        timeslice_level=timeslice_level,
    ).drop_vars("technology")
    if "timeslice" in capacity.dims:
        capacity = timeslice_max(capacity)

    result = xr.Dataset({"capacity": capacity, "production": production})
    return result


@register_investment(name=["scipy", "match_demand"])
def scipy_match_demand(
    costs: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    constraints: list[Constraint],
    year: Optional[int] = None,
    timeslice_level: Optional[str] = None,
    **options,
) -> xr.DataArray:
    from logging import getLogger

    from scipy.optimize import linprog

    from muse.constraints import ScipyAdapter

    if "timeslice" in costs.dims:
        costs = timeslice_max(costs)

    # Select technodata for the current year
    if "year" in technologies.dims and year is None:
        raise ValueError("Missing year argument")
    elif "year" in technologies.dims:
        techs = technologies.sel(year=year).drop_vars("year")
    else:
        techs = technologies

    # Run scipy optimization with highs solver
    adapter = ScipyAdapter.factory(
        techs, cast(np.ndarray, costs), *constraints, timeslice_level=timeslice_level
    )
    res = linprog(**adapter.kwargs, method="highs")

    # Backup: try with highs-ipm
    if not res.success and (res.status != 0):
        res = linprog(
            **adapter.kwargs,
            method="highs-ipm",
            options={
                "disp": True,
                "presolve": False,
                "dual_feasibility_tolerance": 1e-2,
                "primal_feasibility_tolerance": 1e-2,
                "ipm_optimality_tolerance": 1e-2,
            },
        )
        if not res.success:
            msg = (
                res.message
                + "\n"
                + f"Error in sector containing {technologies.technology.values}"
            )
            getLogger(__name__).critical(msg)
            raise GrowthOfCapacityTooConstrained

    # Convert results to a MUSE friendly format
    result = cast(Callable[[np.ndarray], xr.Dataset], adapter.to_muse)(res.x)
    return result


@register_investment(name=["cvxopt"])
def cvxopt_match_demand(
    costs: xr.DataArray,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    constraints: list[Constraint],
    year: Optional[int] = None,
    **options,
) -> xr.DataArray:
    from importlib import import_module
    from logging import getLogger

    from muse.constraints import ScipyAdapter

    if "year" in technologies.dims and year is None:
        raise ValueError("Missing year argument")
    elif "year" in technologies.dims:
        techs = technologies.interp(year=year).drop_vars("year")
    else:
        techs = technologies

    def default_to_scipy():
        return scipy_match_demand(costs, search_space, techs, constraints)

    try:
        cvxopt = import_module("cvxopt")
    except ModuleNotFoundError:
        msg = (
            "cvxopt is not installed\n"
            "It can be installed with `pip install cvxopt`\n"
            "Using the scipy linear solver instead."
        )
        getLogger(__name__).critical(msg)
        return default_to_scipy()

    if "timeslice" in costs.dims:
        costs = timeslice_max(costs)
    timeslice = next(cs.timeslice for cs in constraints if "timeslice" in cs.dims)
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
    res = cvxopt.solvers.lp(*map(cvxopt.matrix, args), **options)  # type: ignore
    if res["status"] != "optimal":
        getLogger(__name__).info(res["status"])
    if res["x"] is None:
        getLogger(__name__).critical("infeasible system")
        raise LinearProblemError("Infeasible system", res)

    solution = cast(Callable[[np.ndarray], xr.Dataset], adapter.to_muse)(list(res["x"]))
    return solution
