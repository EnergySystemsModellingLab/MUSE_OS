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
    def investment(agent: Agent, demand: DataArray,
                   ranking: DataArray, max_capacity: DataArray,
                   technologies: Dataset) -> DataArray:
        pass

Arguments:
    agent: the agent relevant to the investment procedure. The agent can be
        queried for parameters specific to the investment procedure.
    demand: specifies the demand that is expected to be fulfilled. It is an
        array with dimensions `asset` and `technology`.
    ranking: specifies for each `asset` which `technology` should be invested in
        preferentially (lower is more favorable). This should be an integer or
        floating point array with dimensions `asset` and `technology`.
    max_capacity: a limit on how much capacity each technology can be ramped up.
    technologies: a dataset containing all constant data characterizing the
        technologies.

Returns:
    A data array with dimensions `asset` and `technology` specifying the amount
    of newly invested capacity.
"""
__all__ = [
    "match_demand",
    "cliff_retirement_profile",
    "register_investment",
    "INVESTMENT_SIGNATURE",
]
from typing import Callable, Mapping, MutableMapping, Optional, Text, Union

from xarray import DataArray, Dataset

from muse.registration import registrator

INVESTMENT_SIGNATURE = Callable[
    [DataArray, DataArray, DataArray, DataArray, Dataset], DataArray
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
        demand: DataArray,
        ranking: DataArray,
        max_capacity: DataArray,
        search_space: DataArray,
        technologies: Dataset,
        *args,
        log_mismatch_params: float = 1e-3,
        **kwargs,
    ) -> DataArray:
        from logging import getLogger
        from muse.commodities import is_enduse

        result = function(  # type: ignore
            demand, ranking, max_capacity, search_space, technologies, *args, **kwargs
        )
        result = result.rename("investment")

        # log mismatch if requested
        if log_mismatch_params <= 0:
            return result

        mismatch = demand - (
            result
            * technologies.fixed_outputs.sel(
                commodity=is_enduse(technologies.comm_usage)
            )
        ).sum("replacement")
        mismatch = mismatch.rename("mismatch")

        logger = getLogger(function.__module__)
        if mismatch.max() < log_mismatch_params:
            m = "Minimized normalized capacity constraints successfully. "
            logger.info(m)
            m = "Investment matches demand up to {}".format(mismatch)
            logger.debug(m)
        else:
            m = (
                "Could not find investment to match demand, "
                "with maximum mismatch: {}".format(mismatch.max())
            )
            logger.error(m)
            m = "Total mismatch {}".format(mismatch)
            logger.debug(m)

        return result

    return decorated


def factory(settings: Union[Text, Mapping] = "match_demand") -> Callable:
    from typing import Dict

    if isinstance(settings, Text):
        name = settings
        params: Dict = {}
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

    if "log_mismatch_params" not in params:
        params["log_mismatch_params"] = 1e-3

    def compute_investment(
        demand: DataArray,
        search: Dataset,
        max_capacity: DataArray,
        technologies: Dataset,
        **kwargs,
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

        function = INVESTMENTS[name]
        return function(  # type: ignore
            demand,
            search.decision.rank("replacement").astype(int),
            max_capacity,
            search.space,
            technologies,
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


@register_investment(name="demand_matching")
def match_demand(
    demand: DataArray,
    ranking: DataArray,
    max_capacity: DataArray,
    search_space: DataArray,
    technologies: Dataset,
    year: int,
    timeslice_op: Optional[Callable[[DataArray], DataArray]] = None,
) -> DataArray:
    from muse.commodities import is_enduse
    from muse.demand_matching import demand_matching
    from muse.quantities import maximum_production, capacity_in_use
    from muse.timeslices import convert_timeslice, QuantityType

    demand = demand.sel(
        commodity=is_enduse(technologies.comm_usage.sel(commodity=demand.commodity))
    )
    technologies = technologies.sel(commodity=is_enduse(technologies.comm_usage))
    max_prod = maximum_production(
        technologies,
        max_capacity,
        year=year,
        technology=ranking.replacement,
        commodity=demand.commodity,
    ).drop_vars("technology")
    if "timeslice" in demand.dims and "timeslice" not in max_prod.dims:
        max_prod = convert_timeslice(max_prod, demand, QuantityType.EXTENSIVE)

    # Push disabled techs to last rank.
    # Any production assigned to them by the demand-matching algorithm will be removed.
    minobj = ranking.min()
    maxobj = ranking.where(search_space, minobj).max("replacement") + 1
    decision = ranking.where(search_space, maxobj)

    production = demand_matching(
        demand.sel(asset=demand.asset.isin(search_space.asset)), decision, max_prod
    ).where(search_space, 0)

    capacity = capacity_in_use(
        production, technologies, year=year, technology=production.replacement
    ).drop_vars("technology")
    if "timeslice" in capacity.dims and timeslice_op is not None:
        capacity = timeslice_op(capacity)
    return capacity.rename("capacity addition")
