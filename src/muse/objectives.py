"""Valuation functions for replacement technologies.

.. currentmodule:: muse.objectives

Objectives are used to compare replacement technologies. They should correspond to
a single well defined economic concept. Multiple objectives can later be combined
via decision functions.

Objectives should be registered via the
:py:func:`@register_objective<register_objective>` decorator. This makes it possible to
refer to them by name in agent input files, and nominally to set extra input parameters.

The :py:func:`factory` function creates a function that calls all objectives defined in
its input argument and returns a dataset with each objective as a separate data array.

Objectives are not expected to modify their arguments. Furthermore they should
conform the following signatures:

.. code-block:: Python

    @register_objective
    def comfort(
        technologies: xr.Dataset,
        demand: xr.DataArray,
        prices: xr.DataArray,
        **kwargs
    ) -> xr.DataArray:
        pass

Arguments:
    technologies: A data set characterising the technologies from which the
        agent can draw assets. This has been pre-filtered according to the agent's
        search space.
    demand: Demand to fulfill.
    prices: Commodity prices.
    kwargs: Extra input parameters. These parameters are expected to be set from the
        input file.

        .. warning::

            The standard :ref:`agent csv file<inputs-agents>` does not allow to set
            these parameters.

Returns:
    A DataArray with at least one dimension corresponding to ``replacement``.
    Other dimensions can be present, as long as the subsequent decision function knows
    how to reduce them.
"""

__all__ = [
    "register_objective",
    "comfort",
    "efficiency",
    "fixed_costs",
    "capital_costs",
    "emission_cost",
    "fuel_consumption_cost",
    "lifetime_levelized_cost_of_energy",
    "net_present_value",
    "equivalent_annual_cost",
    "capacity_to_service_demand",
    "factory",
]

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Callable, Union

import numpy as np
import xarray as xr
from mypy_extensions import KwArg

from muse.outputs.cache import cache_quantity
from muse.registration import registrator
from muse.timeslices import drop_timeslice
from muse.utilities import filter_input

OBJECTIVE_SIGNATURE = Callable[
    [xr.Dataset, xr.DataArray, xr.DataArray, KwArg(Any)], xr.DataArray
]
"""Objectives signature."""

OBJECTIVES: MutableMapping[str, OBJECTIVE_SIGNATURE] = {}
"""Dictionary of objectives when selecting replacement technology."""


def objective_factory(settings=Union[str, Mapping]):
    from functools import partial

    if isinstance(settings, str):
        params = dict(name=settings)
    else:
        params = dict(**settings)
    name = params.pop("name")
    function = OBJECTIVES[name]
    return partial(function, **params)


def factory(
    settings: Union[str, Mapping, Sequence[Union[str, Mapping]]] = "LCOE",
) -> Callable:
    """Creates a function computing multiple objectives.

    The input can be a single objective defined by its name alone. Or it can be a single
    objective defined by a dictionary which must include at least a "name" item, as well
    as any extra parameters to pass to the objective. Or it can be a sequence of
    objectives defined by name or by dictionary.
    """
    from logging import getLogger

    if isinstance(settings, str):
        params: list[dict] = [{"name": settings}]
    elif isinstance(settings, Mapping):
        params = [dict(**settings)]
    else:
        params = [
            {"name": param} if isinstance(param, str) else dict(**param)
            for param in settings
        ]

    if len(set(param["name"] for param in params)) != len(params):
        msg = (
            "The same objective is named twice."
            " The result may be undefined if parameters differ."
        )
        getLogger(__name__).critical(msg)

    functions = [(param["name"], objective_factory(param)) for param in params]

    def objectives(
        technologies: xr.Dataset,
        demand: xr.DataArray,
        prices: xr.DataArray,
        *args,
        **kwargs,
    ) -> xr.Dataset:
        result = xr.Dataset()
        for name, objective in functions:
            obj = objective(
                technologies=technologies, demand=demand, prices=prices, *args, **kwargs
            )
            if "timeslice" in obj.dims and "timeslice" in result.dims:
                obj = drop_timeslice(obj)
            result[name] = obj
        return result

    return objectives


@registrator(registry=OBJECTIVES, loglevel="info")
def register_objective(function: OBJECTIVE_SIGNATURE):
    """Decorator to register a function as a objective.

    Registers a function as a objective so that it can be applied easily
    when sorting technologies one against the other.

    The input name is expected to be in lower_snake_case, since it ought to be a
    python function. CamelCase, lowerCamelCase, and kebab-case names are also
    registered.
    """
    from functools import wraps

    @wraps(function)
    def decorated_objective(technologies: xr.Dataset, *args, **kwargs) -> xr.DataArray:
        from logging import getLogger

        result = function(technologies, *args, **kwargs)

        dtype = result.values.dtype
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            msg = f"dtype of objective {function.__name__} is not a number ({dtype})"
            getLogger(function.__module__).warning(msg)

        if "replacement" not in result.dims:
            raise RuntimeError("Objective should return a dimension 'replacement'")
        if "technology" in result.dims:
            raise RuntimeError("Objective should not return a dimension 'technology'")
        if "technology" in result.coords:
            raise RuntimeError("Objective should not return a coordinate 'technology'")
        if "year" in result.dims:
            raise RuntimeError("Objective should not return a dimension 'year'")
        result.name = function.__name__
        cache_quantity(**{result.name: result})
        return result

    return decorated_objective


@register_objective
def comfort(
    technologies: xr.Dataset,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Comfort value provided by technologies."""
    return technologies.comfort


@register_objective
def efficiency(
    technologies: xr.Dataset,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Efficiency of the technologies."""
    return technologies.efficiency


@register_objective(name="capacity")
def capacity_to_service_demand(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Minimum capacity required to fulfill the demand."""
    from muse.quantities import capacity_to_service_demand
    from muse.timeslices import represent_hours

    hours = represent_hours(demand.timeslice)
    return capacity_to_service_demand(
        demand=demand, technologies=technologies, hours=hours
    )


@register_objective
def capacity_in_use(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
):
    from muse.commodities import is_enduse
    from muse.timeslices import represent_hours

    hours = represent_hours(demand.timeslice)
    enduses = is_enduse(technologies.comm_usage.sel(commodity=demand.commodity))
    return (
        (demand.sel(commodity=enduses).sum("commodity") / hours).sum("timeslice")
        * hours.sum()
        / technologies.utilization_factor
    )


@register_objective
def consumption(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Commodity consumption when fulfilling the whole demand.

    Currently, the consumption is implemented for commodity_max == +infinity.
    """
    from muse.quantities import consumption

    result = consumption(technologies=technologies, prices=prices, production=demand)
    return result.sum("commodity")


@register_objective
def fixed_costs(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    r"""Fixed costs associated with a technology.

    Given a factor :math:`\alpha` and an  exponent :math:`\beta`, the fixed costs
    :math:`F` are computed from the :py:func:`capacity fulfilling the current demand
    <capacity_to_service_demand>` :math:`C` as:

    .. math::

        F = \alpha * C^\beta

    :math:`\alpha` and :math:`\beta` are "fix_par" and "fix_exp" in
    :ref:`inputs-technodata`, respectively.
    """
    capacity = capacity_to_service_demand(technologies, demand)
    result = technologies.fix_par * (capacity**technologies.fix_exp)
    return result


@register_objective
def capital_costs(
    technologies: xr.Dataset,
    *args,
    **kwargs,
) -> xr.DataArray:
    r"""Capital costs for input technologies.

    The capital costs are computed as :math:`a * b^\alpha`, where :math:`a` is
    "cap_par" from the :ref:`inputs-technodata`, :math:`b` is the "scaling_size", and
    :math:`\alpha` is "cap_exp". In other words, capital costs are constant across the
    simulation for each technology.
    """
    result = technologies.cap_par * (technologies.scaling_size**technologies.cap_exp)
    return result


@register_objective(name="emissions")
def emission_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    r"""Emission cost for each technology when fulfilling whole demand.

    Given the demand share :math:`D`, the emissions per amount produced :math:`E`, and
    the prices per emittant :math:`P`, then emissions costs :math:`C` are computed
    as:

    .. math::

        C = \sum_s \left(\sum_cD\right)\left(\sum_cEP\right),

    with :math:`s` the timeslices and :math:`c` the commodity.
    """
    from muse.commodities import is_enduse, is_pollutant

    enduses = is_enduse(technologies.comm_usage.sel(commodity=demand.commodity))
    total = demand.sel(commodity=enduses).sum("commodity")
    envs = is_pollutant(technologies.comm_usage)
    prices = filter_input(prices, year=demand.year.item(), commodity=envs)
    return total * (technologies.fixed_outputs * prices).sum("commodity")


@register_objective
def fuel_consumption_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Cost of fuels when fulfilling whole demand."""
    from muse.commodities import is_fuel
    from muse.quantities import consumption

    commodity = is_fuel(technologies.comm_usage.sel(commodity=demand.commodity))
    fcons = consumption(technologies=technologies, prices=prices, production=demand)
    prices = filter_input(prices, year=demand.year.item(), commodity=commodity)
    return (fcons * prices).sum("commodity")


@register_objective(name=["ALCOE"])
def annual_levelized_cost_of_energy(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Annual cost of energy (LCOE) of technologies - not dependent on production.

    It needs to be used for trade agents where the actual service is unknown. It follows
    the `simplified LCOE` given by NREL.

    See :py:func:`muse.costs.annual_levelized_cost_of_energy` for more details.

    """
    from muse.costs import annual_levelized_cost_of_energy as aLCOE

    return filter_input(
        aLCOE(technologies=technologies, prices=prices).max("timeslice"),
        year=demand.year.item(),
    )


@register_objective(name=["LCOE", "LLCOE"])
def lifetime_levelized_cost_of_energy(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    See :py:func:`muse.costs.lifetime_levelized_cost_of_energy` for more details.

    The LCOE is set to zero for those timeslices where the production is zero, normally
    due to a zero utilisation factor.
    """
    from muse.costs import lifetime_levelized_cost_of_energy as LCOE
    from muse.timeslices import QuantityType, convert_timeslice

    capacity = capacity_to_service_demand(technologies, demand)
    production = capacity * technologies.fixed_outputs * technologies.utilization_factor
    production = convert_timeslice(production, demand.timeslice, QuantityType.EXTENSIVE)

    results = LCOE(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        year=demand.year.item(),
    )

    return results.where(np.isfinite(results)).fillna(0.0)


@register_objective(name="NPV")
def net_present_value(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Net present value (NPV) of the relevant technologies.

    See :py:func:`muse.costs.net_present_value` for more details.
    """
    from muse.costs import net_present_value as NPV
    from muse.timeslices import QuantityType, convert_timeslice

    capacity = capacity_to_service_demand(technologies, demand)
    production = capacity * technologies.fixed_outputs * technologies.utilization_factor
    production = convert_timeslice(production, demand.timeslice, QuantityType.EXTENSIVE)

    results = NPV(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        year=demand.year.item(),
    )
    return results


@register_objective(name="NPC")
def net_present_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Net present cost (NPC) of the relevant technologies.

    See :py:func:`muse.costs.net_present_cost` for more details.
    """
    from muse.costs import net_present_cost as NPC
    from muse.timeslices import QuantityType, convert_timeslice

    capacity = capacity_to_service_demand(technologies, demand)
    production = capacity * technologies.fixed_outputs * technologies.utilization_factor
    production = convert_timeslice(production, demand.timeslice, QuantityType.EXTENSIVE)

    results = NPC(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        year=demand.year.item(),
    )
    return results


@register_objective(name="EAC")
def equivalent_annual_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    *args,
    **kwargs,
):
    """Equivalent annual costs (or annualized cost) of a technology.

    See :py:func:`muse.costs.equivalent_annual_cost` for more details.
    """
    from muse.costs import equivalent_annual_cost as EAC
    from muse.timeslices import QuantityType, convert_timeslice

    capacity = capacity_to_service_demand(technologies, demand)
    production = capacity * technologies.fixed_outputs * technologies.utilization_factor
    production = convert_timeslice(production, demand.timeslice, QuantityType.EXTENSIVE)

    results = EAC(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        year=demand.year.item(),
    )
    return results
