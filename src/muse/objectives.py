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
    A DataArray with at least two dimension corresponding to `replacement` and `asset`.
    A `timeslice` dimension may also be present.
"""

from __future__ import annotations

__all__ = [
    "capacity_to_service_demand",
    "capital_costs",
    "comfort",
    "efficiency",
    "emission_cost",
    "equivalent_annual_cost",
    "factory",
    "fixed_costs",
    "fuel_consumption_cost",
    "lifetime_levelized_cost_of_energy",
    "net_present_value",
    "register_objective",
]

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Callable, Union

import numpy as np
import xarray as xr
from mypy_extensions import KwArg

from muse.outputs.cache import cache_quantity
from muse.registration import registrator
from muse.timeslices import broadcast_timeslice, distribute_timeslice, drop_timeslice
from muse.utilities import check_dimensions

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
    settings: str | Mapping | Sequence[str | Mapping] = "LCOE",
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
        timeslice_level: str | None = None,
        *args,
        **kwargs,
    ) -> xr.Dataset:
        result = xr.Dataset()
        for name, objective in functions:
            obj = objective(
                technologies=technologies,
                demand=demand,
                prices=prices,
                timeslice_level=timeslice_level,
                *args,
                **kwargs,
            )
            if "timeslice" not in obj.dims:
                obj = broadcast_timeslice(obj, level=timeslice_level)
            if "timeslice" in result.dims:
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
    def decorated_objective(
        technologies: xr.Dataset, demand: xr.DataArray, *args, **kwargs
    ) -> xr.DataArray:
        from logging import getLogger

        # Check inputs
        check_dimensions(
            demand, ["asset", "timeslice", "commodity"], optional=["region"]
        )
        check_dimensions(
            technologies, ["replacement", "commodity"], optional=["timeslice", "asset"]
        )

        # Calculate objective
        result = function(technologies, demand, *args, **kwargs)
        result.name = function.__name__

        # Check result
        dtype = result.values.dtype
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            msg = f"dtype of objective {function.__name__} is not a number ({dtype})"
            getLogger(function.__module__).warning(msg)
        check_dimensions(result, ["replacement", "asset"], optional=["timeslice"])

        cache_quantity(**{result.name: result})
        return result

    return decorated_objective


@register_objective
def comfort(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Comfort value provided by technologies."""
    result = xr.broadcast(technologies.comfort, demand.asset)[0]
    return result


@register_objective
def efficiency(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Efficiency of the technologies."""
    result = xr.broadcast(technologies.efficiency, demand.asset)[0]
    return result


@register_objective(name="capacity")
def capacity_to_service_demand(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Minimum capacity required to fulfill the demand."""
    from muse.quantities import capacity_to_service_demand

    return capacity_to_service_demand(
        demand=demand, technologies=technologies, timeslice_level=timeslice_level
    )


@register_objective
def capacity_in_use(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    *args,
    **kwargs,
):
    from muse.commodities import is_enduse
    from muse.timeslices import TIMESLICE

    enduses = is_enduse(technologies.comm_usage.sel(commodity=demand.commodity))
    return (
        (demand.sel(commodity=enduses).sum("commodity") / TIMESLICE).sum("timeslice")
        * TIMESLICE.sum()
        / technologies.utilization_factor
    )


@register_objective
def consumption(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Commodity consumption when fulfilling the whole demand.

    Currently, the consumption is implemented for commodity_max == +infinity.
    """
    from muse.quantities import consumption
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )
    return consump.sum("commodity")


@register_objective
def fixed_costs(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    timeslice_level: str | None = None,
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
    from muse.costs import fixed_costs

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    result = fixed_costs(technologies, capacity)
    return result


@register_objective
def capital_costs(
    technologies: xr.Dataset,
    demand: xr.Dataset,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Capital costs for input technologies."""
    from muse.costs import capital_costs

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    result = capital_costs(technologies, capacity, method="lifetime")
    result = xr.broadcast(result, demand.asset)[0]
    return result


@register_objective(name="emissions")
def emission_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
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
    from muse.costs import environmental_costs

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    result = environmental_costs(technologies, prices, production)
    return result


@register_objective
def fuel_consumption_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Cost of fuels when fulfilling whole demand."""
    from muse.costs import fuel_costs
    from muse.quantities import consumption
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )
    result = fuel_costs(technologies, prices, consump)
    return result


@register_objective(name=["ALCOE"])
def annual_levelized_cost_of_energy(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Annual cost of energy (LCOE) of technologies - not dependent on production.

    It needs to be used for trade agents where the actual service is unknown. It follows
    the `simplified LCOE` given by NREL.

    See :py:func:`muse.costs.annual_levelized_cost_of_energy` for more details.

    """
    from muse.costs import levelized_cost_of_energy as LCOE
    from muse.quantities import consumption

    capacity = capacity_to_service_demand(
        technologies, demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )

    results = LCOE(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        consumption=consump,
        method="annual",
        aggregate_timeslices=True,
    )
    return results


@register_objective(name=["LCOE", "LLCOE"])
def lifetime_levelized_cost_of_energy(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    See :py:func:`muse.costs.lifetime_levelized_cost_of_energy` for more details.

    The LCOE is set to zero for those timeslices where the production is zero, normally
    due to a zero utilization factor.
    """
    from muse.costs import levelized_cost_of_energy as LCOE
    from muse.quantities import capacity_to_service_demand, consumption

    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )

    results = LCOE(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        consumption=consump,
        method="lifetime",
        aggregate_timeslices=True,
    )
    return results


@register_objective(name="NPV")
def net_present_value(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Net present value (NPV) of the relevant technologies.

    See :py:func:`muse.costs.net_present_value` for more details.
    """
    from muse.costs import net_present_value as NPV
    from muse.quantities import capacity_to_service_demand, consumption

    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )

    results = NPV(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        consumption=consump,
        aggregate_timeslices=True,
    )
    return results


@register_objective(name="NPC")
def net_present_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Net present cost (NPC) of the relevant technologies.

    See :py:func:`muse.costs.net_present_cost` for more details.
    """
    from muse.costs import net_present_cost as NPC
    from muse.quantities import capacity_to_service_demand, consumption

    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )

    results = NPC(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        consumption=consump,
        aggregate_timeslices=True,
    )
    return results


@register_objective(name="EAC")
def equivalent_annual_cost(
    technologies: xr.Dataset,
    demand: xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
    *args,
    **kwargs,
):
    """Equivalent annual costs (or annualized cost) of a technology.

    See :py:func:`muse.costs.equivalent_annual_cost` for more details.
    """
    from muse.costs import equivalent_annual_cost as EAC
    from muse.quantities import capacity_to_service_demand, consumption

    capacity = capacity_to_service_demand(
        technologies=technologies, demand=demand, timeslice_level=timeslice_level
    )
    production = (
        broadcast_timeslice(capacity, level=timeslice_level)
        * distribute_timeslice(technologies.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    )
    consump = consumption(
        technologies=technologies,
        prices=prices,
        production=production,
        timeslice_level=timeslice_level,
    )

    results = EAC(
        technologies=technologies,
        prices=prices,
        capacity=capacity,
        production=production,
        consumption=consump,
        aggregate_timeslices=True,
    )
    return results
