"""Demand share computations.

The demand share splits a demand amongst agents. It is used within a sector to assign
part of the input MCA demand to each agent.

Demand shares functions should be registered via the decorator `register_demand_share`.

Demand share functions are not expected to modify any of their arguments. They
should all have the following signature:

.. code-block:: Python

    @register_demand_share
    def demand_share(
        agents: Sequence[AbstractAgent],
        demand: xr.Dataarray,
        technologies: xr.Dataset,
        **kwargs
    ) -> xr.DataArray:
        pass

Arguments:
    agents: a sequence of  agent relevant to the demand share procedure. The agent can
        be queried for parameters specific to the demand share procedure. For instance,
        :py:func`new_and_retro` will query the agents for the assets they own, the
        region they are contained with, their category (new or retrofit), etc...
    demand: DataArray of commodity demands for the current year and investment year.
    technologies: a dataset containing all constant data characterizing the
        technologies.
    kwargs: Any number of keyword arguments that can parametrize how the demand is
        shared. These keyword arguments can be modified from the TOML file.

Returns:
    The unmet consumption. Unless indicated, all agents will compete for a the full
    demand. However, if there exists a coordinate "agent" of dimension "asset" giving
    the :py:attr:`~muse.agents.agent.AbstractAgent.uuid` of the agent, then agents will
    only service that part of the demand.
"""

from __future__ import annotations

__all__ = [
    "DEMAND_SHARE_SIGNATURE",
    "factory",
    "new_and_retro",
    "register_demand_share",
    "unmet_demand",
    "unmet_forecasted_demand",
]

from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    cast,
)

import xarray as xr
from mypy_extensions import KwArg

from muse.agents import AbstractAgent
from muse.errors import (
    AgentWithNoAssetsInDemandShare,
    RetrofitAgentInStandardDemandShare,
)
from muse.registration import registrator
from muse.utilities import check_dimensions

DEMAND_SHARE_SIGNATURE = Callable[
    [Sequence[AbstractAgent], xr.DataArray, xr.Dataset, KwArg(Any)], xr.DataArray
]
"""Demand share signature."""

DEMAND_SHARE: MutableMapping[str, DEMAND_SHARE_SIGNATURE] = {}
"""Dictionary of demand share functions."""


@registrator(registry=DEMAND_SHARE, loglevel="info")
def register_demand_share(function: DEMAND_SHARE_SIGNATURE):
    """Decorator to register a function as a demand share calculation."""
    return function


def factory(
    settings: str | Mapping[str, Any] | None = None,
) -> DEMAND_SHARE_SIGNATURE:
    if settings is None or isinstance(settings, str):
        name = settings or "standard_demand"
        params: Mapping[str, Any] = {}
    else:
        name = settings.get("name", "standard_demand")
        params = {k: v for k, v in settings.items() if k != "name"}

    function = DEMAND_SHARE[name]
    keywords = dict(**params)

    def demand_share(
        agents: Sequence[AbstractAgent],
        demand: xr.DataArray,
        technologies: xr.Dataset,
        **kwargs,
    ) -> xr.DataArray:
        from copy import copy

        keyword_args = copy(keywords)
        keyword_args.update(**kwargs)

        # Check inputs
        check_dimensions(
            demand,
            ["commodity", "year", "timeslice", "region"],
            optional=["dst_region"],
        )
        assert len(demand.year) == 2
        check_dimensions(
            technologies,
            ["technology", "region"],
            optional=["timeslice", "commodity", "dst_region"],
        )

        # Calculate demand share
        result = function(agents, demand, technologies, **keyword_args)

        # Check result
        check_dimensions(
            result, ["timeslice", "commodity"], optional=["asset", "region"]
        )  # TODO: asset should be required, but trade model is failing
        return result

    return cast(DEMAND_SHARE_SIGNATURE, demand_share)


@register_demand_share(name="new_and_retro")
def new_and_retro(
    agents: Sequence[AbstractAgent],
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    r"""Splits demand across new and retro agents.

    The input demand is split amongst both *new* and *retrofit* agents. *New* agents get
    a share of the increase in demand for the investment year, whereas *retrofit* agents
    are assigned a share of the demand that occurs from decommissioned assets.

    Args:
        agents: a list of all agents. This list should mainly be used to determine the
            type of an agent and the assets it owns. The agents will not be modified in
            any way.
        demand: commodity demands for the current year and investment year.
        technologies: quantities describing the technologies.
        timeslice_level: the timeslice level of the sector (e.g. "hour", "day")

    Pseudo-code:

    #. the capacity is reduced over agents and  expanded over timeslices (extensive
       quantity) and aggregated over agents. Generally:

       .. math::

           A_{a, s}^r = w_s\sum_i A_a^{r, i}

       with :math:`w_s` a weight associated with each timeslice and determined via
       :py:func:`muse.timeslices.distribute_timeslice`.

    #. An intermediate quantity, the :py:func:`unmet demand
       <muse.demand_share.unmet_demand>` :math:`U` is defined from
       :math:`P[\mathcal{M}, \mathcal{A}]`, a function giving the production for a given
       market :math:`\mathcal{M}`, the associated consumption :math:`\mathcal{C}`, and
       aggregate assets :math:`\mathcal{A}`:

       .. math::
           U[\mathcal{M}, \mathcal{A}] =
             \max(\mathcal{C} - P[\mathcal{M}, \mathcal{A}], 0)

       where :math:`\max` operates element-wise, and indices have been dropped for
       simplicity. The resulting expression has the same indices as the consumption
       :math:`\mathcal{C}_{c, s}^r`.

       :math:`P` is the maximum production, given by
       <muse.quantities.maximum_production>`.


    #. the *new* demand :math:`N` is defined as:

        .. math::

            N = \min\left(
                \mathcal{C}_{c, s}^r(y + \Delta y) - \mathcal{C}_{c, s}^r(y),
                U[\mathcal{M}^r(y + \Delta y), \mathcal{A}_{a, s}^r(y)]
            \right)

    #. the *retrofit* demand :math:`R` is defined from the identity

       .. math::

           C_{c, s}^r(y + \Delta y) =
            P[\mathcal{M}^r(y+\Delta y), \mathcal{A}_{a, s}^r(y + \Delta y)]
            + N_{c, s}^r
            + R_{c, s}^r

       In other words, it is the share of the forecasted consumption that is serviced
       neither by the current assets still present in the investment year, nor by the
       *new* agent.

    #. then each *new* agent gets a share of :math:`N` proportional to it's
        share of the :py:func:`production <muse.quantities.maximum_production>`,
        :math:`P[\mathcal{A}_{a, s}^{r, i}(y)]`.  Then the share of the demand for new
        agent :math:`i` is:

        .. math::

            N_{c, s, t}^{i, r}(y) = N_{c, s}^r
                \frac{\sum_\iota P[\mathcal{A}_{s, t, \iota}^{r, i}(y)]}
                     {\sum_{i, t, \iota}P[\mathcal{A}_{s, t, \iota}^{r, i}(y)]}


    #. similarly, each *retrofit* agent gets a share of :math:`N` proportional to it's
        share of the :py:func:`decommissioning demand
        <muse.quantities.decommissioning_demand>`, :math:`D^{r, i}_{t, c}`.
        Then the share of the demand for retrofit agent :math:`i` is:

        .. math::

            R_{c, s, t}^{i, r}(y) = R_{c, s}^r
                \frac{\sum_\iota\mathcal{D}_{t, c, \iota}^{i, r}(y)}
                    {\sum_{i, t, \iota}\mathcal{D}_{t, c, \iota}^{i, r}(y)}


    Note that in the last two steps, the assets owned by the agent are aggregated over
    the installation year. The effect is that the demand serviced by agents is
    disaggregated over each technology, rather than not over each *model* of each
    technology (asset).

    .. SeeAlso::

        :ref:`indices`, :ref:`quantities`,
        :ref:`Agent investments<model, agent investment>`,
        :py:func:`~muse.quantities.decommissioning_demand`,
        :py:func:`~muse.quantities.maximum_production`
    """
    from functools import partial

    from muse.commodities import is_enduse
    from muse.quantities import maximum_production
    from muse.utilities import (
        agent_concatenation,
        broadcast_over_assets,
        interpolate_capacity,
        reduce_assets,
    )

    current_year, investment_year = map(int, demand.year.values)

    def decommissioning(capacity, technologies):
        return decommissioning_demand(
            technologies=technologies,
            capacity=interpolate_capacity(
                capacity, year=[current_year, investment_year]
            ),
            timeslice_level=timeslice_level,
        )

    capacity = interpolate_capacity(
        reduce_assets([u.assets.capacity for u in agents]),
        year=[current_year, investment_year],
    )

    # Select technodata for assets
    technodata = broadcast_over_assets(technologies, capacity, installed_as_year=True)

    demands = new_and_retro_demands(
        capacity,
        demand,
        technodata,
        timeslice_level=timeslice_level,
    )

    demands = demands.where(
        is_enduse(technologies.comm_usage.sel(commodity=demands.commodity)), 0
    )

    quantity = {
        agent.name: agent.quantity for agent in agents if agent.category != "retrofit"
    }

    for agent in agents:
        if agent.category == "retrofit":
            setattr(agent, "quantity", quantity[agent.name])

    id_to_share: MutableMapping[Hashable, xr.DataArray] = {}
    for region in demands.region.values:
        retro_capacity: MutableMapping[Hashable, xr.DataArray] = {
            agent.uuid: interpolate_capacity(
                agent.assets.capacity, year=[current_year, investment_year]
            )
            for agent in agents
            if agent.category == "retrofit" and agent.region == region
        }
        retro_technodata: MutableMapping[Hashable, xr.Dataset] = {
            agent_uuid: technodata.sel(asset=retro_capacity[agent_uuid].asset)
            for agent_uuid in retro_capacity.keys()
        }
        name_to_id = {
            (agent.name, agent.region): agent.uuid
            for agent in agents
            if agent.category == "retrofit" and agent.region == region
        }
        id_to_rquantity = {
            agent.uuid: (agent.name, agent.region, agent.quantity)
            for agent in agents
            if agent.category == "retrofit" and agent.region == region
        }

        retro_demands: MutableMapping[Hashable, xr.DataArray] = _inner_split(
            retro_capacity,
            retro_technodata,
            demands.retrofit.sel(region=region),
            decommissioning,
            id_to_rquantity,
        )
        assert len(name_to_id) == len(retro_capacity)

        id_to_share.update(retro_demands)

        new_capacity: Mapping[Hashable, xr.DataArray] = {
            agent.uuid: retro_capacity[name_to_id[(agent.name, agent.region)]]
            #            * agent.quantity
            for agent in agents
            if agent.category != "retrofit" and agent.region == region
        }
        new_technodata: MutableMapping[Hashable, xr.Dataset] = {
            agent_uuid: technodata.sel(asset=new_capacity[agent_uuid].asset)
            for agent_uuid in new_capacity.keys()
        }
        id_to_nquantity = {
            agent.uuid: (agent.name, agent.region, agent.quantity)
            for agent in agents
            if agent.category != "retrofit" and agent.region == region
        }
        new_demands = _inner_split(
            new_capacity,
            new_technodata,
            demands.new.sel(region=region),
            partial(
                maximum_production,
                year=current_year,
                timeslice_level=timeslice_level,
            ),
            id_to_nquantity,
        )

        id_to_share.update(new_demands)

    result = cast(xr.DataArray, agent_concatenation(id_to_share))
    return result


@register_demand_share(name="default")
def standard_demand(
    agents: Sequence[AbstractAgent],
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    r"""Splits demand across new agents.

    The input demand is split amongst *new* agents. *New* agents get a
    share of the increase in demand for the investment year, as well as the demand that
    occurs from decommissioned assets.

    Args:
        agents: a list of all agents. This list should mainly be used to determine the
            type of an agent and the assets it owns. The agents will not be modified in
            any way.
        demand: commodity demands for the current year and investment year.
        technologies: quantities describing the technologies.
        timeslice_level: the timeslice level of the sector (e.g. "hour", "day")

    """
    from functools import partial

    from muse.commodities import is_enduse
    from muse.quantities import maximum_production
    from muse.utilities import (
        agent_concatenation,
        broadcast_over_assets,
        interpolate_capacity,
        reduce_assets,
    )

    current_year, investment_year = map(int, demand.year.values)

    def decommissioning(capacity, technologies):
        return decommissioning_demand(
            technologies=technologies,
            capacity=interpolate_capacity(
                capacity, year=[current_year, investment_year]
            ),
            timeslice_level=timeslice_level,
        )

    # Make sure there are no retrofit agents
    for agent in agents:
        if agent.category == "retrofit":
            raise RetrofitAgentInStandardDemandShare()

    # Calculate existing capacity
    capacity = interpolate_capacity(
        reduce_assets([agent.assets.capacity for agent in agents]),
        year=[current_year, investment_year],
    )

    # Select technodata for assets
    technodata = broadcast_over_assets(technologies, capacity, installed_as_year=True)

    # Calculate new and retrofit demands
    demands = new_and_retro_demands(
        capacity=capacity,
        demand=demand,
        technologies=technodata,
        timeslice_level=timeslice_level,
    )

    # Only consider end-use commodities
    demands = demands.where(
        is_enduse(technologies.comm_usage.sel(commodity=demands.commodity)), 0
    )

    id_to_share: MutableMapping[Hashable, xr.DataArray] = {}
    for region in demands.region.values:
        # Calculate current capacity
        current_capacity: MutableMapping[Hashable, xr.DataArray] = {
            agent.uuid: interpolate_capacity(
                agent.assets.capacity, year=[current_year, investment_year]
            )
            for agent in agents
            if agent.region == region
        }
        current_technodata: MutableMapping[Hashable, xr.Dataset] = {
            agent_uuid: technodata.sel(asset=current_capacity[agent_uuid].asset)
            for agent_uuid in current_capacity.keys()
        }

        # Split demands between agents
        id_to_quantity = {
            agent.uuid: (agent.name, agent.region, agent.quantity)
            for agent in agents
            if agent.region == region
        }
        retro_demands: MutableMapping[Hashable, xr.DataArray] = _inner_split(
            current_capacity,
            current_technodata,
            demands.retrofit.sel(region=region),
            decommissioning,
            id_to_quantity,
        )
        new_demands = _inner_split(
            current_capacity,
            current_technodata,
            demands.new.sel(region=region),
            partial(
                maximum_production,
                year=current_year,
                timeslice_level=timeslice_level,
            ),
            id_to_quantity,
        )

        # Sum new and retrofit demands
        total_demands = {
            k: new_demands[k] + retro_demands[k] for k in new_demands.keys()
        }
        id_to_share.update(total_demands)

    result = cast(xr.DataArray, agent_concatenation(id_to_share))
    assert "year" not in result.dims
    return result


@register_demand_share(name="unmet_demand")
def unmet_forecasted_demand(
    agents: Sequence[AbstractAgent],
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Forecast demand that cannot be serviced by non-decommissioned current assets."""
    from muse.commodities import is_enduse
    from muse.utilities import (
        broadcast_over_assets,
        interpolate_capacity,
        reduce_assets,
    )

    current_year, investment_year = map(int, demand.year.values)

    demand = demand.where(
        is_enduse(technologies.comm_usage.sel(commodity=demand.commodity)), 0
    )

    # Calculate existing capacity
    capacity = interpolate_capacity(
        reduce_assets([agent.assets.capacity for agent in agents]),
        year=[current_year, investment_year],
    )

    # Select data for future years
    future_demand = demand.sel(year=investment_year, drop=True)
    future_capacity = capacity.sel(year=investment_year)

    # Select technology data for assets
    techs = broadcast_over_assets(technologies, capacity, installed_as_year=True)

    # Calculate unmet demand
    result = unmet_demand(
        demand=future_demand,
        capacity=future_capacity,
        technologies=techs,
        timeslice_level=timeslice_level,
    )
    assert "year" not in result.dims
    return result


def _inner_split(
    assets: Mapping[Hashable, xr.DataArray],
    technologies: Mapping[Hashable, xr.DataSet],
    demand: xr.DataArray,
    method: Callable,
    quantity: Mapping,
) -> MutableMapping[Hashable, xr.DataArray]:
    r"""Compute share of the demand for a set of agents.

    The input ``demand`` is split between agents according to their share of the
    demand computed by ``method``.
    """
    from numpy import logical_and

    # Find decrease in capacity production by each asset over time
    shares: Mapping[Hashable, xr.DataArray] = {
        key: method(capacity=capacity, technologies=technologies[key])
        .groupby("technology")
        .sum("asset")
        .rename(technology="asset")
        for key, capacity in assets.items()
    }

    # Total decrease in production across assets
    try:
        summed_shares: xr.DataArray = xr.concat(shares.values(), dim="concat_dim").sum(
            "concat_dim"
        )
        total: xr.DataArray = summed_shares.sum("asset")
    except AttributeError:
        raise AgentWithNoAssetsInDemandShare()

    # Calculates the demand divided by the number of assets times the number of agents
    # if the demand is bigger than zero and the total demand assigned with the "method"
    # function is zero.
    n_agents = len(quantity)
    n_assets = summed_shares.sizes["asset"]
    unassigned = (demand / (n_agents * n_assets)).where(
        logical_and(demand > 1e-12, total <= 1e-12), 0
    )

    totals = {
        key: (share / share.sum("asset")).fillna(0) for key, share in shares.items()
    }

    newshares = {
        key: (total * quantity[key][2] * demand).fillna(0)
        + unassigned * quantity[key][2]
        for key, total in totals.items()
    }
    return newshares


def unmet_demand(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
):
    r"""Share of the demand that cannot be serviced by the existing assets.

    .. math::
        U[\mathcal{M}, \mathcal{A}] =
          \max(\mathcal{C} - P[\mathcal{M}, \mathcal{A}], 0)

    :math:`\max` operates element-wise, and indices have been dropped for simplicity.
    The resulting expression has the same indices as the consumption
    :math:`\mathcal{C}_{c, s}^r`.

    :math:`P` is the maximum production, given by <muse.quantities.maximum_production>.
    """
    from muse.quantities import maximum_production

    # Check inputs
    assert "year" not in technologies.dims
    assert "year" not in capacity.dims
    assert "year" not in demand.dims

    # Calculate maximum production by existing assets
    produced = maximum_production(
        capacity=capacity,
        technologies=technologies,
        timeslice_level=timeslice_level,
    )

    # Total commodity production by summing over assets
    if "dst_region" in produced.dims:
        produced = produced.sum("asset").rename(dst_region="region")
    elif "region" in produced.coords and produced.region.dims:
        produced = produced.groupby("region").sum("asset")
    else:
        produced = produced.sum("asset")

    # Unmet demand is the difference between the consumption and the production
    _unmet_demand = (demand - produced).clip(min=0)
    assert "year" not in _unmet_demand.dims
    return _unmet_demand


def new_consumption(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    r"""Computes share of the demand attributed to new agents.

    The new agents service the demand that can be attributed specifically to growth and
    that cannot be serviced by existing assets. In other words:

    .. math::
        N_{c, s}^r = \min\left(
            C_{c, s}^(y + \Delta y) - C_{c, s}^(y),
            C_{c, s}^(y + \Delta y)
                - P[\mathcal{M}(y + \Delta y), \mathcal{A}_{a, s}^r(y)]
        \right)

    Where :math:`P` the maximum production by existing assets, given by
    <muse.quantities.maximum_production>.
    """
    from numpy import minimum

    # Check inputs
    assert len(demand.year) == 2
    assert len(capacity.year) == 2
    assert (demand.year.values == capacity.year.values).all()
    assert "year" not in technologies.dims
    current_year, investment_year = map(int, demand.year.values)

    # Select data for current/future years
    current_demand = demand.sel(year=current_year, drop=True)
    future_demand = demand.sel(year=investment_year, drop=True)
    future_capacity = capacity.sel(year=investment_year)

    # Calculate the increase in consumption over the investment period
    delta = (future_demand - current_demand).clip(min=0)
    missing = unmet_demand(
        demand=future_demand,
        capacity=future_capacity,
        technologies=technologies,
        timeslice_level=timeslice_level,
    )
    consumption = minimum(delta, missing)
    assert "year" not in consumption.dims
    return consumption


def new_and_retro_demands(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.Dataset:
    """Splits demand into *new* and *retrofit* demand.

    The demand in the investment year is split three ways:

    #. the demand that can be serviced by the assets that will still be operational that
        year.
    #. the *new* demand is defined as the growth in consumption that cannot be serviced
        by existing assets in the current year, as computed in :py:func:`new_demand`.
    #. the retrofit demand is everything else.
    """
    from numpy import minimum

    from muse.quantities import maximum_production

    # Check inputs
    assert len(demand.year) == 2
    assert len(capacity.year) == 2
    assert (demand.year.values == capacity.year.values).all()
    assert "year" not in technologies.dims
    investment_year = int(demand.year[1])

    if hasattr(capacity, "region") and capacity.region.dims == ():
        capacity["region"] = (
            "asset",
            [str(capacity.region.values)] * len(capacity.asset),
        )

    # Calculate demand to allocate to "new" agents
    new_demand = new_consumption(
        capacity=capacity,
        demand=demand,
        technologies=technologies,
        timeslice_level=timeslice_level,
    )

    # Maximum production in the investment year by existing assets
    service = (
        maximum_production(
            technologies=technologies,
            capacity=capacity.sel(year=investment_year),
            timeslice_level=timeslice_level,
        )
        .groupby("region")
        .sum("asset")
    )

    # Existing asset should not execute beyond demand
    service = minimum(service, demand.sel(year=investment_year, drop=True))

    # Leftover demand that cannot be serviced by existing assets or "new" agents
    retro_demand = (
        demand.sel(year=investment_year, drop=True) - new_demand - service
    ).clip(min=0)
    assert "year" not in retro_demand.dims

    return xr.Dataset({"new": new_demand, "retrofit": retro_demand})


def decommissioning_demand(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    r"""Computes demand from process decommissioning.

    Let :math:`M_t^r(y)` be the retrofit demand, :math:`^{(s)}\mathcal{D}_t^r(y)` be the
    decommissioning demand at the level of the sector, and :math:`A^r_{t, \iota}(y)` be
    the assets owned by the agent. Then, the decommissioning demand for agent :math:`i`
    is :

    .. math::

        \mathcal{D}^{r, i}_{t, c}(y) =
            \sum_\iota \alpha_{t, \iota}^r \beta_{t, \iota, c}^r
                \left(A^{i, r}_{t, \iota}(y) - A^{i, r}_{t, \iota, c}(y + 1) \right)

    given the utilization factor :math:`\alpha_{t, \iota}` and the fixed output factor
    :math:`\beta_{t, \iota, c}`.

    Furthermore, decommissioning demand is non-zero only for end-use commodities.

    ncsearch-nohlsearch).. SeeAlso:
        :ref:`indices`, :ref:`quantities`,
        :py:func:`~muse.quantities.maximum_production`
        :py:func:`~muse.commodities.is_enduse`
    """
    from muse.quantities import maximum_production

    assert len(capacity.year) == 2
    assert "year" not in technologies.dims
    current_year, investment_year = map(int, capacity.year.values)

    # Calculate the decrease in capacity from the current year to future years
    capacity_decrease = capacity.sel(year=current_year) - capacity.sel(
        year=investment_year
    )

    # Calculate production associated with this capacity
    result = maximum_production(
        technologies,
        capacity_decrease,
        timeslice_level=timeslice_level,
    ).clip(min=0)
    assert "year" not in result.dims
    return result
