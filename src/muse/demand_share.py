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
    kwargs: Additional keyword arguments.

Returns:
    A DataArray of demand shares.
"""

from __future__ import annotations

from collections.abc import Hashable, MutableMapping, Sequence
from functools import wraps
from logging import getLogger
from typing import Any, Callable, cast

import numpy as np
import xarray as xr
from mypy_extensions import KwArg

from muse.agents import AbstractAgent
from muse.commodities import is_enduse
from muse.errors import RetrofitAgentInStandardDemandShare
from muse.quantities import maximum_production
from muse.registration import registrator
from muse.utilities import (
    agent_concatenation,
    broadcast_over_assets,
    check_dimensions,
    interpolate_capacity,
    reduce_assets,
)

__all__ = [
    "DEMAND_SHARE_SIGNATURE",
    "factory",
    "new_and_retro",
    "register_demand_share",
    "unmet_demand",
    "unmet_forecasted_demand",
]

DEMAND_SHARE_SIGNATURE = Callable[
    [Sequence[AbstractAgent], xr.DataArray, xr.Dataset, KwArg(Any)], xr.DataArray
]
"""Demand share signature."""

DEMAND_SHARE: MutableMapping[str, DEMAND_SHARE_SIGNATURE] = {}
"""Dictionary of demand share functions."""


@registrator(registry=DEMAND_SHARE, loglevel="info")
def register_demand_share(function: DEMAND_SHARE_SIGNATURE) -> DEMAND_SHARE_SIGNATURE:
    """Registers a demand share function with MUSE."""

    @wraps(function)
    def decorated(
        agents: Sequence[AbstractAgent],
        demand: xr.DataArray,
        technologies: xr.Dataset,
        **kwargs,
    ) -> xr.DataArray:
        """Computes and validates a demand share."""
        # Validate inputs
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

        # We can only share demand for enduse commodities
        # So we need to check that demand is zero for all non-enduse commodities
        enduse_names = technologies.commodity[is_enduse(technologies.comm_usage)]
        non_enduse = demand.commodity[~demand.commodity.isin(enduse_names)]
        assert (demand.sel(commodity=non_enduse) == 0).all()

        # Calculate demand share
        result = function(agents, demand, technologies, **kwargs)

        # Validate output
        check_dimensions(
            result, ["timeslice", "commodity"], optional=["asset", "region"]
        )  # TODO: asset should be required, but trade model is failing
        return result

    return cast(DEMAND_SHARE_SIGNATURE, decorated)


def factory(name: str) -> DEMAND_SHARE_SIGNATURE:
    """Get a demand share function by name."""
    return DEMAND_SHARE[name]


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
    current_year, investment_year = map(int, demand.year.values)

    # Calculate existing capacity and broadcast technologies
    capacity = interpolate_capacity(
        reduce_assets([agent.assets.capacity for agent in agents]),
        year=[current_year, investment_year],
    )
    technodata = broadcast_over_assets(technologies, capacity, installed_as_year=True)

    # Calculate new and retrofit demands
    demands = new_and_retro_demands(
        capacity=capacity,
        demand=demand,
        technologies=technodata,
        timeslice_level=timeslice_level,
    )

    # Split demand between agents
    agent_demands: MutableMapping[Hashable, xr.DataArray] = {}
    for region in demands.region.values:
        total_retro_quantity = 0
        total_new_quantity = 0
        for agent in agents:
            if agent.region != region:
                continue

            # Select data for the agent
            current_capacity = interpolate_capacity(
                agent.assets.capacity, year=[current_year, investment_year]
            )
            current_technodata = technodata.sel(asset=agent.assets.asset)

            # Calculate the agent's share of the retrofit and new demands
            agent_retrofit_demand = demands.retrofit.sel(region=region) * agent.quantity
            agent_new_demand = demands.new.sel(region=region) * agent.quantity

            if agent.category == "retrofit":
                total_retro_quantity += agent.quantity
                # Split retrofit demand over assets based on decommissioning demand
                retro_shares = decommissioning_demand(
                    technologies=current_technodata,
                    capacity=interpolate_capacity(
                        current_capacity, year=[current_year, investment_year]
                    ),
                    timeslice_level=timeslice_level,
                )
                agent_demands[agent.uuid] = _inner_split(
                    agent_retrofit_demand,
                    retro_shares,
                )
            elif agent.category == "newcapa":
                total_new_quantity += agent.quantity
                # Split new demand over assets based on maximum production
                new_shares = maximum_production(
                    capacity=current_capacity,
                    technologies=current_technodata,
                    year=current_year,
                    timeslice_level=timeslice_level,
                )
                agent_demands[agent.uuid] = _inner_split(
                    agent_new_demand,
                    new_shares,
                )
            else:
                raise ValueError(f"Unknown agent category: {agent.category}")

        # Make sure the total new/retro agent quantity = 1
        # TODO: ideally we should check this in the input layer
        if abs(total_retro_quantity - 1) > 1e-2:
            msg = (
                f"Total retrofit agent quantity in region {region} "
                f"is not 1: {total_retro_quantity}"
            )
            getLogger(__name__).critical(msg)
        if abs(total_new_quantity - 1) > 1e-2:
            msg = (
                f"Total new agent quantity in region {region} "
                f"is not 1: {total_new_quantity}"
            )
            getLogger(__name__).critical(msg)

    result = agent_concatenation(agent_demands)
    assert "year" not in result.dims
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
    # Validate no retrofit agents
    if any(agent.category == "retrofit" for agent in agents):
        raise RetrofitAgentInStandardDemandShare(
            "Standard demand share cannot be used with retrofit agents"
        )

    current_year, investment_year = map(int, demand.year.values)

    # Calculate existing capacity and broadcast technologies
    capacity = interpolate_capacity(
        reduce_assets([agent.assets.capacity for agent in agents]),
        year=[current_year, investment_year],
    )
    technodata = broadcast_over_assets(technologies, capacity, installed_as_year=True)

    # Calculate new and retrofit demands
    demands = new_and_retro_demands(
        capacity=capacity,
        demand=demand,
        technologies=technodata,
        timeslice_level=timeslice_level,
    )

    # Split demand between agents
    agent_demands: MutableMapping[Hashable, xr.DataArray] = {}
    for region in demands.region.values:
        total_quantity = 0
        for agent in agents:
            if agent.region != region:
                continue

            # Select data for the agent
            agent_capacity = interpolate_capacity(
                agent.assets.capacity, year=[current_year, investment_year]
            )
            agent_technodata = technodata.sel(asset=agent.assets.asset)

            # Calculate the agent's share of the retrofit and new demands
            agent_retrofit_demand = demands.retrofit.sel(region=region) * agent.quantity
            agent_new_demand = demands.new.sel(region=region) * agent.quantity
            total_quantity += agent.quantity

            # Split retrofit demand over assets based on decommissioning demand
            retro_shares = decommissioning_demand(
                technologies=agent_technodata,
                capacity=agent_capacity,
                timeslice_level=timeslice_level,
            )
            retro_demands = _inner_split(
                agent_retrofit_demand,
                retro_shares,
            )

            # Split new demand over assets based on maximum production
            new_shares = maximum_production(
                capacity=agent_capacity,
                technologies=agent_technodata,
                year=current_year,
                timeslice_level=timeslice_level,
            )
            new_demands = _inner_split(
                agent_new_demand,
                new_shares,
            )

            # Sum new and retrofit demands for the agent
            agent_demands[agent.uuid] = retro_demands + new_demands

        # Make sure the total agent quantity = 1
        # TODO: ideally we should check this in the input layer
        if abs(total_quantity - 1) > 1e-2:
            msg = f"Total agent quantity in region {region} is not 1: {total_quantity}"
            getLogger(__name__).critical(msg)

    result = agent_concatenation(agent_demands)
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
    current_year, investment_year = map(int, demand.year.values)

    # Calculate existing capacity
    capacity = interpolate_capacity(
        reduce_assets([agent.assets.capacity for agent in agents]),
        year=[current_year, investment_year],
    )

    # Select data for future years
    future_demand = demand.sel(year=investment_year, drop=True)
    future_capacity = capacity.sel(year=investment_year)

    # Calculate unmet demand
    result = unmet_demand(
        demand=future_demand,
        capacity=future_capacity,
        technologies=broadcast_over_assets(
            technologies, capacity, installed_as_year=True
        ),
        timeslice_level=timeslice_level,
    )
    assert "year" not in result.dims
    return result


def _inner_split(
    demand: xr.DataArray,
    shares: xr.DataArray,
) -> xr.DataArray:
    """Split demand over assets based on shares.

    Args:
        demand: the demand to split
        shares: the shares to apply to each asset
    """
    # Assets of the same technology type are grouped together
    grouped_shares: xr.DataArray = (
        shares.groupby("technology").sum("asset").rename(technology="asset")
    )

    # Split demand over assets according to shares
    split_demand = demand * (grouped_shares / grouped_shares.sum("asset")).fillna(0)

    # Split unassigned demand equally over assets
    unassigned = (demand - split_demand.sum("asset")).clip(min=0)
    unassigned_per_asset = unassigned / grouped_shares.sizes["asset"]
    split_demand += unassigned_per_asset
    return split_demand


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

    # Return unmet demand
    return (demand - produced).clip(min=0)


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
    # Validate inputs have matching years
    if not (
        len(demand.year) == len(capacity.year) == 2
        and (demand.year.values == capacity.year.values).all()
    ):
        raise ValueError("Capacity and demand must have matching years")

    current_year, investment_year = map(int, demand.year.values)

    # Calculate demand growth
    current_demand = demand.sel(year=current_year, drop=True)
    future_demand = demand.sel(year=investment_year, drop=True)
    new_demand = (future_demand - current_demand).clip(min=0)

    # If future capacity is higher than existing capacity, it's possible that
    # this might already be able to make up some of the increase in demand
    missing = unmet_demand(
        demand=future_demand,
        capacity=capacity.sel(year=investment_year),
        technologies=technologies,
        timeslice_level=timeslice_level,
    )

    # Return minimum of growth and unmet demand
    return np.minimum(new_demand, missing)


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
    # Validate inputs have matching years
    if not (
        len(demand.year) == len(capacity.year) == 2
        and (demand.year.values == capacity.year.values).all()
    ):
        raise ValueError("Capacity and demand must have matching years")

    investment_year = int(demand.year[1])

    if hasattr(capacity, "region") and capacity.region.dims == ():
        capacity["region"] = (
            "asset",
            [str(capacity.region.values)] * len(capacity.asset),
        )

    # Calculate new demand from growth
    new_demand = new_consumption(
        capacity=capacity,
        demand=demand,
        technologies=technologies,
        timeslice_level=timeslice_level,
    )

    # Calculate retrofit demand as remaining unmet demand
    retrofit_demand = (
        unmet_demand(
            demand=demand.sel(year=investment_year, drop=True),
            capacity=capacity.sel(year=investment_year),
            technologies=technologies,
            timeslice_level=timeslice_level,
        )
        - new_demand
    ).clip(0)

    return xr.Dataset({"new": new_demand, "retrofit": retrofit_demand})


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
