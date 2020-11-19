"""Demand share computations.

The demand share splits a demand amongst agents. It is used within a sector to assign
part of the input MCA demand to each agent.

Demand shares functions should be registered via the decorator `register_demand_share`.

Demand share functions are not expected to modify any of their arguments. They
should all have the following signature:

.. code-block:: Python

    @register_demand_share
    def demand_share(
<<<<<<< HEAD
        agents: Sequence[AbstractAgent],
        market: xr.Dataset,
        technologies: xr.Dataset,
        **kwargs
    ) -> xr.DataArray:
        pass

Arguments:
    agents: a sequence of  agent relevant to the demand share procedure. The agent can
        be queried for parameters specific to the demand share procedure. For instance,
=======
        agents: List[AgentBase],
        market: Dataset,
        technologies: Dataset,
        **kwargs
    ) -> MutableMapping[UUID, DataArray]:
        pass

Arguments:
    agents: a list of  agent relevant to the demand share procedure. The agent can be
        queried for parameters specific to the demand share procedure. For instance,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        :py:func`new_and_retro` will query the agents for the assets they own, the
        region they are contained with, their category (new or retrofit), etc...
    market: Market variables, including prices, consumption and supply.
    technologies: a dataset containing all constant data characterizing the
        technologies.
    kwargs: Any number of keyword arguments that can parametrize how the demand is
        shared. These keyword arguments can be modified from the TOML file.

Returns:
<<<<<<< HEAD
    The unmet consumption. Unless indicated, all agents will compete for a the full
    demand. However, if there exists a coordinate "agent" of dimension "asset" giving
    the :py:attr:`~muse.agents.agent.AbstractAgent.uuid` of the agent, then agents will
    only service that par of the demand.
=======
    A dictionnary mapping the :py:attr:`muse.agent.AgentBase.uuid` (`Universally
    Unique Identifier`__) of the agent to the demand share. In most cases, it is
    expected summing over shares should return the *unmet* consumption for the forecast
    year.

__ :: https://en.wikipedia.org/wiki/Universally_unique_identifier
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
"""
__all__ = [
    "new_and_retro",
    "factory",
    "register_demand_share",
    "unmet_demand",
<<<<<<< HEAD
    "unmet_forecasted_demand",
    "DEMAND_SHARE_SIGNATURE",
]
from typing import (
    Any,
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Text,
    Union,
    cast,
)

import xarray as xr
from mypy_extensions import KwArg

from muse.agents import AbstractAgent
from muse.registration import registrator

DEMAND_SHARE_SIGNATURE = Callable[
    [Sequence[AbstractAgent], xr.Dataset, xr.Dataset, KwArg(Any)], xr.DataArray
=======
    "DEMAND_SHARE_SIGNATURE",
]
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Union
from uuid import UUID

from xarray import DataArray, Dataset

from muse.agent import AgentBase
from muse.registration import registrator

DEMAND_SHARE_SIGNATURE = Callable[
    [List[AgentBase], Dataset, Dataset], Mapping[UUID, DataArray]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
]
"""Demand share signature."""

DEMAND_SHARE: MutableMapping[Text, DEMAND_SHARE_SIGNATURE] = {}
"""Dictionary of demand share functions."""


@registrator(registry=DEMAND_SHARE, loglevel="info")
def register_demand_share(function: DEMAND_SHARE_SIGNATURE):
    """Decorator to register a function as a demand share calculation."""
    return function


def factory(
    settings: Optional[Union[Text, Mapping[Text, Any]]] = None
) -> DEMAND_SHARE_SIGNATURE:
    if settings is None or isinstance(settings, Text):
        name = settings or "new_and_retro"
        params: Mapping[Text, Any] = {}
    else:
        name = settings.get("name", "new_and_retro")
        params = {k: v for k, v in settings.items() if k != "name"}

<<<<<<< HEAD
    function = DEMAND_SHARE[name]
    keywords = dict(**params)

    def demand_share(
        agents: Sequence[AbstractAgent],
        market: xr.Dataset,
        technologies: xr.Dataset,
        **kwargs
    ) -> xr.DataArray:
        keywords.update(**kwargs)
        return function(agents, market, technologies, **keywords)

    return cast(DEMAND_SHARE_SIGNATURE, demand_share)
=======
    def demand_share(
        agents: List[AgentBase], market: Dataset, technologies: Dataset, **kwargs
    ) -> Mapping[UUID, DataArray]:
        function = DEMAND_SHARE[name]
        keywords = dict(**params)
        keywords.update(**kwargs)
        return function(agents, market, technologies, **keywords)  # type: ignore

    return demand_share
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


@register_demand_share(name="default")
def new_and_retro(
<<<<<<< HEAD
    agents: Sequence[AbstractAgent],
    market: xr.Dataset,
    technologies: xr.Dataset,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> xr.DataArray:
=======
    agents: List[AgentBase],
    market: Dataset,
    technologies: Dataset,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> MutableMapping[UUID, DataArray]:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    r"""Splits demand across new and retro agents.

    The input demand is split amongst both *new* and *retro* agents. *New* agents get a
    share of the increase in demand for the forecast year, whereas *retrofi* agents are
    assigned a share of the demand that occurs from decommissioned assets.

    Args:
        agents: a list of all agents. This list should mainly be used to determine the
            type of an agent and the assets it owns. The agents will not be modified in
            any way.
        market: the market for which to satisfy the demand. It should contain at-least
            ``consumption`` and ``supply``. It may contain ``prices`` if that is of use
            to the production method. The ``consumption`` reflects the demand for the
            commodities produced by the current sector.
        technologies: quantities describing the technologies.

    Pseudo-code:

    #. the capacity is reduced over agents and  expanded over timeslices (extensive
       quantity) and aggregated over agents. Generally:

       .. math::

           A_{a, s}^r = w_s\sum_i A_a^{r, i}

       with :math:`w_s` a weight associated with each timeslice and determined via
       :py:func:`muse.timeslices.convert_timeslice`.

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

       :math:`P` is any function registered with
       :py:func:`@register_production<muse.production.register_production>`.


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
       neither by the current assets still present in the forecast year, nor by the
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


    Note that tin the last two steps, the assets owned by the agent are aggregated over
    the installation year. The effect is that the demand serviced by agents is
    disaggregated over each technology, rather than not over each *model* of each
    technology.

    .. SeeAlso::

        :ref:`indices`, :ref:`quantities`,
        :ref:`Agent investments<model, agent investment>`,
        :py:func:`~muse.quantities.decommissioning_demand`,
        :py:func:`~muse.quantities.maximum_production`
    """
    from functools import partial
    from muse.commodities import is_enduse
<<<<<<< HEAD
    from muse.utilities import reduce_assets, agent_concatenation
=======
    from muse.utilities import reduce_assets
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from muse.quantities import maximum_production

    if current_year is None:
        current_year = market.year.min()

    capacity = reduce_assets([u.assets.capacity for u in agents])

    demands = new_and_retro_demands(
        capacity,
        market,
        technologies,
        production=production,
        current_year=current_year,
        forecast=forecast,
    )
    demands = demands.where(
        is_enduse(technologies.comm_usage.sel(commodity=demands.commodity)), 0
    )

    def decommissioning(capacity):
        from muse.quantities import decommissioning_demand

        return decommissioning_demand(
            technologies, capacity, year=[current_year, current_year + forecast]
        ).squeeze("year")

<<<<<<< HEAD
    id_to_share: MutableMapping[Hashable, xr.DataArray] = {}
    for region in demands.region.values:
        regional_techs = technologies.sel(region=region)
        retro_capacity: MutableMapping[Hashable, xr.DataArray] = {
=======
    results: MutableMapping[UUID, DataArray] = {}
    for region in demands.region.values:
        regional_techs = technologies.sel(region=region)
        retro_capacity = {
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            agent.uuid: agent.assets.capacity
            for agent in agents
            if agent.category == "retrofit" and agent.region == region
        }
<<<<<<< HEAD
        retro_demands: MutableMapping[Hashable, xr.DataArray] = _inner_split(
            retro_capacity, demands.retrofit.sel(region=region), decommissioning
        )
        id_to_share.update(retro_demands)

        name_to_id = {
=======
        retro_demands = _inner_split(
            retro_capacity, demands.retrofit.sel(region=region), decommissioning
        )
        results.update(retro_demands)

        name_to_uuid = {
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            (agent.name, agent.region): agent.uuid
            for agent in agents
            if agent.category == "retrofit" and agent.region == region
        }
<<<<<<< HEAD
        assert len(name_to_id) == len(retro_capacity)
        new_capacity: Mapping[Hashable, xr.DataArray] = {
            agent.uuid: retro_capacity[name_to_id[(agent.name, agent.region)]]
=======
        assert len(name_to_uuid) == len(retro_capacity)
        new_capacity = {
            agent.uuid: retro_capacity[name_to_uuid[(agent.name, agent.region)]]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            * getattr(agent, "quantity", 0.3)
            for agent in agents
            if agent.category != "retrofit" and agent.region == region
        }
        new_demands = _inner_split(
            new_capacity,
            demands.new.sel(region=region),
            partial(maximum_production, technologies=regional_techs, year=current_year),
        )
<<<<<<< HEAD
        id_to_share.update(new_demands)
    result = cast(xr.DataArray, agent_concatenation(id_to_share))
    return result


@register_demand_share(name="unmet_demand")
def unmet_forecasted_demand(
    agents: Sequence[AbstractAgent],
    market: xr.Dataset,
    technologies: xr.Dataset,
    current_year: Optional[int] = None,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    forecast: int = 5,
) -> xr.DataArray:
    """Forecast demand that cannot be serviced by non-decommissioned current assets."""
    from muse.commodities import is_enduse
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.utilities import reduce_assets

    if current_year is None:
        current_year = market.year.min()

    year = current_year + forecast
    comm_usage = technologies.comm_usage.sel(commodity=market.commodity)
    smarket: xr.Dataset = market.where(is_enduse(comm_usage), 0).interp(year=year)
    capacity = reduce_assets([u.assets.capacity.interp(year=year) for u in agents])
    ts_capacity = cast(
        xr.DataArray,
        convert_timeslice(capacity, market.timeslice, QuantityType.EXTENSIVE),
    )

    result = unmet_demand(smarket, ts_capacity, technologies, production)
    if "year" in result.dims:
        result = result.squeeze("year")
    return result


def _inner_split(
    assets: Mapping[Hashable, xr.DataArray], demand: xr.DataArray, method: Callable
) -> MutableMapping[Hashable, xr.DataArray]:
=======
        results.update(new_demands)
    return results


def _inner_split(
    assets: Mapping[UUID, DataArray], demand: DataArray, method: Callable, **filters
) -> Mapping[UUID, DataArray]:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    r"""compute share of the demand for a set of agents.

    The input ``demand`` is split between agents according to their share of the
    demand computed by ``method``.
    """
    from numpy import logical_and

    shares = {
<<<<<<< HEAD
        key: method(capacity=capacity)
        .groupby("technology")
        .sum("asset")
        .rename(technology="asset")
        for key, capacity in assets.items()
    }
    total = sum(shares.values()).sum("asset")  # type: ignore
    unassigned = (
        demand / (len(shares) * len(cast(xr.DataArray, sum(shares.values())).asset))
    ).where(logical_and(demand > 1e-12, total <= 1e-12), 0)
    return {
        key: ((share / total).fillna(0) * demand).fillna(0) + unassigned
        for key, share in shares.items()
=======
        uuid: method(capacity=capacity)
        .groupby("technology")
        .sum("asset")
        .rename(technology="asset")
        for uuid, capacity in assets.items()
    }
    total = sum(shares.values()).sum("asset")  # type: ignore
    unassigned = (demand / (len(shares) * len(sum(shares.values()).asset))).where(
        logical_and(demand > 1e-12, total <= 1e-12), 0
    )
    return {
        uuid: ((share / total).fillna(0) * demand).fillna(0) + unassigned
        for uuid, share in shares.items()
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    }


def unmet_demand(
<<<<<<< HEAD
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
=======
    market: Dataset,
    capacity: DataArray,
    technologies: Dataset,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    production: Union[Text, Mapping, Callable] = "maximum_production",
):
    r"""Share of the demand that cannot be serviced by the existing assets.

    .. math::
        U[\mathcal{M}, \mathcal{A}] =
          \max(\mathcal{C} - P[\mathcal{M}, \mathcal{A}], 0)

    :math:`\max` operates element-wise, and indices have been dropped for simplicity.
    The resulting expression has the same indices as the consumption
    :math:`\mathcal{C}_{c, s}^r`.

    :math:`P` is any function registered with
    :py:func:`@register_production<muse.production.register_production>`.
    """
    from muse.production import factory as prod_factory

    prod_method = production if callable(production) else prod_factory(production)
    assert callable(prod_method)

<<<<<<< HEAD
    production = prod_method(
        market=market, capacity=capacity, technologies=technologies
    )
    if "region" in production.coords and production.region.dims:
        production = production.groupby("region").sum("asset")
    else:
        production = production.sum("asset")
    return (market.consumption - production).clip(min=0)


def new_consumption(
    capacity: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> xr.DataArray:
=======
    prod = (
        prod_method(market=market, capacity=capacity, technologies=technologies)
        .groupby("region")
        .sum("asset")
    )
    return (market.consumption - prod).clip(min=0)


def new_consumption(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> DataArray:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    r""" Computes share of the demand attributed to new agents.

    The new agents service the demand that can be attributed specificaly to growth and
    that cannot be serviced by existing assets. In other words:

    .. math::
        N_{c, s}^r = \min\left(
            C_{c, s}^(y + \Delta y) - C_{c, s}^(y),
            C_{c, s}^(y + \Delta y)
                - P[\mathcal{M}(y + \Delta y), \mathcal{A}_{a, s}^r(y)]
        \right)

    Where :math:`P` is a production function taking the market and assets as arguments.
    """
    from muse.timeslices import convert_timeslice, QuantityType

    if current_year is None:
        current_year = market.year.min()

    ts_capa = convert_timeslice(
        capacity.interp(year=current_year), market.timeslice, QuantityType.EXTENSIVE
    )
<<<<<<< HEAD
    assert isinstance(ts_capa, xr.DataArray)
=======
    assert isinstance(ts_capa, DataArray)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    market = market.interp(year=[current_year, current_year + forecast])
    current = market.sel(year=current_year, drop=True)
    forecasted = market.sel(year=current_year + forecast, drop=True)

    delta = (forecasted.consumption - current.consumption).clip(min=0)
    missing = unmet_demand(forecasted, ts_capa, technologies)

    return delta.where(delta < missing, missing)


def new_and_retro_demands(
<<<<<<< HEAD
    capacity: xr.DataArray,
    market: xr.Dataset,
    technologies: xr.Dataset,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> xr.Dataset:
=======
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    production: Union[Text, Mapping, Callable] = "maximum_production",
    current_year: Optional[int] = None,
    forecast: int = 5,
) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    """Splits demand into *new* and *retrofit* demand.

    The demand (.i.e. `market.consumption`) in the forecast year is split three ways:

    #. the demand that can be serviced the assets that will still be operational that
        year.
    #. the *new* demand is defined as the growth in consumption that cannot be serviced
        by existing assets in the current year, as computed in :py:func:`new_demand`.
    #. the retrofit demand is everything else.
    """
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.production import factory as prod_factory

    production_method = production if callable(production) else prod_factory(production)
    assert callable(production_method)
    if current_year is None:
        current_year = market.year.min()

<<<<<<< HEAD
    smarket: xr.Dataset = market.interp(year=[current_year, current_year + forecast])
=======
    smarket: Dataset = market.interp(year=[current_year, current_year + forecast])
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    ts_capa = convert_timeslice(
        capacity.interp(year=[current_year, current_year + forecast]),
        market.timeslice,
        QuantityType.EXTENSIVE,
    )
<<<<<<< HEAD
    assert isinstance(ts_capa, xr.DataArray)
=======
    assert isinstance(ts_capa, DataArray)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    if hasattr(ts_capa, "region") and ts_capa.region.dims == ():
        ts_capa["region"] = "asset", [str(ts_capa.region.values)] * len(ts_capa.asset)

    new_demand = new_consumption(
<<<<<<< HEAD
        ts_capa, smarket, technologies, current_year=current_year, forecast=forecast
=======
        ts_capa,
        smarket,
        technologies,
        current_year=current_year,
        forecast=forecast,
        production=production_method,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    )
    if "year" in new_demand.dims:
        new_demand = new_demand.squeeze("year")

    service = (
        production_method(
            smarket.sel(year=current_year + forecast),
            ts_capa.sel(year=current_year + forecast),
            technologies,
        )
        .groupby("region")
        .sum("asset")
    )
    retro_demand = (
        smarket.consumption.sel(year=current_year + forecast, drop=True)
        - new_demand
        - service
    ).clip(min=0)
    if "year" in retro_demand.dims:
        retro_demand = retro_demand.squeeze("year")

<<<<<<< HEAD
    return xr.Dataset({"new": new_demand, "retrofit": retro_demand})
=======
    return Dataset({"new": new_demand, "retrofit": retro_demand})
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
