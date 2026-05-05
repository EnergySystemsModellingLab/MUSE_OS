"""Various ways and means to compute production.

Production is the amount of commodities produced by an asset. However, depending on the
context, it could be computed several ways. For  instance, it can be obtained straight
from the capacity of the asset. Or it can be obtained by matching for the same
commodities with a set of assets.

Production methods can be registered via the :py:func:`@register_production
<register_production>` production decorator.  Registering a function makes the function
accessible from MUSE's input file. Production methods are not expected to modify their
arguments. Furthermore they should conform the
following signatures:

.. code-block:: python

    @register_production
    def production(
        market: xr.Dataset, capacity: xr.DataArray, technologies: xr.Dataset, **kwargs
    ) -> xr.DataArray:
        pass


Arguments:
    market: Market, including demand and prices.
    capacity: The capacity of each asset within a market.
    technologies: A dataset characterising the technologies of the same assets.
    **kwargs: Any number of keyword arguments

Returns:
    A `xr.DataArray` with the amount produced for each good from each asset.
"""

from __future__ import annotations

__all__ = [
    "PRODUCTION_SIGNATURE",
    "factory",
    "maximum_production",
    "register_production",
    "share_based_supply",
]

from collections.abc import MutableMapping
from typing import Callable

import numpy as np
import xarray as xr

from muse.registration import registrator

"""Production signature."""
PRODUCTION_SIGNATURE = Callable[
    [xr.Dataset, xr.DataArray, xr.Dataset, str], xr.DataArray
]

"""Dictionary of production methods. """
PRODUCTION_METHODS: MutableMapping[str, PRODUCTION_SIGNATURE] = {}


@registrator(registry=PRODUCTION_METHODS, loglevel="info")
def register_production(function: PRODUCTION_SIGNATURE):
    """Decorator to register a function as a production method.

    .. seealso::

        :py:mod:`muse.production`
    """
    return function


def factory(name) -> PRODUCTION_SIGNATURE:
    from muse.dispatch import PRODUCTION_METHODS

    return PRODUCTION_METHODS[name]


@register_production(name=("max", "maximum"))
def maximum_production(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilization factor. For more details, see
    :py:func:`muse.quantities.maximum_production`.
    """
    from muse.quantities import maximum_production

    return maximum_production(technologies, capacity, timeslice_level)


@register_production(name=("share", "shares"))
def share_based_supply(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Service current demand equally from all assets.

    "Equally" means that equivalent technologies are used to the same percentage of
    their respective capacity.
    """
    return _share_based_supply_internal(
        capacity, market.consumption, technologies, timeslice_level=timeslice_level
    )


@register_production(name=("merit", "merit_order"))
def merit_order_supply(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Service current demand from the cheapest assets."""
    return _merit_order_supply_internal(
        capacity,
        market.consumption,
        technologies,
        prices=market.prices,
        timeslice_level=timeslice_level,
    )


def _share_based_supply_internal(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: xr.Dataset | xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Production and emission for a given capacity servicing a given demand.

    This method distributes the demand across assets in proportion to their maximum
    production. For example, if asset A can produce 10 units at full capacity and asset
    B can produce 20 units, then A will service 1/3 of the total demand and B will
    service 2/3of the total demand. If demand is lower than total maximum production,
    then the supply from each asset is reduced proportionally. If demand exceeds
    capacity, production is capped at each asset's maximum.

    This function is most appropriate for sectors where demand is inherently
    distributed across assets and cannot be shifted between them (e.g.
    residential heating systems, where each household meets its own demand).

    Arguments:
        capacity: number/quantity of assets that can service the demand
        demand: amount of each end-use required. The supply of each process will not
            exceed its share of the demand.
        technologies: factors bindings the capacity of an asset with its production of
            commodities and environmental pollutants.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        A data array where the commodity dimension only contains actual outputs (i.e. no
        input commodities).
    """
    from muse.commodities import CommodityUsage, check_usage, is_pollutant
    from muse.quantities import emission, maximum_production, minimum_production
    from muse.utilities import broadcast_over_assets

    assert "asset" not in demand.dims
    assert "asset" in capacity.dims

    # Maximum and minimum production for each asset
    maxprod = maximum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )
    minprod = minimum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )

    # Split commodity-level demands over assets in proportion to maxprod
    if len(set(maxprod.region.values.flatten())) == 1:
        # Single region models
        if "region" in demand.dims:
            demand = demand.sel(region=maxprod.region)
        share_by_asset = maxprod / maxprod.sum("asset")
        demand_by_asset = (demand * share_by_asset).fillna(0)

    elif "dst_region" in maxprod.dims:
        # Trade models
        demand = demand.rename(region="dst_region")
        total_maxprod_by_dst_region = maxprod.groupby("dst_region").sum(dim="asset")
        share_by_asset = maxprod / total_maxprod_by_dst_region
        demand_by_asset = (demand * share_by_asset).fillna(0)
        # TODO: Haven't verified that this is correct - to examine in the future

    else:
        # Multi-region models
        demand = broadcast_over_assets(demand, maxprod, installed_as_year=False)
        total_maxprod_by_region = maxprod.groupby("region").sum(dim="asset")
        share_by_asset = maxprod / broadcast_over_assets(
            total_maxprod_by_region, maxprod, installed_as_year=False
        )
        demand_by_asset = (demand * share_by_asset).fillna(0)

    # Supply is equal to demand, bounded between minprod and maxprod
    assert "asset" in demand_by_asset.dims
    result = np.minimum(demand_by_asset, maxprod)
    result = np.maximum(result, minprod)

    # Add production of environmental pollutants
    env = is_pollutant(technologies.comm_usage)
    result[{"commodity": env}] = emission(
        result, technologies, timeslice_level=timeslice_level
    ).transpose(*result.dims)
    result[
        {"commodity": ~check_usage(technologies.comm_usage, CommodityUsage.PRODUCT)}
    ] = 0

    return result


def _merit_order_supply_internal(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: xr.Dataset | xr.DataArray,
    prices: xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    from muse.commodities import CommodityUsage, check_usage, is_pollutant
    from muse.costs import levelized_cost_of_energy
    from muse.quantities import (
        consumption,
        emission,
        maximum_production,
        minimum_production,
    )

    assert "asset" not in demand.dims
    assert "asset" in capacity.dims

    # Maximum and minimum production for each asset
    maxprod = maximum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )
    minprod = minimum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )

    # Consumption of each asset assuming full dispatch, for calculating costs later on.
    maxcons = consumption(
        technologies, maxprod, prices=prices, timeslice_level=timeslice_level
    )

    # Normalise region dimension
    if len(set(maxprod.region.values.flatten())) == 1:
        if "region" in demand.dims:
            demand = demand.sel(region=maxprod.region)
            prices = prices.sel(region=maxprod.region)
    else:
        raise ValueError("merit_order_supply not yet supported in multi-region models")

    # Set capital costs to zero so they're not included in the cost-minimisation
    technologies = technologies.assign(cap_par=xr.zeros_like(technologies.cap_par))

    # All assets operate at at least their minimum production
    result = minprod.copy()

    for y in maxprod.year.values:
        # Calculate timeslice-level costs for each asset in this year assuming full
        # dispatch. We use LCOE excluding capital costs.
        technology_costs = levelized_cost_of_energy(
            technologies,
            prices.sel(year=y),
            capacity.sel(year=y),
            production=maxprod.sel(year=y),
            consumption=maxcons.sel(year=y),
        )

        # Meet supply in each timeslice by utilising the cheapest assets first, up to
        # their maximum production
        for ts in maxprod.timeslice.values:
            # Remaining demand after minimum production (i.e. what we need to meet with
            # the available headroom above minimum)
            remaining = demand.sel(year=y, timeslice=ts) - minprod.sel(
                year=y, timeslice=ts
            ).sum("asset")

            # Rank assets by cost (cheapest first)
            costs_ts = technology_costs.sel(timeslice=ts)
            order = costs_ts.sortby(costs_ts).asset

            # Available extra production above minimum, sorted by cost
            available = (maxprod - minprod).sel(year=y, timeslice=ts).sel(asset=order)

            # cumsum_before[i] = total available production of all assets cheaper than
            # i
            cumsum_before = available.cumsum("asset") - available

            # (remaining - cumsum_before)[i] = demand still unmet after cheaper assets
            # are fully utilised. Clipping to [0, available[i]] ensures we never
            # produce negative amounts or exceed this asset's headroom above minprod.
            addition = (remaining - cumsum_before).clip(min=0, max=available)
            result.loc[dict(year=y, timeslice=ts)] += addition.reindex(
                asset=result.asset.values
            )

    # Add production of environmental pollutants
    env = is_pollutant(technologies.comm_usage)
    result[{"commodity": env}] = emission(
        result, technologies, timeslice_level=timeslice_level
    ).transpose(*result.dims)
    result[
        {"commodity": ~check_usage(technologies.comm_usage, CommodityUsage.PRODUCT)}
    ] = 0

    return result
