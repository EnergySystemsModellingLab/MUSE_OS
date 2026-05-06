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
        demand: xr.DataArray, capacity: xr.DataArray, technologies: xr.Dataset, **kwargs
    ) -> xr.DataArray:
        pass


Arguments:
    demand: The demand for each commodity.
    capacity: The capacity of each asset within a market.
    technologies: A dataset characterising the technologies of the same assets.
    **kwargs: Any number of keyword arguments

Returns:
    A `xr.DataArray` with the amount produced for each good from each asset.
"""

from __future__ import annotations

__all__ = [
    "PRODUCTION_SIGNATURE",
    "dispatch_by_merit_order",
    "factory",
    "maximum_production",
    "merit_order_production",
    "register_production",
    "share_based_production",
]

from collections.abc import MutableMapping
from typing import Any, Protocol

import numpy as np
import xarray as xr

from muse.registration import registrator


class PRODUCTION_SIGNATURE(Protocol):
    """Callable protocol for registered production methods."""

    def __call__(
        self,
        demand: xr.DataArray,
        capacity: xr.DataArray,
        technologies: xr.Dataset,
        timeslice_level: str | None = None,
        **kwargs: Any,
    ) -> xr.DataArray: ...


"""Dictionary of production methods. """
PRODUCTION_METHODS: MutableMapping[str, PRODUCTION_SIGNATURE] = {}


@registrator(registry=PRODUCTION_METHODS, loglevel="info")
def register_production(function: PRODUCTION_SIGNATURE):
    """Decorator to register a function as a production method.

    .. seealso::

        :py:mod:`muse.dispatch`
    """
    return function


def factory(name) -> PRODUCTION_SIGNATURE:
    from muse.dispatch import PRODUCTION_METHODS

    return PRODUCTION_METHODS[name]


@register_production(name=("max", "maximum"))
def maximum_production(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
    **kwargs,
) -> xr.DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilization factor. For more details, see
    :py:func:`muse.quantities.maximum_production`.
    """
    from muse.quantities import maximum_production

    return maximum_production(technologies, capacity, timeslice_level)


@register_production(name=("share", "shares"))
def share_based_production(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
    **kwargs,
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
        demand: amount of each end-use required. The supply of each process will not
            exceed its share of the demand.
        capacity: number/quantity of assets that can service the demand
        technologies: factors bindings the capacity of an asset with its production of
            commodities and environmental pollutants.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")
        **kwargs: any number of keyword arguments (not used in this method)

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


@register_production(name=("merit", "merit-order"))
def merit_order_production(
    demand: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
    *,
    prices: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """Service demand by preferentially dispatching the cheapest assets.

    This function allocates production across a set of assets by dispatching them
    in order of increasing marginal cost (i.e. cheapest first), until demand is met.
    Each asset is operated between its minimum and maximum production constraints,
    and higher-cost assets are only used once cheaper options are exhausted.

    The merit-order principle is widely used in systems where:
    - Multiple assets can supply a homogeneous or substitutable service
    - Supply is pooled or centrally coordinated
    - Short-run marginal cost is the dominant driver of dispatch decisions

    A good example is the power sector (electricity production), where supply from all
    assets feeds into a common grid. Low cost generators are drawn on first, followed by
    more expensive generators as grid demand increases.

    Each timeslice and year is treated independently, so the dispatch order can vary
    across timeslices and years. For a numerical example in a single
    timeslice/year/region, see the docstring of :py:func:`dispatch_by_merit_order`.
    """
    from muse.commodities import CommodityUsage, check_usage, is_pollutant
    from muse.costs import marginal_cost
    from muse.quantities import (
        consumption,
        emission,
        maximum_production,
        minimum_production,
    )
    from muse.utilities import broadcast_over_assets

    assert "asset" not in demand.dims
    assert "asset" in capacity.dims
    assert "asset" not in prices.dims

    # Normalise demand/prices dataarrays to ensure they have a region dimension
    # Multi-region models will already have a region dimension
    if "region" not in demand.dims:
        region = np.unique(technologies.region.values).item()
        demand = demand.expand_dims(region=[region])
    if "region" not in prices.dims:
        region = np.unique(technologies.region.values).item()
        prices = prices.expand_dims(region=[region])

    # Normalise technologies/capacity dataarrays to ensure that the region coordinate is
    # aligned with the asset coordinate
    # This will already be the case for multi-region models
    if "asset" not in technologies.region.dims:
        technologies = technologies.assign_coords(
            region=("asset", [technologies.region.item()] * technologies.sizes["asset"])
        )
    if "asset" not in capacity.region.dims:
        capacity = capacity.assign_coords(
            region=("asset", [capacity.region.item()] * capacity.sizes["asset"])
        )

    # Maximum and minimum production for each asset
    maxprod = maximum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )
    minprod = minimum_production(
        technologies, capacity, timeslice_level=timeslice_level
    )

    # Consumption of each asset assuming full dispatch, for calculating costs later on.
    maxcons = consumption(
        technologies,
        maxprod,
        prices=broadcast_over_assets(prices, capacity, installed_as_year=False),
        timeslice_level=timeslice_level,
    )

    # Verify all asset-level inputs are positionally aligned before using isel.
    xr.align(
        technologies, capacity, maxprod, minprod, maxcons, join="exact", copy=False
    )

    # Initialise result with zeros
    result = xr.zeros_like(maxprod)

    for y in maxprod.year.values:
        prices_y = prices.sel(year=y)
        maxprod_y = maxprod.sel(year=y)
        minprod_y = minprod.sel(year=y)
        maxcons_y = maxcons.sel(year=y)

        for region in demand.region.values:
            region_assets = maxprod_y.get_index("asset")[
                maxprod_y.region.values == region
            ]
            maxprod_region = maxprod_y.sel(asset=region_assets)
            techs_region = technologies.sel(asset=region_assets)
            minprod_region = minprod_y.sel(asset=region_assets)

            # Calculate timeslice-level costs for each asset in this year assuming full
            # dispatch. We use LCOE excluding capital costs.
            technology_costs = marginal_cost(
                techs_region,
                prices_y.sel(region=region),
                production=maxprod_region,
                consumption=maxcons_y.sel(asset=region_assets),
            )

            # Calculate production for this year by dispatching assets in order of
            # increasing cost until demand is met
            for ts in maxprod_y.timeslice.values:
                result.loc[dict(year=y, timeslice=ts, asset=region_assets)] = (
                    dispatch_by_merit_order(
                        demand=demand.sel(year=y, timeslice=ts, region=region),
                        minprod=minprod_region.sel(timeslice=ts),
                        maxprod=maxprod_region.sel(timeslice=ts),
                        technology_costs=technology_costs.sel(timeslice=ts),
                    )
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


def dispatch_by_merit_order(
    demand: xr.DataArray,
    minprod: xr.DataArray,
    maxprod: xr.DataArray,
    technology_costs: xr.DataArray,
) -> xr.DataArray:
    """Dispatch assets in order of increasing cost until demand is met.

    For example, we have the following three assets (ordered from cheapest to most
    expensive):
    - Asset A: minprod=10, maxprod=100, cost=5
    - Asset B: minprod=20, maxprod=50, cost=10
    - Asset C: minprod=0, maxprod=30, cost=15

    If the total demand is 140 units, we would dispatch as follows:
    - First we dispatch Assets A and B at their minimum production, which gives us 30
    units of supply, leaving 110 units of remaining demand.
    - Next we dispatch Asset A up to its maximum, which gives us an additional 90 units
    of supply, leaving 20 units of remaining demand.
    - The remaining demand of 20 is less than the headroom of Asset B above its
    minimum (i.e. 50 - 20 = 30), so we dispatch Asset B for an additional 20 units,
    which meets the total demand of 140.
    - Asset C is not dispatched at all since it's the most expensive and demand is
    already met by cheaper assets.

    Arguments:
        demand: DataArray of demand values, which should have a single 'commodity'
            dimension
        minprod: DataArray of minimum production values for the candidate assets, with
            dimensions 'asset' and 'commodity'
        maxprod: DataArray of maximum production values for the candidate assets, with
            dimensions 'asset' and 'commodity'
        technology_costs: DataArray of technology costs for the candidate assets, which
            should have a single 'asset' dimension. This is the cost per unit of
            production for each asset, which determines the order of dispatch (cheapest
            first).
    """
    assert set(demand.dims) == {"commodity"}
    assert set(minprod.dims) == {"asset", "commodity"}
    assert set(maxprod.dims) == {"asset", "commodity"}
    assert set(technology_costs.dims) == {"asset"}

    # Start with minimum production
    result = minprod.copy()

    # Calculate remaining demand after minimum production (i.e. what we need to meet
    # with the available headroom above minimum)
    remaining = demand - minprod.sum("asset")

    # Rank assets by cost (cheapest first)
    order = technology_costs.sortby(technology_costs).asset

    # Available extra production above minimum, sorted by cost
    available = (maxprod - minprod).sel(asset=order)

    # cumsum_before[i] = total available production of all assets cheaper than
    # i
    cumsum_before = available.cumsum("asset") - available

    # (remaining - cumsum_before)[i] = demand still unmet after cheaper assets
    # are fully utilised. Clipping to [0, available[i]] ensures we never
    # produce negative amounts or exceed this asset's headroom above minprod.
    addition = (remaining - cumsum_before).clip(min=0, max=available)

    # Final production is minimum production plus any additional production from these
    # assets
    result = minprod + addition
    return result
