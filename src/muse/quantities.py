"""Collection of functions to compute model quantities.

This module is meant to collect functions computing quantities of interest to the model,
e.g. maximum production for a given capacity, etc, especially where these
functions are used in different areas of the model.

Functions for calculating costs (e.g. LCOE, EAC) are in the `costs` module.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from muse.timeslices import broadcast_timeslice, distribute_timeslice


def supply(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: xr.Dataset | xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Production and emission for a given capacity servicing a given demand.

    Supply includes two components, end-uses outputs and environmental pollutants. The
    former consists of the demand that the current capacity is capable of servicing.
    Where there is excess capacity, then service is assigned to each asset a share of
    the maximum production (e.g. utilization across similar assets is the same in
    percentage). Then, environmental pollutants are computing as a function of
    commodity outputs.

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


def emission(
    production: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
):
    """Computes emission from current products.

    Arguments:
        production: Commodity-level production for a series of assets.
        technologies: `technologies` dataset containing `fixed_output`.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        A data array containing emissions (and only emissions).
    """
    from muse.commodities import is_pollutant

    # Calculate the production amplitude of each asset
    prod_amplitude = production_amplitude(production, technologies)

    # Calculate the production of environmental pollutants
    # = prod_amplitude * fixed_outputs
    envs = is_pollutant(technologies.comm_usage)
    envs_production = prod_amplitude * broadcast_timeslice(
        technologies.sel(commodity=envs).fixed_outputs, level=timeslice_level
    )
    return envs_production


def consumption(
    technologies: xr.Dataset,
    production: xr.DataArray,
    prices: xr.DataArray | None = None,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Commodity consumption when fulfilling the whole production.

    Firstly, the degree of technology activity is calculated (i.e. the amount of
    technology flow required to meet the production). Then, the consumption of fixed
    commodities is calculated in proportion to this activity.

    In addition, if there are flexible inputs, then the single lowest-cost option is
    selected (minimising price * quantity). If prices are not given, then flexible
    consumption is *not* considered.

    Arguments:
        technologies: Dataset of technology parameters. Must contain `fixed_inputs`,
            `flexible_inputs`, and `fixed_outputs`.
        production: DataArray of production data. Must have "timeslice" and "commodity"
            dimensions.
        prices: DataArray of prices for each commodity. Must have "timeslice" and
            "commodity" dimensions. If not given, then flexible inputs are not
            considered.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        A data array containing the consumption of each commodity. Will have the same
        dimensions as `production`.

    """
    # Calculate degree of technology activity
    prod_amplitude = production_amplitude(production, technologies)

    # Calculate consumption of fixed commodities
    consumption_fixed = prod_amplitude * broadcast_timeslice(
        technologies.fixed_inputs, level=timeslice_level
    )
    assert all(consumption_fixed.commodity.values == production.commodity.values)

    # If there are no flexible inputs, then we are done
    if not (technologies.flexible_inputs > 0).any():
        return consumption_fixed

    # If prices are not given, then we can't consider flexible inputs, so just return
    # the fixed consumption
    if prices is None:
        return consumption_fixed

    # Flexible inputs
    flexs = broadcast_timeslice(technologies.flexible_inputs, level=timeslice_level)

    # Calculate the cheapest fuel for each flexible technology
    priceflex = prices * flexs
    minprices = flexs.commodity[
        priceflex.where(flexs > 0, priceflex.max() + 1).argmin("commodity")
    ]

    # Consumption of flexible commodities
    assert all(flexs.commodity.values == consumption_fixed.commodity.values)
    flex = flexs.where(
        broadcast_timeslice(flexs.commodity, level=timeslice_level) == minprices, 0
    )
    consumption_flex = flex * prod_amplitude
    return consumption_fixed + consumption_flex


def maximum_production(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    timeslice_level: str | None = None,
    **filters,
):
    r"""Production for a given capacity.

    Given a capacity :math:`\mathcal{A}_{t, \iota}^r`, the utilization factor
    :math:`\alpha^r_{t, \iota}` and the the fixed outputs of each technology
    :math:`\beta^r_{t, \iota, c}`, then the result production is:

    .. math::

        P_{t, \iota}^r =
            \alpha^r_{t, \iota}\beta^r_{t, \iota, c}\mathcal{A}_{t, \iota}^r

    The dimensions above are only indicative. The function should work with many
    different input values, e.g. with capacities expanded over time-slices :math:`t` or
    agents :math:`i`.

    Arguments:
        capacity: Capacity of each technology of interest. In practice, the capacity can
            refer to asset capacity, the max capacity, or the capacity-in-use.
        technologies: xr.Dataset describing the features of the technologies of
            interests.  It should contain `fixed_outputs` and `utilization_factor`.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        `capacity * fixed_outputs * utilization_factor`, whittled down according to the
        filters and the set of technologies in `capacity`.
    """
    from muse.commodities import is_enduse

    capa = capacity.sel(**{k: v for k, v in filters.items() if k in capacity.dims})
    ftechs = technologies.sel(
        **{k: v for k, v in filters.items() if k in technologies.dims}
    )

    result = (
        broadcast_timeslice(capa, level=timeslice_level)
        * distribute_timeslice(ftechs.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(ftechs.utilization_factor, level=timeslice_level)
    )
    return result.where(is_enduse(ftechs.comm_usage), 0)


def capacity_in_use(
    production: xr.DataArray,
    technologies: xr.Dataset,
    max_dim: str | tuple[str] | None = "commodity",
    timeslice_level: str | None = None,
    **filters,
):
    """Capacity-in-use for each asset, given production.

    Conceptually, this operation is the inverse of `production`.

    Arguments:
        production: Production from each technology of interest.
        technologies: xr.Dataset describing the features of the technologies of
            interests.  It should contain `fixed_outputs` and `utilization_factor`.
        max_dim: reduces the given dimensions using `max`. Defaults to "commodity". If
            None, then no reduction is performed.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        Capacity-in-use for each technology, whittled down by the filters.
    """
    from muse.commodities import is_enduse

    prod = production.sel(**{k: v for k, v in filters.items() if k in production.dims})
    ftechs = technologies.sel(
        **{k: v for k, v in filters.items() if k in technologies.dims}
    )

    factor = 1 / (ftechs.fixed_outputs * ftechs.utilization_factor)
    capa_in_use = (prod * broadcast_timeslice(factor, level=timeslice_level)).where(
        ~np.isinf(factor), 0
    )

    capa_in_use = capa_in_use.where(
        is_enduse(technologies.comm_usage.sel(commodity=capa_in_use.commodity)), 0
    )
    if max_dim:
        capa_in_use = capa_in_use.max(max_dim)

    return capa_in_use


def minimum_production(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    timeslice_level: str | None = None,
    **filters,
):
    r"""Minimum production for a given capacity.

    Given a capacity :math:`\mathcal{A}_{t, \iota}^r`, the minimum service factor
    :math:`\alpha^r_{t, \iota}` and the the fixed outputs of each technology
    :math:`\beta^r_{t, \iota, c}`, then the result production is:

    .. math::

        P_{t, \iota}^r =
            \alpha^r_{t, \iota}\beta^r_{t, \iota, c}\mathcal{A}_{t, \iota}^r

    The dimensions above are only indicative. The function should work with many
    different input values, e.g. with capacities expanded over time-slices :math:`t` or
    agents :math:`i`.

    Arguments:
        capacity: Capacity of each technology of interest. In practice, the capacity can
            refer to asset capacity, the max capacity, or the capacity-in-use.
        technologies: xr.Dataset describing the features of the technologies of
            interests.  It should contain `fixed_outputs` and `minimum_service_factor`.
        timeslices: xr.DataArray of the timeslicing scheme. Production data will be
            returned in this format.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        `capacity * fixed_outputs * minimum_service_factor`, whittled down according to
        the filters and the set of technologies in `capacity`.
    """
    from muse.commodities import is_enduse

    capa = capacity.sel(**{k: v for k, v in filters.items() if k in capacity.dims})
    ftechs = technologies.sel(
        **{k: v for k, v in filters.items() if k in technologies.dims}
    )
    result = (
        broadcast_timeslice(capa, level=timeslice_level)
        * distribute_timeslice(ftechs.fixed_outputs, level=timeslice_level)
        * broadcast_timeslice(ftechs.minimum_service_factor, level=timeslice_level)
    )
    return result.where(is_enduse(ftechs.comm_usage), 0)


def capacity_to_service_demand(
    demand: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Minimum capacity required to fulfill the demand."""
    timeslice_outputs = distribute_timeslice(
        technologies.fixed_outputs.sel(commodity=demand.commodity),
        level=timeslice_level,
    ) * broadcast_timeslice(technologies.utilization_factor, level=timeslice_level)
    capa_to_service_demand = demand / timeslice_outputs
    return capa_to_service_demand.where(np.isfinite(capa_to_service_demand), 0).max(
        ("commodity", "timeslice")
    )


def production_amplitude(
    production: xr.DataArray,
    technologies: xr.Dataset,
) -> xr.DataArray:
    """Calculates the degree of technology activity based on production data.

    We do this by dividing the production data by the output flow per unit of activity.
    Taking the max of this across all commodities, we get the minimum units of
    technology activity required to meet (at least) the specified production of all
    commodities.

    For example:
    A technology has the following reaction: 1A -> 2B + 3C
    If production is 4B & 6C, this is equal to a production amplitude of 2

    Args:
        production: DataArray with commodity-level production for a set of technologies.
            Must have `timeslice` and `commodity` dimensions. May also have other
            dimensions e.g. `region`, `year`, etc.
        technologies: Dataset of technology parameters

    Returns:
        DataArray with production amplitudes for each technology in each timeslice.
        Will have the same dimensions as `production`, minus the `commodity` dimension.
    """
    from muse.timeslices import get_level

    assert set(technologies.dims).issubset(set(production.dims))
    timeslice_level = get_level(production)

    return (
        production
        / broadcast_timeslice(technologies.fixed_outputs, level=timeslice_level)
    ).max("commodity")
