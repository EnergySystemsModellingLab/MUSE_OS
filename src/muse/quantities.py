"""Collection of functions to compute model quantities.

This module is meant to collect functions computing quantities of interest to the model,
e.g. maximum production for a given capacity, etc, especially where these
functions are used in different areas of the model.

Functions for calculating costs (e.g. LCOE, EAC) are in the `costs` module.
"""

from collections.abc import Sequence
from typing import Callable, Optional, Union, cast

import numpy as np
import xarray as xr


def supply(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: Union[xr.Dataset, xr.DataArray],
    interpolation: str = "linear",
    production_method: Optional[Callable] = None,
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
        interpolation: Interpolation type
        production_method: Production for a given capacity

    Return:
        A data array where the commodity dimension only contains actual outputs (i.e. no
        input commodities).
    """
    from muse.commodities import CommodityUsage, check_usage, is_pollutant
    from muse.timeslices import broadcast_timeslice

    if production_method is None:
        production_method = maximum_production

    maxprod = production_method(technologies, capacity)
    minprod = minimum_production(technologies, capacity)
    size = np.array(maxprod.region).size
    # in presence of trade demand needs to map maxprod dst_region
    if (
        "region" in demand.dims
        and "region" in maxprod.coords
        and "dst_region" not in maxprod.dims
        and size == 1
    ):
        demand = demand.sel(region=maxprod.region)
        prodsum = set(demand.dims).difference(maxprod.dims)
        demsum = set(maxprod.dims).difference(demand.dims)
        expanded_demand = (demand * maxprod / maxprod.sum(demsum)).fillna(0)

    elif (
        "region" in demand.dims
        and "region" in maxprod.coords
        and "dst_region" not in maxprod.dims
        and size > 1
    ):
        prodsum = set(demand.dims).difference(maxprod.dims)
        demsum = set(maxprod.dims).difference(demand.dims)
        expanded_demand = (demand * maxprod / maxprod.sum(demsum)).fillna(0)

    elif (
        "region" in demand.dims
        and "region" in maxprod.coords
        and "dst_region" in maxprod.dims
    ):
        demand = demand.rename(region="dst_region")
        prodsum = {"timeslice"}
        demsum = {"asset"}
        expanded_demand = (demand * maxprod / maxprod.sum(demsum)).fillna(0)

    else:
        prodsum = set(demand.dims).difference(maxprod.dims)
        demsum = set(maxprod.dims).difference(demand.dims)
        expanded_demand = (demand * maxprod / maxprod.sum(demsum)).fillna(0)

    expanded_maxprod = (
        maxprod * demand / broadcast_timeslice(demand.sum(prodsum))
    ).fillna(0)
    expanded_minprod = (
        minprod * demand / broadcast_timeslice(demand.sum(prodsum))
    ).fillna(0)
    expanded_demand = expanded_demand.reindex_like(maxprod)
    expanded_minprod = expanded_minprod.reindex_like(maxprod)

    result = expanded_demand.where(
        expanded_demand <= expanded_maxprod, expanded_maxprod
    )
    result = result.where(result >= expanded_minprod, expanded_minprod)

    # add production of environmental pollutants
    env = is_pollutant(technologies.comm_usage)
    result[{"commodity": env}] = emission(result, technologies.fixed_outputs).transpose(
        *result.dims
    )
    result[
        {"commodity": ~check_usage(technologies.comm_usage, CommodityUsage.PRODUCT)}
    ] = 0

    return result


def emission(production: xr.DataArray, fixed_outputs: xr.DataArray):
    """Computes emission from current products.

    Emissions are computed as `sum(product) * fixed_outputs`.

    Arguments:
        production: Produced goods. Only those with non-environmental products are used
            when computing emissions.
        fixed_outputs: factor relating total production to emissions. For convenience,
            this can also be a `technologies` dataset containing `fixed_output`.

    Return:
        A data array containing emissions (and only emissions).
    """
    from muse.commodities import is_enduse, is_pollutant
    from muse.timeslices import broadcast_timeslice
    from muse.utilities import broadcast_techs

    # just in case we are passed a technologies dataset, like in other functions
    fouts = broadcast_techs(
        getattr(fixed_outputs, "fixed_outputs", fixed_outputs), production
    )
    envs = is_pollutant(fouts.comm_usage)
    enduses = is_enduse(fouts.comm_usage)
    return production.sel(commodity=enduses).sum("commodity") * broadcast_timeslice(
        fouts.sel(commodity=envs)
    )


def gross_margin(
    technologies: xr.Dataset, capacity: xr.DataArray, prices: xr.Dataset
) -> xr.DataArray:
    """The percentage of revenue after direct expenses have been subtracted.

    .. _reference:
    https://www.investopedia.com/terms/g/grossmargin.asp
    We first calculate the revenues, which depend on prices
    We then deduct the direct expenses
    - energy commodities INPUTS are related to fuel costs
    - environmental commodities OUTPUTS are related to environmental costs
    - variable costs is given as technodata inputs
    - non-environmental commodities OUTPUTS are related to revenues.
    """
    from muse.commodities import is_enduse, is_pollutant
    from muse.timeslices import distribute_timeslice
    from muse.utilities import broadcast_techs

    tech = broadcast_techs(  # type: ignore
        cast(
            xr.Dataset,
            technologies[
                [
                    "technical_life",
                    "interest_rate",
                    "var_par",
                    "var_exp",
                    "fixed_outputs",
                    "fixed_inputs",
                ]
            ],
        ),
        capacity,
    )

    var_par = tech.var_par
    var_exp = tech.var_exp
    fixed_outputs = tech.fixed_outputs
    fixed_inputs = tech.fixed_inputs
    # We separate the case where we have one or more regions
    caparegions = np.array(capacity.region.values).reshape(-1)
    if len(caparegions) > 1:
        prices.sel(region=capacity.region)
    else:
        prices = prices.where(prices.region == capacity.region, drop=True)
    prices = prices.interp(year=capacity.year.values)

    # Filters for pollutants and output commodities
    environmentals = is_pollutant(technologies.comm_usage)
    enduses = is_enduse(technologies.comm_usage)

    # Variable costs depend on factors such as labour
    variable_costs = distribute_timeslice(
        var_par * ((fixed_outputs.sel(commodity=enduses)).sum("commodity")) ** var_exp,
    )

    # The individual prices are selected
    # costs due to consumables, direct inputs
    consumption_costs = (prices * distribute_timeslice(fixed_inputs)).sum("commodity")
    # costs due to pollutants
    production_costs = prices * distribute_timeslice(fixed_outputs)
    environmental_costs = (production_costs.sel(commodity=environmentals)).sum(
        "commodity"
    )
    # revenues due to product sales
    revenues = (production_costs.sel(commodity=enduses)).sum("commodity")

    # Gross margin is the net between revenues and all costs
    result = revenues - environmental_costs - variable_costs - consumption_costs

    # Gross margin is defined as a ratio on revenues and as a percentage
    result *= 100 / revenues
    return result


def decommissioning_demand(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    year: Optional[Sequence[int]] = None,
) -> xr.DataArray:
    r"""Computes demand from process decommissioning.

    If `year` is not given, it defaults to all years in capacity. If there are more than
    two years, then decommissioning is with respect to first (or minimum) year.

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
    if year is None:
        year = capacity.year.values
    year = sorted(year)
    capacity = capacity.interp(year=year, kwargs={"fill_value": 0.0})
    baseyear = min(year)
    dyears = [u for u in year if u != baseyear]

    # Calculate the decrease in capacity from the current year to future years
    capacity_decrease = capacity.sel(year=baseyear) - capacity.sel(year=dyears)

    # Calculate production associated with this capacity
    return maximum_production(
        technologies,
        capacity_decrease,
    ).clip(min=0)


def consumption(
    technologies: xr.Dataset,
    production: xr.DataArray,
    prices: Optional[xr.DataArray] = None,
    **kwargs,
) -> xr.DataArray:
    """Commodity consumption when fulfilling the whole production.

    Currently, the consumption is implemented for commodity_max == +infinity. If prices
    are not given, then flexible consumption is *not* considered.
    """
    from muse.commodities import is_fuel
    from muse.timeslices import broadcast_timeslice
    from muse.utilities import filter_with_template

    params = filter_with_template(
        technologies[["fixed_inputs", "flexible_inputs", "fixed_outputs"]],
        production,
        **kwargs,
    )
    params_fuels = is_fuel(params.comm_usage)

    # Calculate degree of technology activity
    # We do this by dividing the production by the output flow per unit of activity
    prod_amplitude = (production / broadcast_timeslice(params.fixed_outputs)).max(
        "commodity"
    )

    # Calculate consumption of fixed commodities
    consumption_fixed = prod_amplitude * broadcast_timeslice(params.fixed_inputs)

    # If there are no flexible inputs, then we are done
    if not (params.flexible_inputs.sel(commodity=params_fuels) > 0).any():
        return consumption_fixed

    # If prices are not given, then we can't consider flexible inputs, so just return
    # the fixed consumption
    if prices is None:
        return consumption_fixed

    # Flexible inputs
    flexs = params.flexible_inputs.where(params_fuels, 0)

    # Calculate the cheapest fuel for each flexible technology
    priceflex = prices.loc[dict(commodity=flexs.commodity)] * flexs
    minprices = flexs.commodity[
        priceflex.where(flexs > 0, priceflex.max() + 1).argmin("commodity")
    ]

    # Consumption of flexible commodities
    assert all(flexs.commodity.values == consumption_fixed.commodity.values)
    flex = flexs.where(broadcast_timeslice(flexs.commodity) == minprices, 0)
    consumption_flex = flex * prod_amplitude
    return consumption_fixed + consumption_flex


def maximum_production(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
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
            interests.  It should contain `fixed_outputs` and `utilization_factor`. It's
            shape is matched to `capacity` using `muse.utilities.broadcast_techs`.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.

    Return:
        `capacity * fixed_outputs * utilization_factor`, whittled down according to the
        filters and the set of technologies in `capacity`.
    """
    from muse.commodities import is_enduse
    from muse.timeslices import broadcast_timeslice, distribute_timeslice
    from muse.utilities import broadcast_techs, filter_input

    capa = filter_input(
        capacity, **{k: v for k, v in filters.items() if k in capacity.dims}
    )
    btechs = broadcast_techs(  # type: ignore
        cast(xr.Dataset, technologies[["fixed_outputs", "utilization_factor"]]), capa
    )
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in btechs.dims}
    )
    result = (
        broadcast_timeslice(capa)
        * distribute_timeslice(ftechs.fixed_outputs)
        * broadcast_timeslice(ftechs.utilization_factor)
    )
    return result.where(is_enduse(result.comm_usage), 0)


def capacity_in_use(
    production: xr.DataArray,
    technologies: xr.Dataset,
    max_dim: Optional[Union[str, tuple[str]]] = "commodity",
    **filters,
):
    """Capacity-in-use for each asset, given production.

    Conceptually, this operation is the inverse of `production`.

    Arguments:
        production: Production from each technology of interest.
        technologies: xr.Dataset describing the features of the technologies of
            interests.  It should contain `fixed_outputs` and `utilization_factor`. It's
            shape is matched to `capacity` using `muse.utilities.broadcast_techs`.
        max_dim: reduces the given dimensions using `max`. Defaults to "commodity". If
            None, then no reduction is performed.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.

    Return:
        Capacity-in-use for each technology, whittled down by the filters.
    """
    from muse.commodities import is_enduse
    from muse.timeslices import broadcast_timeslice
    from muse.utilities import broadcast_techs, filter_input

    prod = filter_input(
        production, **{k: v for k, v in filters.items() if k in production.dims}
    )

    techs = technologies[["fixed_outputs", "utilization_factor"]]
    assert isinstance(techs, xr.Dataset)
    btechs = broadcast_techs(techs, prod)
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in technologies.dims}
    )

    factor = 1 / (ftechs.fixed_outputs * ftechs.utilization_factor)
    capa_in_use = (prod * broadcast_timeslice(factor)).where(~np.isinf(factor), 0)

    capa_in_use = capa_in_use.where(
        is_enduse(technologies.comm_usage.sel(commodity=capa_in_use.commodity)), 0
    )
    if max_dim:
        capa_in_use = capa_in_use.max(max_dim)

    return capa_in_use


def minimum_production(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
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
            Its shape is matched to `capacity` using `muse.utilities.broadcast_techs`.
        timeslices: xr.DataArray of the timeslicing scheme. Production data will be
            returned in this format.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `technologies`, are
            silently ignored.

    Return:
        `capacity * fixed_outputs * minimum_service_factor`, whittled down according to
        the filters and the set of technologies in `capacity`.
    """
    from muse.commodities import is_enduse
    from muse.timeslices import broadcast_timeslice, distribute_timeslice
    from muse.utilities import broadcast_techs, filter_input

    capa = filter_input(
        capacity, **{k: v for k, v in filters.items() if k in capacity.dims}
    )

    if "minimum_service_factor" not in technologies:
        return broadcast_timeslice(xr.zeros_like(capa))

    btechs = broadcast_techs(  # type: ignore
        cast(
            xr.Dataset,
            technologies[["fixed_outputs", "minimum_service_factor"]],
        ),
        capa,
    )
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in btechs.dims}
    )
    result = (
        broadcast_timeslice(capa)
        * distribute_timeslice(ftechs.fixed_outputs)
        * broadcast_timeslice(ftechs.minimum_service_factor)
    )
    return result.where(is_enduse(result.comm_usage), 0)


def capacity_to_service_demand(
    demand: xr.DataArray,
    technologies: xr.Dataset,
) -> xr.DataArray:
    """Minimum capacity required to fulfill the demand."""
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    timeslice_outputs = distribute_timeslice(
        technologies.fixed_outputs.sel(commodity=demand.commodity)
    ) * broadcast_timeslice(technologies.utilization_factor)
    capa_to_service_demand = demand / timeslice_outputs
    return capa_to_service_demand.max(("commodity", "timeslice"))
