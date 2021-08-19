"""Collection of functions to compute model quantities.

This module is meant to collect functions computing quantities of interest to the model,
e.g. lcoe, maximum production for a given capacity, etc, especially where these
functions are used in different areas of the model.
"""
from typing import Callable, Optional, Sequence, Text, Tuple, Union, cast

import numpy as np
import xarray as xr


def supply(
    capacity: xr.DataArray,
    demand: xr.DataArray,
    technologies: Union[xr.Dataset, xr.DataArray],
    interpolation: Text = "linear",
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
            exceed it's share of the demand.
        technologies: factors bindings the capacity of an asset with its production of
            commodities and environmental pollutants.

    Return:
        A data array where the commodity dimension only contains actual outputs (i.e. no
        input commodities).
    """
    from muse.commodities import is_pollutant, CommodityUsage, check_usage

    if production_method is None:
        production_method = maximum_production

    maxprod = production_method(technologies, capacity)
    if "region" in demand.dims and "region" in maxprod.coords:
        demand = demand.sel(region=maxprod.region)
    expanded_maxprod = (
        maxprod * demand / demand.sum(set(demand.dims).difference(maxprod.dims))
    ).fillna(0)
    expanded_demand = (
        demand * maxprod / maxprod.sum(set(maxprod.dims).difference(demand.dims))
    ).fillna(0)
    expanded_demand = expanded_demand.reindex_like(maxprod)

    result = expanded_demand.where(
        expanded_demand <= expanded_maxprod, expanded_maxprod
    )

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
    from muse.utilities import broadcast_techs
    from muse.commodities import is_enduse, is_pollutant

    # just in case we are passed a technologies dataset, like in other functions
    fouts = broadcast_techs(
        getattr(fixed_outputs, "fixed_outputs", fixed_outputs), production
    )
    envs = is_pollutant(fouts.comm_usage)
    enduses = is_enduse(fouts.comm_usage)
    return production.sel(commodity=enduses).sum("commodity") * fouts.sel(
        commodity=envs
    )


def gross_margin(
    technologies: xr.Dataset, capacity: xr.DataArray, prices: xr.Dataset
) -> xr.DataArray:
    """profit of increasing the production by one unit.

    - energy commodities INPUTS are related to fuel costs
    - environmental commodities OUTPUTS are related to environmental costs
    - variable costs is given as technodata inputs
    - non-environmental commodities OUTPUTS are related to revenues
    """
    from muse.utilities import broadcast_techs
    from muse.commodities import is_pollutant, is_enduse
    from muse.timeslices import convert_timeslice, QuantityType

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

    # Hours ratio
    variable_costs = convert_timeslice(
        var_par * capacity ** var_exp, prices.timeslice, QuantityType.EXTENSIVE
    )

    prices = prices.sel(region=capacity.region).interp(year=capacity.year)

    # Filters
    environmentals = is_pollutant(technologies.comm_usage)
    enduses = is_enduse(technologies.comm_usage)

    # The individual prices
    consumption_costs = (prices * fixed_inputs).sum("commodity")
    production_costs = prices * fixed_outputs
    environmental_costs = (production_costs.sel(commodity=environmentals)).sum(
        ("commodity")
    )

    revenues = (production_costs.sel(commodity=enduses)).sum("commodity")

    result = revenues - environmental_costs - variable_costs - consumption_costs
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
    return maximum_production(
        technologies, capacity.sel(year=baseyear) - capacity.sel(year=dyears)
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
    from muse.utilities import filter_with_template
    from muse.commodities import is_fuel, is_enduse
    from muse.timeslices import convert_timeslice, QuantityType

    params = filter_with_template(
        technologies[["fixed_inputs", "flexible_inputs"]], production, **kwargs
    )

    # sum over end-use products, if the dimension exists in the input

    comm_usage = technologies.comm_usage.sel(commodity=production.commodity)

    production = production.sel(commodity=is_enduse(comm_usage)).sum("commodity")

    if prices is not None and "timeslice" in prices.dims:
        production = convert_timeslice(  # type: ignore
            production, prices, QuantityType.EXTENSIVE
        )

    params_fuels = is_fuel(params.comm_usage)
    consumption = production * params.fixed_inputs.where(params_fuels, 0)

    if prices is None:
        return consumption

    if not (params.flexible_inputs.sel(commodity=params_fuels) > 0).any():
        return consumption

    prices = filter_with_template(prices, production, installed_as_year=False, **kwargs)
    # technology with flexible inputs
    flexs = params.flexible_inputs.where(params_fuels, 0)
    # cheapest fuel for each flexible technology
    assert prices is not None
    assert all(flexs.commodity.values == prices.commodity.values)
    minprices = flexs.commodity[
        prices.where(flexs > 0, prices.max() + 1).argmin("commodity")
    ]
    # add consumption from cheapest fuel
    assert all(flexs.commodity.values == consumption.commodity.values)
    flex = flexs.where(minprices == flexs.commodity, 0)
    flex = flex / (flex > 0).sum("commodity").clip(min=1)
    return consumption + flex * production


def annual_levelized_cost_of_energy(
    prices: xr.DataArray,
    technologies: xr.Dataset,
    interpolation: Text = "linear",
    fill_value: Union[int, Text] = "extrapolate",
    **filters,
) -> xr.DataArray:
    """Undiscounted levelized cost of energy (LCOE) of technologies on each given year.

    It mostly follows the `simplified LCOE`_ given by NREL. In the argument description,
    we use the following:

    * [h]: hour
    * [y]: year
    * [$]: unit of currency
    * [E]: unit of energy
    * [1]: dimensionless

    Arguments:
        prices: [$/(Eh)] the price of all commodities, including consumables and fuels.
            This dataarray contains at least timeslice and commodity dimensions.

        technologies: Describe the technologies, with at least the following parameters:

            * cap_par: [$/E] overnight capital cost
            * interest_rate: [1]
            * fix_par: [$/(Eh)] fixed costs of operation and maintenance costs
            * var_par: [$/(Eh)] variable costs of operation and maintenance costs
            * fixed_inputs: [1] == [(Eh)/(Eh)] ratio indicating the amount of commodity
                consumed per units of energy created.
            * fixed_outputs: [1] == [(Eh)/(Eh)] ration indicating the amount of
                environmental pollutants produced per units of energy created.

        interpolation: interpolation method.
        fill_value: Fill value for values outside the extrapolation range.
        **filters: Anything by which prices can be filtered.

    Return:
        The lifetime LCOE in [$/(Eh)] for each technology at each timeslice.

    .. _simplified LCOE: https://www.nrel.gov/analysis/tech-lcoe-documentation.html
    """
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.commodities import is_pollutant

    techs = technologies[
        [
            "technical_life",
            "interest_rate",
            "cap_par",
            "var_par",
            "fix_par",
            "fixed_inputs",
            "flexible_inputs",
            "fixed_outputs",
            "utilization_factor",
        ]
    ]
    if "year" in techs.dims:
        techs = techs.interp(
            year=prices.year, method=interpolation, kwargs={"fill_value": fill_value}
        )
    if filters is not None:
        prices = prices.sel({k: v for k, v in filters.items() if k in prices.dims})
        techs = techs.sel({k: v for k, v in filters.items() if k in techs.dims})

    assert {"timeslice", "commodity"}.issubset(prices.dims)

    life = techs.technical_life.astype(int)

    annualized_capital_costs = (
        convert_timeslice(
            techs.cap_par
            * techs.interest_rate
            / (1 - (1 + techs.interest_rate) ** (-life)),
            prices.timeslice,
            QuantityType.EXTENSIVE,
        )
        / techs.utilization_factor
    )

    o_and_e_costs = (
        convert_timeslice(
            (techs.fix_par + techs.var_par),
            prices.timeslice,
            QuantityType.EXTENSIVE,
        )
        / techs.utilization_factor
    )

    fuel_costs = (techs.fixed_inputs * prices).sum("commodity")

    fuel_costs += (techs.flexible_inputs * prices).sum("commodity")
    if "region" in techs.dims:
        env_costs = (
            (techs.fixed_outputs * prices)
            .sel(region=techs.region)
            .sel(commodity=is_pollutant(techs.comm_usage))
            .sum("commodity")
        )
    else:
        env_costs = (
            (techs.fixed_outputs * prices)
            .sel(commodity=is_pollutant(techs.comm_usage))
            .sum("commodity")
        )
    return annualized_capital_costs + o_and_e_costs + env_costs + fuel_costs


def maximum_production(technologies: xr.Dataset, capacity: xr.DataArray, **filters):
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
            filters that are not a dimension of `capacity` or `techologies`, are
            silently ignored.
    Return:
        `capacity * fixed_outputs * utilization_factor`, whittled down according to the
        filters and the set of technologies in `capacity`.
    """
    from muse.utilities import filter_input, broadcast_techs
    from muse.commodities import is_enduse

    capa = filter_input(
        capacity, **{k: v for k, v in filters.items() if k in capacity.dims}
    )
    btechs = broadcast_techs(  # type: ignore
        cast(xr.Dataset, technologies[["fixed_outputs", "utilization_factor"]]), capa
    )
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in btechs.dims}
    )
    result = capa * ftechs.fixed_outputs * ftechs.utilization_factor
    return result.where(is_enduse(result.comm_usage), 0)


def demand_matched_production(
    demand: xr.DataArray,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    **filters,
) -> xr.DataArray:
    """Production matching the input demand.

    Arguments:
        demand: demand to match.
        prices: price from which to compute the annual levelized cost of energy.
        capacity: capacity from which to obtain the maximum production constraints.
        **filters: keyword arguments with which to filter the input datasets and
            data arrays., e.g. region, or year.
    """
    from muse.demand_matching import demand_matching
    from muse.utilities import broadcast_techs
    from muse.timeslices import convert_timeslice, QuantityType

    technodata = cast(xr.Dataset, broadcast_techs(technologies, capacity))
    cost = annual_levelized_cost_of_energy(prices, technodata, **filters)
    max_production = maximum_production(technodata, capacity, **filters)
    assert ("timeslice" in demand.dims) == ("timeslice" in cost.dims)
    if "timeslice" in demand.dims and "timeslice" not in max_production.dims:
        max_production = convert_timeslice(
            max_production, demand.timeslice, QuantityType.EXTENSIVE
        )
    return demand_matching(demand, cost, max_production)


def capacity_in_use(
    production: xr.DataArray,
    technologies: xr.Dataset,
    max_dim: Optional[Union[Text, Tuple[Text]]] = "commodity",
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
            filters that are not a dimension of `capacity` or `techologies`, are
            silently ignored.
    Return:
        Capacity-in-use for each technology, whittled down by the filters.
    """
    from muse.utilities import filter_input, broadcast_techs
    from muse.commodities import is_enduse

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
    capa_in_use = (prod * factor).where(~np.isinf(factor), 0)
    capa_in_use = capa_in_use.where(
        is_enduse(technologies.comm_usage.sel(commodity=capa_in_use.commodity)), 0
    )
    if max_dim:
        capa_in_use = capa_in_use.max(max_dim)
    return capa_in_use


def supply_cost(
    production: xr.DataArray, lcoe: xr.DataArray, asset_dim: Optional[Text] = "asset"
) -> xr.DataArray:
    """Supply cost given production and the levelized cost of energy.

    In practice, the supply cost is the weighted average LCOE over assets (`asset_dim`),
    where the weights are the production.

    Arguments:
        production: Amount of goods produced. In practice, production can be obtained
            from the capacity for each asset via the method
            `muse.quantities.production`.
        lcoe: Levelized cost of energy for each good produced. In practice, it can be
            obtained from market prices via
            `muse.quantities.annual_levelized_cost_of_energy` or
            `muse.quantities.lifetime_levelized_cost_of_energy`.
        asset_dim: Name of the dimension(s) holding assets, processes or technologies.
    """
    data = xr.Dataset(dict(production=production, prices=production * lcoe))
    if asset_dim is not None:
        if "region" not in data.coords or len(data.region.dims) == 0:
            data = data.sum(asset_dim)

        else:
            data = data.groupby("region").sum(asset_dim)

    total = data.production.where(np.abs(data.production) > 1e-15, np.infty).sum(
        "timeslice"
    )
    return data.prices / total


def costed_production(
    demand: xr.Dataset,
    costs: xr.DataArray,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    with_minimum_service: bool = True,
) -> xr.DataArray:
    """Computes production from ranked assets.
    The assets are ranked according to their cost. The asset with least cost are allowed
    to service the demand first, up to the maximum production. By default, the mininum
    service is applied first.
    """

    from muse.quantities import maximum_production
    from muse.utilities import broadcast_techs
    from muse.timeslices import convert_timeslice, QuantityType

    technodata = cast(xr.Dataset, broadcast_techs(technologies, capacity))

    if len(capacity.region.dims) == 0:

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return x.sum("asset")

    else:

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return xr.Dataset(dict(x=x)).groupby("region").sum("asset").x

    ranking = costs.rank("asset")
    maxprod = convert_timeslice(
        maximum_production(technodata, capacity),
        demand.timeslice,
        QuantityType.EXTENSIVE,
    )
    commodity = (maxprod > 0).any([i for i in maxprod.dims if i != "commodity"])
    commodity = commodity.drop_vars(
        [u for u in commodity.coords if u not in commodity.dims]
    )
    demand = demand.sel(commodity=commodity).copy()

    constraints = (
        xr.Dataset(dict(maxprod=maxprod, ranking=ranking, has_output=maxprod > 0))
        .set_coords("ranking")
        .set_coords("has_output")
        .sel(commodity=commodity)
    )

    if not with_minimum_service:
        production = xr.zeros_like(constraints.maxprod)
    else:
        production = (
            getattr(technodata, "minimum_service_factor", 0) * constraints.maxprod
        )
        demand = np.maximum(demand - group_assets(production), 0)

    for rank in sorted(set(constraints.ranking.values.flatten())):
        condition = (constraints.ranking == rank) & constraints.has_output
        current_maxprod = constraints.maxprod.where(condition, 0)
        fullprod = group_assets(current_maxprod)
        if (fullprod <= demand + 1e-10).all():
            current_demand = fullprod
            current_prod = current_maxprod
        else:
            if "region" in demand.dims:
                demand_prod = demand.sel(region=production.region)
            else:
                demand_prod = demand
            demand_prod = (
                current_maxprod / current_maxprod.sum("asset") * demand_prod
            ).where(condition, 0)
            current_prod = np.minimum(demand_prod, current_maxprod)
            current_demand = group_assets(current_prod)
        demand -= np.minimum(current_demand, demand)
        production = production + current_prod

    result = xr.zeros_like(maxprod)
    result[dict(commodity=commodity)] = result[dict(commodity=commodity)] + production
    return result
