"""Collection of functions to compute model quantities.

This module is meant to collect functions computing quantities of interest to the model,
e.g. lcoe, maximum production for a given capacity, etc, especially where these
functions are used in different areas of the model.
"""
from typing import Callable, Optional, Sequence, Text, Tuple, Union, cast

from xarray import DataArray, Dataset


def supply(
    capacity: DataArray,
    demand: DataArray,
    technologies: Union[Dataset, DataArray],
    interpolation: Text = "linear",
    production_method: Optional[Callable] = None,
) -> DataArray:
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
    expanded_maxprod = (
        maxprod * demand / demand.sum(set(demand.dims).difference(maxprod.dims))
    ).fillna(0)
    expanded_demand = (
        demand * maxprod / maxprod.sum(set(maxprod.dims).difference(demand.dims))
    ).fillna(0)

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


def emission(production: DataArray, fixed_outputs: DataArray):
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
    technologies: Dataset, capacity: DataArray, prices: Dataset
) -> DataArray:
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
            Dataset,
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
    technologies: Dataset, capacity: DataArray, year: Optional[Sequence[int]] = None
) -> DataArray:
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
    technologies: Dataset,
    production: DataArray,
    prices: Optional[DataArray] = None,
    **kwargs,
) -> DataArray:
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
    prices: DataArray,
    technologies: Dataset,
    interpolation: Text = "linear",
    fill_value: Union[int, Text] = "extrapolate",
    **filters,
) -> DataArray:
    """Levelized cost of energy (LCOE) of technologies on each given year.

    It mostly follows the `simplified LCOE`_ given by NREL. However, the
    units are sometimes different. In the argument description, we use the following:

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
            "interest_rate",
            "cap_par",
            "var_par",
            "fix_par",
            "fixed_inputs",
            "fixed_outputs",
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

    annualized_capital_costs = convert_timeslice(
        techs.cap_par * techs.interest_rate / (1 - 1 / (1 + techs.interest_rate)),
        prices.timeslice,
        QuantityType.EXTENSIVE,
    )

    o_and_e_costs = technologies.fix_par + technologies.var_par

    fuel_costs = (technologies.fixed_inputs * prices).sum("commodity")

    env_costs = (
        (technologies.fixed_outputs * prices)
        .sel(commodity=is_pollutant(technologies.comm_usage))
        .sum("commodity")
    )
    return annualized_capital_costs + o_and_e_costs + env_costs + fuel_costs


def lifetime_levelized_cost_of_energy(
    prices: DataArray,
    technologies: Dataset,
    installation_year: Optional[int] = None,
    **filters,
):
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    It mostly follows the `simplified LCOE` given by NREL. However, the units are
    sometimes different. In the argument description, we use the following:

    * [h]: hour
    * [y]: year
    * [$]: unit of currency
    * [E]: unit of energy
    * [1]: dimensionless

    Arguments:
        prices: [$/(Eh)] the price of all commodities, including consumables and fuels.
            This dataarray contains at least timeslice and commodity dimensions.

        technologies: Describe the technologies, with at least the following parameters:

            * technical life: [a] lifetime of each technology
            * cap_par: [$/E] overnight capital cost
            * interest_rate: [1]
            * fix_par: [$/(Eh)] fixed costs of operation and maintenance costs
            * var_par: [$/(Eh)] variable costs of operation and maintenance costs
            * fixed_inputs: [1] == [(Eh)/(Eh)] ratio indicating the amount of commodity
                consumed per units of energy created.
            * fixed_outputs: [1] == [(Eh)/(Eh)] ration indicating the amount of
                environmental pollutants produced per units of energy created.

        installation_year: year when the technologies are installed. If not given, it
            defaults to the first year in `prices`. This should be a single value, there
            is currently no provision for computing LCOE over different installation
            years.

    Return:
        The lifetime LCOE in [$/(Eh)] for each technology at each timeslice.
    """
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.utilities import filter_input
    from muse.commodities import is_pollutant

    techs: Dataset = technologies[  # type: ignore
        [
            "technical_life",
            "interest_rate",
            "cap_par",
            "var_par",
            "fix_par",
            "fixed_inputs",
            "fixed_outputs",
        ]
    ]
    if installation_year is None:
        installation_year = int(prices.year.min())
    assert "year" not in filters
    ftechs = filter_input(
        techs,
        year=installation_year,
        **{k: v for k, v in filters.items() if k in techs.dims},
    )
    fprices = filter_input(
        prices,
        year=range(
            installation_year,
            installation_year + ftechs.technical_life.astype(int).max().values,
        ),
        **{k: v for k, v in filters.items() if k in prices.dims},
    ).ffill("year")

    assert {"timeslice", "commodity"}.issubset(fprices.dims)

    interests = ftechs.interest_rate
    life = ftechs.technical_life.astype(int)
    annualized_capital_costs = convert_timeslice(
        ftechs.cap_par * interests / (1 - (1 + interests) ** (-life)),
        fprices.timeslice,
        QuantityType.EXTENSIVE,
    )

    years = fprices.year - installation_year + 1
    rates = (years <= life) / (1 + interests) ** years
    o_and_m_costs = (ftechs.fix_par + ftechs.var_par) * rates.sum("year")

    fuel_costs = (ftechs.fixed_inputs * fprices * rates).sum(("commodity", "year"))

    envs = is_pollutant(ftechs.comm_usage)
    envs = envs.drop_vars(set(envs.coords).difference(envs.dims))
    env_costs = (
        ftechs.fixed_outputs.sel(commodity=envs) * fprices.sel(commodity=envs) * rates
    ).sum(("commodity", "year"))
    return annualized_capital_costs + o_and_m_costs + env_costs + fuel_costs


def maximum_production(technologies: Dataset, capacity: DataArray, **filters):
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
        technologies: Dataset describing the features of the technologies of interests.
            It should contain `fixed_outputs` and `utilization_factor`. It's shape is
            matched to `capacity` using `muse.utilities.broadcast_techs`.
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
        cast(Dataset, technologies[["fixed_outputs", "utilization_factor"]]), capa
    )
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in btechs.dims}
    )
    result = capa * ftechs.fixed_outputs * ftechs.utilization_factor
    result[{"commodity": ~is_enduse(result.comm_usage)}] = 0
    return result


def demand_matched_production(
    demand: DataArray,
    prices: DataArray,
    capacity: DataArray,
    technologies: Dataset,
    **filters,
) -> DataArray:
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

    technologies = broadcast_techs(technologies, capacity)
    cost = annual_levelized_cost_of_energy(prices, technologies, **filters)
    max_production = maximum_production(technologies, capacity, **filters)
    assert ("timeslice" in demand.dims) == ("timeslice" in cost.dims)
    if "timeslice" in demand.dims and "timeslice" not in max_production.dims:
        max_production = convert_timeslice(
            max_production, demand.timeslice, QuantityType.EXTENSIVE
        )
    return demand_matching(demand, cost, max_production)


def capacity_in_use(
    production: DataArray,
    technologies: Dataset,
    max_dim: Optional[Union[Text, Tuple[Text]]] = "commodity",
    **filters,
):
    """Capacity-in-use for each asset, given production.

    Conceptually, this operation is the inverse of `production`.

    Arguments:
        production: Production from each technology of interest.
        technologies: Dataset describing the features of the technologies of interests.
            It should contain `fixed_outputs` and `utilization_factor`. It's shape is
            matched to `capacity` using `muse.utilities.broadcast_techs`.
        max_dim: reduces the given dimensions using `max`. Defaults to "commodity". If
            None, then no reduction is performed.
        filters: keyword arguments are used to filter down the capacity and
            technologies. Filters not relevant to the quantities of interest, i.e.
            filters that are not a dimension of `capacity` or `techologies`, are
            silently ignored.
    Return:
        Capacity-in-use for each technology, whittled down by the filters.
    """
    from numpy import isinf
    from muse.utilities import filter_input, broadcast_techs
    from muse.commodities import is_enduse

    prod = filter_input(
        production, **{k: v for k, v in filters.items() if k in production.dims}
    )
    techs = technologies[["fixed_outputs", "utilization_factor"]]
    assert isinstance(techs, Dataset)
    btechs = broadcast_techs(techs, prod)
    ftechs = filter_input(
        btechs, **{k: v for k, v in filters.items() if k in technologies.dims}
    )
    factor = 1 / (ftechs.fixed_outputs * ftechs.utilization_factor)
    capa_in_use = (
        (prod * factor)
        .where(~isinf(factor), 0)
        .where(is_enduse(technologies.comm_usage), 0)
    )
    if max_dim:
        capa_in_use = capa_in_use.max(max_dim)
    return capa_in_use


def supply_cost(
    production: DataArray, lcoe: DataArray, asset_dim: Text = "asset"
) -> DataArray:
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
    from numpy import isinf

    inv_total = 1 / production.sum(asset_dim)
    result = (production * lcoe).sum(asset_dim) * inv_total.where(~isinf(inv_total), 0)
    return result
