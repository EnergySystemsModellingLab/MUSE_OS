"""Collection of functions for calculating cost metrics (e.g. LCOE, EAC).

All costs functions take a subset of the following arguments:
- technologies: xr.Dataset of technology parameters
- prices: xr.DataArray with commodity prices
- capacity: xr.DataArray with the capacity of the technologies
- production: xr.DataArray with commodity production by the technologies
- consumption: xr.DataArray with commodity consumption by the technologies
- method: "lifetime" or "annual"

Data should only be provided for a single year (i.e. no "year" dimension in any of the
inputs). Prices, production and consumption data should be split across timeslices
(i.e. have a "timeslice" dimension). Technology parameters may also be specified at
the timeslice level, but capacity should not be.

The `technologies` input will usually contain data for multiple technologies and have
a "technology" dimension (sometimes called "asset" or "replacement"). In this case,
it's important that the `capacity`, `production`, and `consumption` inputs have a
similar dimension to ensure that costs are calculated for all technologies and to
prevent unwanted broadcasting.

Additional dimensions (such as "region") may be present in the inputs, but it's up to
the parent functions to ensure that these are consistent between inputs to prevent
unwanted broadcasting.

The dimensions of the output will be the sum of all dimensions from the input data,
minus "commodity", plus "timeslice" (if not already present).

Some functions have a `method` argument, which can be "annual" or "lifetime":

Costs can either be annual or lifetime:
- annual: calculates the cost in a single year
- lifetime: calculates the total cost over the lifetime of the
    technology, using the `technical_life` attribute from the `technologies` dataset.
    - In this case, technology parameters, production, consumption, capacity and prices
        are assumed to be constant over the lifetime of the technology. The cost in each
        year is discounted according to the `interest_rate` attribute from the
        `technologies` dataset, and summed across years.
    - Capital costs are different, as these are a one time cost for the lifetime of the
        technology. This can be annualized by dividing by the `technical_life`.
Some functions can calculate both lifetime and annual costs, with a `method` argument
to specify. Others can only calculate one or the other (see individual function
docstrings for more details).

"""

from __future__ import annotations

from functools import wraps

import numpy as np
import xarray as xr

from muse.commodities import is_enduse, is_fuel, is_material, is_pollutant
from muse.quantities import production_amplitude
from muse.timeslices import broadcast_timeslice, distribute_timeslice, get_level


def cost(func):
    """Decorator to validate the output dimensions of the cost functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Check dimensions of the result
        assert "year" not in result.dims
        assert "commodity" not in result.dims

        # Check that there are no infs or nans in the result
        assert not result.isnull().any()
        assert not np.isinf(result).any()
        return result

    return wrapper


@cost
def capital_costs(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    method: str = "lifetime",
):
    """Calculate capital costs for the relevant technologies.

    This is the cost of installing each technology to the level specified by the
    `capacity` input.

    Method can be "lifetime" or "annual":
    - lifetime: returns the full capital costs
    - annual: total capital costs are multiplied by the capital recovery factor to get
        annualized costs
    """
    if method not in ["lifetime", "annual"]:
        raise ValueError("method must be either 'lifetime' or 'annual'.")

    _capital_costs = technologies.cap_par * (capacity**technologies.cap_exp)
    if method == "annual":
        crf = capital_recovery_factor(technologies)
        _capital_costs = _capital_costs * crf
    assert "timeslice" not in _capital_costs.dims
    return _capital_costs


@cost
def environmental_costs(
    technologies: xr.Dataset, prices: xr.DataArray, production: xr.DataArray
) -> xr.DataArray:
    """Calculate timeslice-level environmental costs for the relevant technologies.

    This is the total production of pollutants (commodities flagged by `is_pollutant`)
    multiplied by their prices.
    """
    environmentals = is_pollutant(technologies.comm_usage)
    prices_environmental = prices.sel(commodity=environmentals)
    result = (production * prices_environmental).sum("commodity")
    assert "timeslice" in result.dims
    return result


@cost
def fuel_costs(
    technologies: xr.Dataset, prices: xr.DataArray, consumption: xr.DataArray
) -> xr.DataArray:
    """Calculate timeslice-level fuel costs for the relevant technologies.

    This is the total consumption of fuels (commodities flagged by `is_fuel`)
    multiplied by their prices.
    """
    fuels = is_fuel(technologies.comm_usage)
    prices_fuel = prices.sel(commodity=fuels)
    result = (consumption * prices_fuel).sum("commodity")
    assert "timeslice" in result.dims
    return result


@cost
def material_costs(
    technologies: xr.Dataset, prices: xr.DataArray, consumption: xr.DataArray
) -> xr.DataArray:
    """Calculate timeslice-level material costs for the relevant technologies.

    This is the total consumption of materials (commodities flagged by `is_material`)
    multiplied by their prices.
    """
    material = is_material(technologies.comm_usage)
    prices_material = prices.sel(commodity=material)
    result = (consumption * prices_material).sum("commodity")
    assert "timeslice" in result.dims
    return result


@cost
def fixed_costs(technologies: xr.Dataset, capacity: xr.DataArray) -> xr.DataArray:
    """Calculate annual fixed costs for the relevant technologies.

    This is the fixed running cost over the course of a year corresponding to the
    `fix_par` and `fix_exp` technology parameters.
    """
    result = technologies.fix_par * (capacity**technologies.fix_exp)
    assert "timeslice" not in result.dims
    return result


@cost
def variable_costs(
    technologies: xr.Dataset,
    production: xr.DataArray,
) -> xr.DataArray:
    """Calculate annual variable costs for the relevant technologies.

    This is the cost associated with the `var_par` and `var_exp` technology
    parameters.

    The `production_amplitude` function is first used to calculate technology activity
    based on `production`. This is then used to scale the variable costs.
    """
    tech_activity = production_amplitude(production, technologies).sum("timeslice")
    result = technologies.var_par * tech_activity**technologies.var_exp
    assert "timeslice" not in result.dims
    return result


@cost
def running_costs(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    aggregate_timeslices: bool = False,
) -> xr.DataArray:
    """Total annual running costs (excluding capital costs).

    This is the sum of environmental, fuel, material, fixed and variable costs.

    .. seealso::
        :py:func:`environmental_costs`
        :py:func:`fuel_costs`
        :py:func:`material_costs`
        :py:func:`fixed_costs`
        :py:func:`variable_costs`
    """
    # Costs associated with commodity inputs and outputs (timeslice-level)
    _environmental_costs = environmental_costs(technologies, prices, production)
    _fuel_costs = fuel_costs(technologies, prices, consumption)
    _material_costs = material_costs(technologies, prices, consumption)

    # Aggregate over timeslices (if required)
    if aggregate_timeslices:
        _environmental_costs = _environmental_costs.sum("timeslice")
        _fuel_costs = _fuel_costs.sum("timeslice")
        _material_costs = _material_costs.sum("timeslice")

    # Costs associated with capacity and production level (annual)
    _fixed_costs = fixed_costs(technologies, capacity)
    _variable_costs = variable_costs(technologies, production)

    # Split fixed/variable across timeslices in proportion to production (if required)
    if not aggregate_timeslices:
        timeslice_level = get_level(production)
        tech_activity = production_amplitude(production, technologies)
        _fixed_costs = distribute_timeslice(
            _fixed_costs, ts=tech_activity, level=timeslice_level
        )
        _variable_costs = distribute_timeslice(
            _variable_costs, ts=tech_activity, level=timeslice_level
        )

    # Total running costs
    result = (
        _environmental_costs
        + _fuel_costs
        + _material_costs
        + _fixed_costs
        + _variable_costs
    )
    return result


@cost
def net_present_value(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    aggregate_timeslices: bool = False,
) -> xr.DataArray:
    """Net present value (NPV) of the relevant technologies.

    The net present value of a technology is the present value of all the revenues that
    a technology earns over its lifetime minus all the costs of installing and operating
    it. Follows the definition of the `net present cost`_ given by HOMER Energy.
    .. _net present cost:
    ..      https://www.homerenergy.com/products/pro/docs/3.15/net_present_cost.html

    - energy commodities INPUTS are related to fuel costs
    - environmental commodities OUTPUTS are related to environmental costs
    - material and service commodities INPUTS are related to consumable costs
    - fixed and variable costs are given as technodata inputs and depend on the
      installed capacity and production (non-environmental), respectively
    - capacity costs are given as technodata inputs and depend on the installed capacity

    .. seealso::
        :py:func:`capital_costs`
        :py:func:`running_costs`

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        aggregate_timeslices: If True, the LCOE is aggregated over timeslices (result
            will not have a "timeslice" dimension)

    Return:
        xr.DataArray with the NPV calculated for the relevant technologies
    """
    # Capital costs (lifetime)
    _capital_costs = capital_costs(technologies, capacity, method="lifetime")

    # Split capital costs across timeslices in proportion to production (if required)
    if not aggregate_timeslices:
        tech_activity = production_amplitude(production, technologies)
        _capital_costs = distribute_timeslice(
            _capital_costs, ts=tech_activity, level=get_level(production)
        )

    # Revenue (annual)
    products = is_enduse(technologies.comm_usage)
    prices_non_env = prices.sel(commodity=products)
    revenues = (production * prices_non_env).sum("commodity")
    if aggregate_timeslices:
        revenues = revenues.sum("timeslice")

    # Running costs (annual)
    _running_costs = running_costs(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        aggregate_timeslices,
    )

    # Calculate running costs and revenues over lifetime
    _running_costs = annual_to_lifetime(_running_costs, technologies)
    revenues = annual_to_lifetime(revenues, technologies)

    # Net present value
    result = revenues - (_capital_costs + _running_costs)
    return result


@cost
def net_present_cost(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    aggregate_timeslices: bool = False,
) -> xr.DataArray:
    """Net present cost (NPC) of the relevant technologies.

    The net present cost of a Component is the present value of all the costs of
    installing and operating the Component over the project lifetime, minus the present
    value of all the revenues that it earns over the project lifetime.

    .. seealso::
        :py:func:`net_present_value`.

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        aggregate_timeslices: If True, the LCOE is aggregated over timeslices (result
            will not have a "timeslice" dimension)

    Return:
        xr.DataArray with the NPC calculated for the relevant technologies
    """
    result = -net_present_value(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        aggregate_timeslices,
    )
    return result


@cost
def equivalent_annual_cost(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    aggregate_timeslices: bool = False,
) -> xr.DataArray:
    """Equivalent annual costs (or annualized cost) of a technology.

    This is the cost that, if it were to occur equally in every year of the
    project lifetime, would give the same net present cost as the actual cash
    flow sequence associated with that component. The cost is computed using the
    `annualized cost`_ expression given by HOMER Energy.

    .. _annualized cost:
        https://www.homerenergy.com/products/pro/docs/3.15/annualized_cost.html

    .. seealso::
        :py:func:`net_present_cost`

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        aggregate_timeslices: If True, the LCOE is aggregated over timeslices (result
            will not have a "timeslice" dimension)

    Return:
        xr.DataArray with the EAC calculated for the relevant technologies
    """
    npc = net_present_cost(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        aggregate_timeslices,
    )
    crf = capital_recovery_factor(technologies)
    if not aggregate_timeslices:
        crf = broadcast_timeslice(crf, level=get_level(production))
    result = npc * crf
    return result


@cost
def levelized_cost_of_energy(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    method: str = "lifetime",
    aggregate_timeslices: bool = False,
) -> xr.DataArray:
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    It follows the `simplified LCOE` given by NREL.

    .. seealso::
        :py:func:`capital_costs`
        :py:func:`running_costs`

    Can calculate either a lifetime or annual LCOE.
    - lifetime: the average cost per unit of production over the entire lifetime of the
        technology.
        Annual running costs and production are calculated for the full lifetime of the
        technology, and adjusted to a present value using the discount rate. Total
        costs (running costs over the lifetime + initial capital costs) are then divided
        by total production to get the average cost per unit of production.
    - annual: the average cost per unit of production in a single year.
        Annual running costs and production are calculated for a single year. Capital
        costs are multiplied by the capital recovery factor to get an annualized cost.
        Total costs (annualized capital costs + running costs) are then divided by
        production to get the average cost per unit of production.

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        method: "lifetime" or "annual"
        aggregate_timeslices: If True, the LCOE is aggregated over timeslices (result
            will not have a "timeslice" dimension)

    Return:
        xr.DataArray with the LCOE calculated for the relevant technologies
    """
    if method not in ["lifetime", "annual"]:
        raise ValueError("method must be either 'lifetime' or 'annual'.")

    # Capital costs (lifetime or annual depending on method)
    _capital_costs = capital_costs(technologies, capacity, method)

    # Split capital costs across timeslices in proportion to production (if required)
    if not aggregate_timeslices:
        tech_activity = production_amplitude(production, technologies)
        _capital_costs = distribute_timeslice(
            _capital_costs, ts=tech_activity, level=get_level(production)
        )

    # Running costs (annual)
    _running_costs = running_costs(
        technologies, prices, capacity, production, consumption, aggregate_timeslices
    )

    # Production (annual)
    products = is_enduse(technologies.comm_usage)
    prod = (
        production.where(production > 0.0, 1e-6)
        .sel(commodity=products)
        .sum(
            "commodity"
        )  # TODO: is this the correct way to deal with multiple products?
    )
    if aggregate_timeslices:
        prod = prod.sum("timeslice")

    # If method is lifetime, have to adjust running costs and production
    if method == "lifetime":
        _running_costs = annual_to_lifetime(_running_costs, technologies)
        prod = annual_to_lifetime(prod, technologies)

    # LCOE
    result = (_capital_costs + _running_costs) / prod
    return result


def supply_cost(
    production: xr.DataArray, lcoe: xr.DataArray, asset_dim: str | None = "asset"
) -> xr.DataArray:
    """Supply cost given production and the levelized cost of energy.

    In practice, the supply cost is the weighted average LCOE over assets (`asset_dim`),
    where the weights are the production.

    Very low costs are set to zero.

    Arguments:
        production: Amount of goods produced. In practice, production can be obtained
            from the capacity for each asset via the method
            `muse.quantities.production`.
        lcoe: Levelized cost of energy for each good produced. In practice, it can be
            obtained from market prices via
            `muse.costs.levelized_cost_of_energy`.
        asset_dim: Name of the dimension(s) holding assets, processes or technologies.
    """
    data = xr.Dataset(dict(production=production, prices=production * lcoe))
    if asset_dim is not None:
        if "region" not in data.coords or len(data.region.dims) == 0:
            data = data.sum(asset_dim)
        else:
            data = data.groupby("region").sum(asset_dim)

    costs = data.prices / data.production.where(np.abs(data.production) > 1e-15, np.inf)
    return costs.where(costs > 1e-4, 0)


def capital_recovery_factor(technologies: xr.Dataset) -> xr.DataArray:
    """Capital recovery factor using interest rate and expected lifetime.

    The `capital recovery factor`_ is computed using the expression given by HOMER
    Energy.

    .. _capital recovery factor:
        https://www.homerenergy.com/products/pro/docs/3.15/capital_recovery_factor.html

    If the interest rate is zero, this simplifies to 1 / nyears

    Arguments:
        technologies: All the technologies

    Return:
        xr.DataArray with the CRF calculated for the relevant technologies
    """
    nyears = technologies.technical_life.astype(int)
    interest_rate = technologies.interest_rate

    crf = xr.where(
        interest_rate == 0,
        1 / nyears,
        interest_rate / (1 - (1 / (1 + interest_rate) ** nyears)),
    )

    assert "year" not in crf.dims
    return crf


def annual_to_lifetime(costs: xr.DataArray, technologies: xr.Dataset):
    """Convert annual costs to lifetime costs.

    Costs are provided for a single year. These same costs are assumed to apply for the
    full lifetime of the technologies, subject to a discount factor. The costs are then
    summed over the lifetime of the technologies.

    Args:
        costs: xr.DataArray of costs for a single year.
        technologies: xr.Dataset of technology parameters
    """
    assert "year" not in costs.dims
    assert "year" not in technologies.dims
    life = technologies.technical_life.astype(int)
    iyears = range(life.values.max())
    years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")
    rates = discount_factor(
        years=years,
        interest_rate=technologies.interest_rate,
        mask=years <= life,
    )
    if "timeslice" in costs.dims:
        rates = broadcast_timeslice(rates, level=get_level(costs))
    return (costs * rates).sum("year")


def discount_factor(
    years: xr.DataArray, interest_rate: xr.DataArray, mask: xr.DataArray | None = None
):
    """Calculate an array with of discount factor values over the years.

    Args:
        years: xr.DataArray with the years counting from the present year
            (i.e. current year = 0)
        interest_rate: xr.DataArray with the interest rate for different technologies
        mask: Optional mask to apply to the result (e.g. cutting to zero after the
            technology lifetime)
    """
    assert set(years.dims) == {"year"}
    assert "year" not in interest_rate.dims

    # Calculate discount factor over the years
    df = 1 / (1 + interest_rate) ** years

    # Apply mask
    if mask is not None:
        assert set(mask.dims) == set(interest_rate.dims) | {"year"}
        df = df * mask
    return df
