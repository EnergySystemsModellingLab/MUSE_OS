"""Collection of functions for calculating cost metrics (e.g. LCOE, EAC).

In general, these functions take a Dataset of technology parameters, and return a
DataArray of the calculated cost for each technology. Functions may also take additional
data such as commodity prices, capacity of the technologies, and commodity-production
data for the technologies, where appropriate.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
import xarray as xr

from muse.commodities import is_enduse, is_fuel, is_material, is_pollutant
from muse.quantities import production_amplitude
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import filter_input


def cost(func):
    """Decorator to validate the input and output dimensions of the cost functions.

    Rules:
    - Cost functions only support parameter sets for a single year (i.e. no "year"
        dimension in any of the inputs). Costs are then calculated for a single year
        with the parameters provided. In the case of lifetime costs (e.g. lifetime
        LCOE), costs are calculated assuming fixed parameters over the lifetime of the
        technology.
    """

    @wraps(func)
    def wrapper(
        technologies: xr.Dataset,
        prices: xr.DataArray,
        capacity: xr.DataArray,
        production: xr.DataArray,
        consumption: xr.DataArray,
        timeslice_level: str | None = None,
        *args,
        **kwargs,
    ):
        # Check input data
        assert "year" not in technologies.dims
        assert "year" not in prices.dims
        assert "year" not in capacity.dims
        assert "year" not in production.dims
        assert "year" not in consumption.dims

        # Call the function
        result = func(
            technologies,
            prices,
            capacity,
            production,
            consumption,
            timeslice_level,
            *args,
            **kwargs,
        )

        # Check output data
        assert "year" not in result.dims
        return result

    return wrapper


@cost
def capital_costs(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: str | None = None,
    method: str = "lifetime",
):
    if method not in ["lifetime", "annual"]:
        raise ValueError("method must be either 'lifetime' or 'annual'.")

    _capital_costs = distribute_timeslice(
        technologies.cap_par * (capacity**technologies.cap_exp), level=timeslice_level
    )
    if method == "annual":
        # Divide by lifetime to get annualized cost
        _capital_costs /= broadcast_timeslice(
            technologies.technical_life, level=timeslice_level
        )
    return _capital_costs


@cost
def running_costs(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    # Cost related to environmental products
    environmentals = is_pollutant(technologies.comm_usage)
    prices_environmental = filter_input(prices, commodity=environmentals)
    environmental_costs = (production * prices_environmental).sum("commodity")

    # Fuel/energy costs
    fuels = is_fuel(technologies.comm_usage)
    prices_fuel = filter_input(prices, commodity=fuels)
    fuel_costs = (consumption * prices_fuel).sum("commodity")

    # Cost related to material other than fuel/energy and environmentals
    material = is_material(technologies.comm_usage)
    prices_material = filter_input(prices, commodity=material)
    material_costs = (consumption * prices_material).sum("commodity")

    # Fixed costs
    fixed_costs = distribute_timeslice(
        technologies.fix_par * (capacity**technologies.fix_exp), level=timeslice_level
    )

    # Variable costs
    tech_activity = production_amplitude(production, technologies, timeslice_level)
    variable_costs = broadcast_timeslice(
        technologies.var_par, level=timeslice_level
    ) * tech_activity ** broadcast_timeslice(
        technologies.var_exp, level=timeslice_level
    )

    # Total costs
    total_costs = (
        environmental_costs + fuel_costs + material_costs + fixed_costs + variable_costs
    )
    return total_costs


@cost
def net_present_value(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: str | None = None,
    method: str = "lifetime",
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

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")
        method: "lifetime" or "annual"

    Return:
        xr.DataArray with the NPV calculated for the relevant technologies
    """
    if method not in ["lifetime", "annual"]:
        raise ValueError("method must be either 'lifetime' or 'annual'.")

    # Capital costs (lifetime or annual depending on method)
    _capital_costs = capital_costs(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        timeslice_level,
        method,
    )

    # Revenue (annual)
    products = is_enduse(technologies.comm_usage)
    prices_non_env = filter_input(prices, commodity=products)
    revenues = (production * prices_non_env).sum("commodity")

    # Running costs (annual)
    _running_costs = running_costs(
        technologies, prices, capacity, production, consumption, timeslice_level
    )

    # If method is lifetime, have to adjust running costs and revenues
    if method == "lifetime":
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
    timeslice_level: str | None = None,
    method: str = "lifetime",
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
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")
        method: "lifetime" or "annual"

    Return:
        xr.DataArray with the NPC calculated for the relevant technologies
    """
    return -net_present_value(
        technologies, prices, capacity, production, consumption, timeslice_level, method
    )


@cost
def equivalent_annual_cost(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Equivalent annual costs (or annualized cost) of a technology.

    This is the cost that, if it were to occur equally in every year of the
    project lifetime, would give the same net present cost as the actual cash
    flow sequence associated with that component. The cost is computed using the
    `annualized cost`_ expression given by HOMER Energy.

    .. _annualized cost:
        https://www.homerenergy.com/products/pro/docs/3.15/annualized_cost.html

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        xr.DataArray with the EAC calculated for the relevant technologies
    """
    npc = net_present_cost(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        timeslice_level=timeslice_level,
        method="lifetime",
    )
    crf = capital_recovery_factor(technologies)
    return npc * broadcast_timeslice(crf, level=timeslice_level)


@cost
def levelized_cost_of_energy(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: str | None = None,
    method: str = "lifetime",
) -> xr.DataArray:
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    It follows the `simplified LCOE` given by NREL.

    Can calculate either a lifetime or annual LCOE.
    - lifetime: the average cost per unit of production over the entire lifetime of the
        technology.
        Annual running costs and production are calculated for the full lifetime of the
        technology, and adjusted to a present value using the discount rate. Total
        costs (running costs over the lifetime + initial capital costs) are then divided
        by total production to get the average cost per unit of production.
    - annual: the average cost per unit of production in a single year.
        Annual running costs and production are calculated for a single year. Capital
        costs are divided by the lifetime of the technology to get an annualized cost.
        Total costs (annualized capital costs + running costs) are then divided by
        production to get the average cost per unit of production.

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")
        method: "lifetime" or "annual"

    Return:
        xr.DataArray with the LCOE calculated for the relevant technologies
    """
    if method not in ["lifetime", "annual"]:
        raise ValueError("method must be either 'lifetime' or 'annual'.")

    # Capital costs (lifetime or annual depending on method)
    _capital_costs = capital_costs(
        technologies,
        prices,
        capacity,
        production,
        consumption,
        timeslice_level,
        method,
    )

    # Running costs (annual)
    _running_costs = running_costs(
        technologies, prices, capacity, production, consumption, timeslice_level
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

    return data.prices / data.production.where(np.abs(data.production) > 1e-15, np.inf)


def capital_recovery_factor(technologies: xr.Dataset) -> xr.DataArray:
    """Capital recovery factor using interest rate and expected lifetime.

    The `capital recovery factor`_ is computed using the expression given by HOMER
    Energy.

    .. _capital recovery factor:
        https://www.homerenergy.com/products/pro/docs/3.15/capital_recovery_factor.html

    Arguments:
        technologies: All the technologies

    Return:
        xr.DataArray with the CRF calculated for the relevant technologies
    """
    nyears = technologies.technical_life.astype(int)
    crf = technologies.interest_rate / (
        1 - (1 / (1 + technologies.interest_rate) ** nyears)
    )
    return crf


def annual_to_lifetime(costs, technologies, timeslice_level=None):
    life = technologies.technical_life.astype(int)
    iyears = range(life.values.max())
    years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")
    rates = discount_factor(
        years=years,
        interest_rate=technologies.interest_rate,
        mask=years <= life,
    )
    rates = broadcast_timeslice(rates, level=timeslice_level)
    return (costs * rates).sum("year")


def discount_factor(years, interest_rate, mask=1.0):
    """Calculate an array with the rate (aka discount factor) values over the years."""
    return mask / (1 + interest_rate) ** years
