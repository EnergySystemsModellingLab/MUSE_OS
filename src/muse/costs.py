"""Collection of functions for calculating cost metrics (e.g. LCOE, EAC).

In general, these functions take a Dataset of technology parameters, and return a
DataArray of the calculated cost for each technology. Functions may also take additional
data such as commodity prices, capacity of the technologies, and commodity-production
data for the technologies, where appropriate.
"""

from functools import wraps
from typing import Optional

import numpy as np
import xarray as xr

from muse.commodities import is_enduse, is_fuel, is_material, is_pollutant
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import filter_input


def validate(func):
    """Decorator to validate the input and output dimensions of the cost functions."""

    @wraps(func)
    def wrapper(technologies, prices, capacity, production, consumption, **kwargs):
        from muse.utilities import check_dimensions

        # Check input dimensions
        [tech_dim] = set(technologies.dims) - {"commodity", "timeslice", "region"}
        assert tech_dim in ["technology", "asset", "replacement"]
        region_dim = "region" if "region" in technologies.dims else None
        check_dimensions(
            technologies, [tech_dim, region_dim, "commodity"], optional=["timeslice"]
        )
        check_dimensions(prices, [region_dim, "timeslice", "commodity"])
        check_dimensions(capacity, [tech_dim, region_dim, "asset"])
        check_dimensions(
            production, [tech_dim, region_dim, "timeslice", "commodity", "asset"]
        )
        check_dimensions(
            consumption, [tech_dim, region_dim, "timeslice", "commodity", "asset"]
        )

        # Call the function
        result = func(technologies, prices, capacity, production, consumption, **kwargs)

        # Check output dimensions
        check_dimensions(result, [tech_dim, region_dim, "timeslice", "asset"])
        return result

    return wrapper


@validate
def net_present_value(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: Optional[str] = None,
) -> xr.DataArray:
    """Net present value (NPV) of the relevant technologies.

    The net present value of a technology is the present value  of all the revenues that
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

    Note:
        Here, the installation year is always agent.forecast_year,
        since objectives compute the
        NPV for technologies to be installed in the current year. A more general NPV
        computation would have to refer to installation year of the technology.

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        xr.DataArray with the NPV calculated for the relevant technologies
    """
    from muse.quantities import production_amplitude

    # Filtering of the inputs
    techs = technologies[
        [
            "technical_life",
            "interest_rate",
            "cap_par",
            "cap_exp",
            "var_par",
            "var_exp",
            "fix_par",
            "fix_exp",
            "fixed_outputs",
            "utilization_factor",
        ]
    ]

    # Evolution of rates with time
    life = techs.technical_life.astype(int)
    iyears = range(life.values.max())
    years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")
    rates = broadcast_timeslice(
        discount_factor(
            years=years,
            interest_rate=techs.interest_rate,
            mask=years <= life,
        ),
        level=timeslice_level,
    )

    # Filters
    environmentals = is_pollutant(technologies.comm_usage)
    material = is_material(technologies.comm_usage)
    products = is_enduse(technologies.comm_usage)
    fuels = is_fuel(technologies.comm_usage)

    # Revenue
    prices_non_env = filter_input(prices, commodity=products)
    raw_revenues = (production * prices_non_env * rates).sum(("commodity", "year"))

    # Cost of installed capacity
    installed_capacity_costs = distribute_timeslice(
        techs.cap_par * (capacity**techs.cap_exp), level=timeslice_level
    )

    # Cost related to environmental products
    prices_environmental = filter_input(prices, commodity=environmentals)
    environmental_costs = (production * prices_environmental * rates).sum(
        ("commodity", "year")
    )

    # Fuel/energy costs
    prices_fuel = filter_input(prices, commodity=fuels)
    fuel_costs = (consumption * prices_fuel * rates).sum(("commodity", "year"))

    # Cost related to material other than fuel/energy and environmentals
    prices_material = filter_input(prices, commodity=material)
    material_costs = (consumption * prices_material * rates).sum(("commodity", "year"))

    # Fixed costs
    fixed_costs = (
        distribute_timeslice(
            techs.fix_par * (capacity**techs.fix_exp), level=timeslice_level
        )
        * rates
    ).sum("year")

    # Variable costs
    tech_activity = production_amplitude(production, techs, timeslice_level)
    variable_costs = (
        (
            broadcast_timeslice(techs.var_par, level=timeslice_level)
            * tech_activity ** broadcast_timeslice(techs.var_exp, level=timeslice_level)
        )
        * rates
    ).sum("year")

    # Net present value
    result = raw_revenues - (
        installed_capacity_costs
        + fuel_costs
        + environmental_costs
        + material_costs
        + fixed_costs
        + variable_costs
    )
    return result


@validate
def net_present_cost(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
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

    Return:
        xr.DataArray with the NPC calculated for the relevant technologies
    """
    return -net_present_value(technologies, prices, capacity, production, consumption)


@validate
def equivalent_annual_cost(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: Optional[str] = None,
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
    npc = net_present_cost(technologies, prices, capacity, production, consumption)
    crf = capital_recovery_factor(technologies)
    return npc * broadcast_timeslice(crf, level=timeslice_level)


@validate
def lifetime_levelized_cost_of_energy(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: Optional[str] = None,
) -> xr.DataArray:
    """Levelized cost of energy (LCOE) of technologies over their lifetime.

    It follows the `simplified LCOE` given by NREL.

    Arguments:
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        xr.DataArray with the LCOE calculated for the relevant technologies
    """
    from muse.quantities import production_amplitude

    techs = technologies[
        [
            "technical_life",
            "interest_rate",
            "cap_par",
            "cap_exp",
            "var_par",
            "var_exp",
            "fix_par",
            "fix_exp",
            "fixed_outputs",
            "utilization_factor",
        ]
    ]

    # Evolution of rates with time
    life = techs.technical_life.astype(int)
    iyears = range(life.values.max())
    years = xr.DataArray(iyears, coords={"year": iyears}, dims="year")
    rates = broadcast_timeslice(
        discount_factor(
            years=years,
            interest_rate=techs.interest_rate,
            mask=years <= life,
        ),
        level=timeslice_level,
    )

    # Filters
    environmentals = is_pollutant(technologies.comm_usage)
    material = is_material(technologies.comm_usage)
    products = is_enduse(technologies.comm_usage)
    fuels = is_fuel(technologies.comm_usage)

    # Cost of installed capacity
    installed_capacity_costs = distribute_timeslice(
        techs.cap_par * (capacity**techs.cap_exp), level=timeslice_level
    )

    # Cost related to environmental products
    prices_environmental = filter_input(prices, commodity=environmentals)
    environmental_costs = (production * prices_environmental * rates).sum(
        ("commodity", "year")
    )

    # Fuel/energy costs
    prices_fuel = filter_input(prices, commodity=fuels)
    fuel_costs = (consumption * prices_fuel * rates).sum(("commodity", "year"))

    # Cost related to material other than fuel/energy and environmentals
    prices_material = filter_input(prices, commodity=material)
    material_costs = (consumption * prices_material * rates).sum(("commodity", "year"))

    # Fixed costs
    fixed_costs = (
        distribute_timeslice(
            techs.fix_par * (capacity**techs.fix_exp), level=timeslice_level
        )
        * rates
    ).sum("year")

    # Variable costs
    tech_activity = production_amplitude(production, techs, timeslice_level)
    variable_costs = (
        (
            broadcast_timeslice(techs.var_par, level=timeslice_level)
            * tech_activity ** broadcast_timeslice(techs.var_exp, level=timeslice_level)
        )
        * rates
    ).sum("year")

    # Production
    prod = (
        production.where(production > 0.0, 1e-6)
        .sel(commodity=products)
        .sum("commodity")
    )
    total_prod = (prod * rates).sum("year")

    # LCOE
    result = (
        installed_capacity_costs
        + fuel_costs
        + environmental_costs
        + material_costs
        + fixed_costs
        + variable_costs
    ) / total_prod

    return result


@validate
def annual_levelized_cost_of_energy(
    technologies: xr.Dataset,
    prices: xr.DataArray,
    capacity: xr.DataArray,
    production: xr.DataArray,
    consumption: xr.DataArray,
    timeslice_level: Optional[str] = None,
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
        technologies: xr.Dataset of technology parameters
        prices: xr.DataArray with commodity prices
        capacity: xr.DataArray with the capacity of the relevant technologies
        production: xr.DataArray with commodity production by the relevant technologies
        consumption: xr.DataArray with commodity consumption by the relevant
            technologies
        timeslice_level: the desired timeslice level of the result (e.g. "hour", "day")

    Return:
        The lifetime LCOE in [$/(Eh)] for each technology at each timeslice.

    .. _simplified LCOE: https://www.nrel.gov/analysis/tech-lcoe-documentation.html
    """
    from muse.quantities import production_amplitude

    techs = technologies[
        [
            "technical_life",
            "interest_rate",
            "cap_par",
            "cap_exp",
            "var_par",
            "var_exp",
            "fix_par",
            "fix_exp",
            "fixed_outputs",
            "fixed_inputs",
            "flexible_inputs",
            "utilization_factor",
        ]
    ]

    # Filters
    environmentals = is_pollutant(technologies.comm_usage)
    material = is_material(technologies.comm_usage)
    products = is_enduse(technologies.comm_usage)
    fuels = is_fuel(technologies.comm_usage)

    # Cost of installed capacity (annualized)
    installed_capacity_costs = distribute_timeslice(
        techs.cap_par * (capacity**techs.cap_exp), level=timeslice_level
    ) / broadcast_timeslice(techs.technical_life, level=timeslice_level)

    # Cost related to environmental products
    prices_environmental = filter_input(prices, commodity=environmentals)
    environmental_costs = (production * prices_environmental).sum("commodity")

    # Fuel/energy costs
    prices_fuel = filter_input(prices, commodity=fuels)
    fuel_costs = (consumption * prices_fuel).sum("commodity")

    # Cost related to material other than fuel/energy and environmentals
    prices_material = filter_input(prices, commodity=material)
    material_costs = (consumption * prices_material).sum("commodity")

    # Fixed costs
    fixed_costs = distribute_timeslice(
        techs.fix_par * (capacity**techs.fix_exp), level=timeslice_level
    )

    # Variable costs
    tech_activity = production_amplitude(production, techs, timeslice_level)
    variable_costs = broadcast_timeslice(
        techs.var_par, level=timeslice_level
    ) * tech_activity ** broadcast_timeslice(techs.var_exp, level=timeslice_level)

    # Production
    prod = (
        production.where(production > 0.0, 1e-6)
        .sel(commodity=products)
        .sum("commodity")
    )

    # LCOE
    result = (
        installed_capacity_costs
        + fuel_costs
        + environmental_costs
        + material_costs
        + fixed_costs
        + variable_costs
    ) / prod

    return result


def supply_cost(
    production: xr.DataArray, lcoe: xr.DataArray, asset_dim: Optional[str] = "asset"
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
            `muse.costs.annual_levelized_cost_of_energy` or
            `muse.costs.lifetime_levelized_cost_of_energy`.
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


def discount_factor(years, interest_rate, mask=1.0):
    """Calculate an array with the rate (aka discount factor) values over the years."""
    return mask / (1 + interest_rate) ** years
