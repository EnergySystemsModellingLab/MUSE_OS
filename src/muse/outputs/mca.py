"""Output quantities.

Functions that compute MCA quantities for post-simulation analysis should all follow the
same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        sectors: List[AbstractSector],
        market: xr.Dataset,
        year: int,
        **kwargs
    ) -> Union[pd.DataFrame, xr.DataArray]:
        pass

The function should never modify it's arguments. It can return either a pandas dataframe
or an xarray xr.DataArray.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.outputs.sector import market_quantity
from muse.registration import registrator
from muse.sectors import AbstractSector
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import multiindex_to_coords

OUTPUT_QUANTITY_SIGNATURE = Callable[
    [xr.Dataset, list[AbstractSector], KwArg(Any)], Union[xr.DataArray, pd.DataFrame]
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: MutableMapping[str, OUTPUT_QUANTITY_SIGNATURE] = {}
"""Quantity for post-simulation analysis."""

OUTPUTS_PARAMETERS = Union[str, Mapping]
"""Acceptable Datastructures for outputs parameters"""


@registrator(registry=OUTPUT_QUANTITIES)
def register_output_quantity(
    function: OUTPUT_QUANTITY_SIGNATURE | None = None,
) -> Callable:
    """Registers a function to compute an output quantity."""
    from functools import wraps

    assert function is not None

    @wraps(function)
    def decorated(*args, **kwargs):
        result: xr.DataArray = function(*args, **kwargs)
        result.name = function.__name__
        return result

    return decorated


def factory(
    *parameters: OUTPUTS_PARAMETERS,
) -> Callable[[xr.Dataset, list[AbstractSector]], list[Path]]:
    """Creates outputs functions for post-mortem analysis.

    Each parameter is a dictionary containing the following:

    - quantity (mandatory): name of the quantity to output. Mandatory.
    - sink (optional): name of the storage procedure, e.g. the file format
      or database format. When it cannot be guessed from `filename`, it defaults to
      "csv".
    - filename (optional): path to a directory or a file where to store the quantity. In
      the latter case, if sink is not given, it will be determined from the file
      extension. The filename can incorporate markers. By default, it is
      "{default_output_dir}/{sector}{year}{quantity}{suffix}".
    - any other parameter relevant to the sink, e.g. `pandas.to_csv` keyword
      arguments.

    For simplicity, it is also possible to give lone strings as input.
    They default to `{'quantity': string}` (and the sink will default to
    "csv").
    """
    from muse.outputs.sector import _factory

    return _factory(OUTPUT_QUANTITIES, *parameters, sector_name="MCA")


@register_output_quantity
def consumption(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current consumption."""
    result = market_quantity(market.consumption, **kwargs).to_dataframe().reset_index()
    return result[result.consumption != 0]


@register_output_quantity
def supply(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current supply."""
    result = market_quantity(market.supply, **kwargs).to_dataframe().reset_index()
    return result[result.supply != 0]


@register_output_quantity
def prices(
    market: xr.Dataset,
    sectors: list[AbstractSector],
    year: int,
    **kwargs,
) -> pd.DataFrame:
    """Current MCA market prices."""
    return market_quantity(market.prices, **kwargs).to_dataframe().reset_index()


@register_output_quantity
def capacity(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current capacity across all sectors."""
    return _aggregate_sectors(sectors, op=sector_capacity)


def sector_capacity(sector: AbstractSector) -> pd.DataFrame:
    """Sector capacity with agent annotations."""
    capa_sector: list[xr.DataArray] = []
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for agent in agents:
        capa_agent = agent.assets.capacity
        capa_agent["agent"] = agent.name
        capa_agent["type"] = agent.category
        capa_agent["sector"] = getattr(sector, "name", "unnamed")

        if len(capa_agent) > 0 and len(capa_agent.technology.values) > 0:
            if "dst_region" not in capa_agent.coords:
                capa_agent["dst_region"] = agent.region
            a = capa_agent.to_dataframe()
            b = (
                a.groupby(
                    [
                        "technology",
                        "dst_region",
                        "region",
                        "agent",
                        "sector",
                        "type",
                        "year",
                        "installed",
                    ]
                )
                .sum()  # ("asset")
                .fillna(0)
            )
            c = b.reset_index()
            capa_sector.append(c)
    if len(capa_sector) == 0:
        return pd.DataFrame()

    capacity = pd.concat([u for u in capa_sector])
    capacity = capacity[capacity.capacity != 0]
    return capacity


def _aggregate_sectors(
    sectors: list[AbstractSector], *args, op: Callable
) -> pd.DataFrame:
    """Aggregate outputs from all sectors."""
    alldata = [op(sector, *args) for sector in sectors]

    if len(alldata) == 0:
        return pd.DataFrame()
    return pd.concat(alldata, sort=True)


@register_output_quantity(name=["fuel_costs"])
def metric_fuel_costs(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current fuel costs across all sectors."""
    return _aggregate_sectors(sectors, market, year, op=sector_fuel_costs)


def sector_fuel_costs(
    sector: AbstractSector, market: xr.Dataset, year: int, **kwargs
) -> pd.DataFrame:
    """Sector fuel costs with agent annotations."""
    from muse.commodities import is_fuel
    from muse.production import supply
    from muse.quantities import consumption

    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market.copy(deep=True)
    if len(technologies) > 0:
        for a in agents:
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=year
            )
            commodity = is_fuel(technologies.comm_usage)

            capacity = a.filter_input(
                a.assets.capacity,
                year=year,
            ).fillna(0.0)

            production = supply(
                agent_market,
                capacity,
                technologies,
            )

            prices = a.filter_input(market.prices, year=year)
            fcons = consumption(
                technologies=technologies, production=production, prices=prices
            )

            data_agent = (fcons * prices).sel(commodity=commodity)
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = year
            data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
                "fuel_consumption_costs"
            )
            if not data_agent.empty:
                data_sector.append(data_agent)
    if len(data_sector) > 0:
        output = pd.concat(data_sector, sort=True).reset_index()
    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["capital_costs"])
def metric_capital_costs(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current capital costs across all sectors."""
    return _aggregate_sectors(sectors, market, year, op=sector_capital_costs)


def sector_capital_costs(
    sector: AbstractSector, market: xr.Dataset, year: int, **kwargs
) -> pd.DataFrame:
    """Sector capital costs with agent annotations."""
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    if len(technologies) > 0:
        for a in agents:
            capacity = a.filter_input(a.assets.capacity, year=year).fillna(0.0)
            data = a.filter_input(
                technologies[["cap_par", "cap_exp"]],
                year=year,
                technology=capacity.technology,
            )
            data_agent = distribute_timeslice(data.cap_par * (capacity**data.cap_exp))
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = year
            data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
                "capital_costs"
            )
            if not data_agent.empty:
                data_sector.append(data_agent)

    if len(data_sector) > 0:
        output = pd.concat(data_sector, sort=True).reset_index()
    else:
        output = pd.DataFrame()
    return output


@register_output_quantity(name=["emission_costs"])
def metric_emission_costs(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    return _aggregate_sectors(sectors, market, year, op=sector_emission_costs)


def sector_emission_costs(
    sector: AbstractSector, market: xr.Dataset, year: int, **kwargs
) -> pd.DataFrame:
    """Sector emission costs with agent annotations."""
    from muse.commodities import is_enduse, is_pollutant
    from muse.production import supply

    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))

    agent_market = market.copy(deep=True)
    if len(technologies) > 0:
        for a in agents:
            agent_market["consumption"] = (market.consumption * a.quantity).sel(
                year=year
            )

            capacity = a.filter_input(a.assets.capacity, year=year).fillna(0.0)
            allemissions = a.filter_input(
                technologies.fixed_outputs,
                commodity=is_pollutant(technologies.comm_usage),
                technology=capacity.technology,
                year=year,
            )
            envs = is_pollutant(technologies.comm_usage)
            enduses = is_enduse(technologies.comm_usage)
            i = (np.where(envs))[0][0]
            red_envs = envs[i].commodity.values
            prices = a.filter_input(market.prices, year=year, commodity=red_envs)
            production = supply(
                agent_market,
                capacity,
                technologies,
            )

            total = production.sel(commodity=enduses).sum("commodity")
            data_agent = total * (allemissions * prices).sum("commodity")
            data_agent["agent"] = a.name
            data_agent["category"] = a.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = year
            data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
                "emission_costs"
            )
            if not data_agent.empty:
                data_sector.append(data_agent)

    if len(data_sector) > 0:
        output = pd.concat(data_sector, sort=True).reset_index()
    else:
        output = pd.DataFrame()

    return output


@register_output_quantity(name=["LCOE"])
def metric_lcoe(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current lifetime levelised cost across all sectors."""
    return _aggregate_sectors(sectors, market, year, op=sector_lcoe)


def sector_lcoe(
    sector: AbstractSector, market: xr.Dataset, year: int, **kwargs
) -> pd.DataFrame:
    """Levelized cost of energy () of technologies over their lifetime."""
    from muse.commodities import is_enduse
    from muse.costs import levelized_cost_of_energy as LCOE
    from muse.quantities import capacity_to_service_demand, consumption

    market = market.copy(deep=True)

    # Filtering of the inputs
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    retro = [a for a in agents if a.category == "retrofit"]
    new = [a for a in agents if a.category == "newcapa"]
    agents = retro if len(retro) > 0 else new
    if len(technologies) > 0:
        for agent in agents:
            agent_market = market.sel(year=agent.year)
            agent_market["consumption"] = agent_market.consumption * agent.quantity

            # Filter commodities based on end-use status
            enduse_mask = is_enduse(technologies.comm_usage)
            commodities = agent_market.commodity.values
            included_commodities = commodities[
                np.isin(commodities, enduse_mask.commodity[enduse_mask])
            ]
            excluded_commodities = commodities[
                ~np.isin(commodities, enduse_mask.commodity[enduse_mask])
            ]

            agent_market.loc[dict(commodity=excluded_commodities)] = 0
            agent_market["prices"] = agent.filter_input(
                market["prices"], year=agent.year
            )

            techs = agent.filter_input(
                technologies,
                year=agent.year,
            )
            prices = agent_market["prices"].sel(commodity=techs.commodity)
            demand = agent_market.consumption.sel(commodity=included_commodities)
            capacity = agent.filter_input(capacity_to_service_demand(demand, techs))
            production = (
                broadcast_timeslice(capacity)
                * distribute_timeslice(techs.fixed_outputs)
                * broadcast_timeslice(techs.utilization_factor)
            )
            consump = consumption(
                technologies=techs, prices=prices, production=production
            )

            result = LCOE(
                technologies=techs,
                prices=prices,
                capacity=capacity,
                production=production,
                consumption=consump,
                method="lifetime",
            )

            data_agent = result
            data_agent["agent"] = agent.name
            data_agent["category"] = agent.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = agent.year
            data_agent = data_agent.fillna(0)
            data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
                "LCOE"
            )
            if not data_agent.empty:
                data_sector.append(data_agent)

    if len(data_sector) > 0:
        output = pd.concat(data_sector, sort=True).reset_index()
    else:
        output = pd.DataFrame()
    return output


@register_output_quantity(name=["EAC"])
def metric_eac(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current emission costs across all sectors."""
    return _aggregate_sectors(sectors, market, year, op=sector_eac)


def sector_eac(
    sector: AbstractSector, market: xr.Dataset, year: int, **kwargs
) -> pd.DataFrame:
    """Net Present Value of technologies over their lifetime."""
    from muse.commodities import is_enduse
    from muse.costs import equivalent_annual_cost as EAC
    from muse.quantities import capacity_to_service_demand, consumption

    market = market.copy(deep=True)

    # Filtering of the inputs
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies", [])
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    retro = [a for a in agents if a.category == "retrofit"]
    new = [a for a in agents if a.category == "newcapa"]
    agents = retro if len(retro) > 0 else new
    if len(technologies) > 0:
        for agent in agents:
            agent_market = market.sel(year=agent.year)
            agent_market["consumption"] = agent_market.consumption * agent.quantity

            # Filter commodities based on end-use status
            enduse_mask = is_enduse(technologies.comm_usage)
            commodities = agent_market.commodity.values
            included_commodities = commodities[
                np.isin(commodities, enduse_mask.commodity[enduse_mask])
            ]
            excluded_commodities = commodities[
                ~np.isin(commodities, enduse_mask.commodity[enduse_mask])
            ]

            agent_market.loc[dict(commodity=excluded_commodities)] = 0
            agent_market["prices"] = agent.filter_input(
                market["prices"], year=agent.year
            )

            techs = agent.filter_input(
                technologies,
                year=agent.year,
            )
            prices = agent_market["prices"].sel(commodity=techs.commodity)
            demand = agent_market.consumption.sel(commodity=included_commodities)
            capacity = agent.filter_input(capacity_to_service_demand(demand, techs))
            production = (
                broadcast_timeslice(capacity)
                * distribute_timeslice(techs.fixed_outputs)
                * broadcast_timeslice(techs.utilization_factor)
            )
            consump = consumption(
                technologies=techs, prices=prices, production=production
            )

            result = EAC(
                technologies=techs,
                prices=prices,
                capacity=capacity,
                production=production,
                consumption=consump,
            )

            data_agent = result
            data_agent["agent"] = agent.name
            data_agent["category"] = agent.category
            data_agent["sector"] = getattr(sector, "name", "unnamed")
            data_agent["year"] = agent.year
            data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
                "capital_costs"
            )
            if not data_agent.empty:
                data_sector.append(data_agent)
    if len(data_sector) > 0:
        output = pd.concat(data_sector, sort=True).reset_index()
    else:
        output = pd.DataFrame()
    return output
