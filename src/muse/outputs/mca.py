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
from typing import (
    Any,
    Callable,
    Union,
    cast,
)

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.outputs.sector import market_quantity
from muse.registration import registrator
from muse.sectors import AbstractSector
from muse.sectors.preset_sector import PresetSector
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
        result = function(*args, **kwargs)
        if isinstance(result, (pd.DataFrame, xr.DataArray)):
            result.name = function.__name__
        return result

    return decorated


def round_values(function: Callable) -> OUTPUT_QUANTITY_SIGNATURE:
    """Rounds the outputs to given number of decimals and drops columns with zeros."""
    from functools import wraps

    @wraps(function)
    def rounded(
        market: xr.Dataset,
        sectors: list[AbstractSector],
        year: int,
        rounding: int = 4,
        **kwargs,
    ) -> xr.DataArray:
        result = function(market=market, sectors=sectors, year=year, **kwargs)

        if hasattr(result, "to_dataframe"):
            result = result.to_dataframe()
        result = result.round(rounding)
        name = getattr(result, "name", function.__name__)
        if len(result) > 0:
            return result[result[name] != 0]

    return rounded


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

    def reformat_finite_resources(params):
        from muse.readers.toml import MissingSettings

        name = params["quantity"]
        if not isinstance(name, str):
            name = name["name"]
        if name.lower() not in {"finite_resources", "finiteresources"}:
            return params

        quantity = params["quantity"]
        if isinstance(quantity, str):
            quantity = dict(name=quantity)
        else:
            quantity = dict(**quantity)

        if "limits_path" in params:
            quantity["limits_path"] = params.pop("limits_path")
        if "commodities" in params:
            quantity["commodities"] = params.pop("commodities")
        if "limits_path" not in quantity:
            msg = "Missing limits_path tag indicating file with finite resource limits"
            raise MissingSettings(msg)
        params["sink"] = params.get("sink", "finite_resource_logger")
        params["quantity"] = quantity
        return params

    parameters = cast(  # type: ignore
        OUTPUTS_PARAMETERS, [reformat_finite_resources(p) for p in parameters]
    )

    return _factory(OUTPUT_QUANTITIES, *parameters, sector_name="MCA")


@register_output_quantity
@round_values
def consumption(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current consumption."""
    return market_quantity(market.consumption, **kwargs).to_dataframe().reset_index()


@register_output_quantity
@round_values
def supply(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current supply."""
    return market_quantity(market.supply, **kwargs).to_dataframe().reset_index()


@register_output_quantity
@round_values
def prices(
    market: xr.Dataset,
    sectors: list[AbstractSector],
    year: int,
    **kwargs,
) -> pd.DataFrame:
    """Current MCA market prices."""
    return market_quantity(market.prices, **kwargs).to_dataframe().reset_index()


@register_output_quantity
@round_values
def capacity(
    market: xr.Dataset, sectors: list[AbstractSector], year: int, **kwargs
) -> pd.DataFrame:
    """Current capacity across all sectors."""
    return _aggregate_sectors(sectors, year, op=sector_capacity)


def sector_capacity(sector: AbstractSector, year: int) -> pd.DataFrame:
    """Sector capacity with agent annotations."""
    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Get capacity data for each agent
    for agent in agents:
        data_agent = agent.assets.capacity.sel(year=year)
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = data_agent.to_dataframe("capacity")
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
    return output


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
    from muse.costs import fuel_costs
    from muse.utilities import broadcast_over_assets

    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies")
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Calculate fuel costs
    _market = market.sel(year=year, commodity=technologies.commodity)
    for agent in agents:
        data_agent = fuel_costs(
            technologies=broadcast_over_assets(technologies, agent.assets),
            prices=broadcast_over_assets(
                _market.prices, agent.assets, installed_as_year=False
            ),
            consumption=agent.consumption.sel(year=year),
        )
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
            "fuel_consumption_costs"
        )
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
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
    from muse.costs import capital_costs
    from muse.utilities import broadcast_over_assets

    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies")
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Calculate capital costs
    for agent in agents:
        data_agent = capital_costs(
            technologies=broadcast_over_assets(technologies, agent.assets),
            capacity=agent.assets.capacity.sel(year=year),
            method="annual",
        )
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = data_agent.to_dataframe("capital_costs")
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
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
    from muse.costs import environmental_costs
    from muse.utilities import broadcast_over_assets

    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies")
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Calculate emission costs
    _market = market.sel(year=year, commodity=technologies.commodity)
    for agent in agents:
        data_agent = environmental_costs(
            technologies=broadcast_over_assets(technologies, agent.assets),
            prices=broadcast_over_assets(
                _market.prices, agent.assets, installed_as_year=False
            ),
            production=agent.supply.sel(year=year),
        )
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe(
            "emission_costs"
        )
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
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
    from muse.costs import levelized_cost_of_energy
    from muse.utilities import broadcast_over_assets

    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies")
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Calculate LCOE
    _market = market.sel(year=year, commodity=technologies.commodity)
    for agent in agents:
        data_agent = levelized_cost_of_energy(
            technologies=broadcast_over_assets(technologies, agent.assets),
            prices=broadcast_over_assets(
                _market.prices, agent.assets, installed_as_year=False
            ),
            capacity=agent.assets.capacity.sel(year=year),
            production=agent.supply.sel(year=year),
            consumption=agent.consumption.sel(year=year),
            method="annual",
        )
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe("lcoe")
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
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
    """Equivalent Annual Cost of technologies over their lifetime."""
    from muse.costs import equivalent_annual_cost
    from muse.utilities import broadcast_over_assets

    if isinstance(sector, PresetSector):
        return pd.DataFrame()

    # Get data for the sector
    data_sector: list[xr.DataArray] = []
    technologies = getattr(sector, "technologies")
    agents = sorted(getattr(sector, "agents"), key=attrgetter("name"))

    # Calculate EAC
    _market = market.sel(year=year, commodity=technologies.commodity)
    for agent in agents:
        data_agent = equivalent_annual_cost(
            technologies=broadcast_over_assets(technologies, agent.assets),
            prices=broadcast_over_assets(
                _market.prices, agent.assets, installed_as_year=False
            ),
            capacity=agent.assets.capacity.sel(year=year),
            production=agent.supply.sel(year=year),
            consumption=agent.consumption.sel(year=year),
        )
        data_agent["agent"] = agent.name
        data_agent["category"] = agent.category
        data_agent["sector"] = getattr(sector, "name", "unnamed")
        data_agent["year"] = year
        data_agent = multiindex_to_coords(data_agent, "timeslice").to_dataframe("lcoe")
        data_sector.append(data_agent)

    output = pd.concat(data_sector, sort=True).reset_index()
    return output
