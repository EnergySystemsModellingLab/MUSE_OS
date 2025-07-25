from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import (
    Any,
    Callable,
    cast,
)

import xarray as xr

from muse.agents import AbstractAgent
from muse.production import PRODUCTION_SIGNATURE
from muse.readers.toml import read_technodata
from muse.sectors.abstract import AbstractSector
from muse.sectors.register import register_sector
from muse.sectors.subsector import Subsector
from muse.timeslices import compress_timeslice, expand_timeslice, get_level


@register_sector(name="default")
class Sector(AbstractSector):  # type: ignore
    """Base class for all sectors."""

    @classmethod
    def factory(cls, name: str, settings: Any) -> Sector:
        from muse.interactions import factory as interaction_factory
        from muse.outputs.sector import factory as ofactory
        from muse.production import factory as pfactory

        # Read sector settings
        sector_settings = getattr(settings.sectors, name)._asdict()

        # Extract required settings
        subsectors = sector_settings.get("subsectors")
        if not subsectors:
            raise RuntimeError(f"Missing 'subsectors' section in sector {name}")
        if len(subsectors._asdict()) == 0:
            raise RuntimeError(f"Empty 'subsectors' section in sector {name}")
        interpolation_mode = sector_settings.get("interpolation", "linear")
        timeslice_level = sector_settings.get("timeslice_level", None)
        dispatch_production = sector_settings.get("dispatch_production", "share")
        outputs_config = sector_settings.get("outputs", [])
        interactions_config = sector_settings.get("interactions", None)

        # Read technologies
        technologies = read_technodata(
            settings,
            name,
            interpolation_mode=interpolation_mode,
        )

        # Create subsectors
        subsectors = [
            Subsector.factory(
                subsec_settings,
                technologies,
                regions=settings.regions,
                current_year=int(min(settings.time_framework)),
                name=subsec_name,
                timeslice_level=timeslice_level,
            )
            for subsec_name, subsec_settings in subsectors._asdict().items()
        ]

        # Check that subsector commodities are disjoint
        sector_commodities = [c for s in subsectors for c in s.commodities]
        duplicates = [
            c for c in set(sector_commodities) if sector_commodities.count(c) > 1
        ]
        if duplicates:
            raise RuntimeError(
                f"Commodities {duplicates} are outputted by multiple subsectors."
            )

        # Create outputs
        outputs = ofactory(*outputs_config, sector_name=name)

        # Create production method
        production = pfactory(dispatch_production)

        # Create interactions
        interactions = interaction_factory(interactions_config)

        # Create sector
        return cls(
            name,
            technologies,
            supply_prod=production,
            subsectors=subsectors,
            commodities=sector_commodities,
            outputs=outputs,
            interactions=interactions,
            timeslice_level=timeslice_level,
        )

    def __init__(
        self,
        name: str,
        technologies: xr.Dataset,
        supply_prod: PRODUCTION_SIGNATURE,
        subsectors: Sequence[Subsector] = [],
        commodities: list[str] = [],
        interactions: Callable[[Sequence[AbstractAgent]], None] | None = None,
        outputs: Callable | None = None,
        timeslice_level: str | None = None,
    ):
        from muse.interactions import factory as interaction_factory
        from muse.outputs.sector import factory as ofactory
        from muse.timeslices import TIMESLICE

        """Name of the sector."""
        self.name: str = name

        """Timeslice level for the sector (e.g. "month")."""
        self.timeslice_level = timeslice_level or get_level(TIMESLICE)

        """Subsectors controlled by this object."""
        self.subsectors: Sequence[Subsector] = list(subsectors)

        """Parameters describing the sector's technologies."""
        self.technologies: xr.Dataset = technologies
        if "timeslice" in self.technologies.dims:
            if not get_level(self.technologies) == self.timeslice_level:
                raise ValueError(
                    f"Technodata for {self.name} sector does not match "
                    "the specified timeslice level for that sector "
                    f"({self.timeslice_level})"
                )

        """Interactions between agents.

        Called right before computing new investments, this function should manage any
        interactions between agents, e.g. passing assets from *new* agents  to *retro*
        agents, and maket make-up from *retro* to *new*.

        Defaults to doing nothing.

        The function takes the sequence of agents as input, and returns nothing. It is
        expected to modify the agents in-place.

        See Also
        --------

        :py:mod:`muse.interactions` contains MUSE's base interactions
        """
        self.interactions = interactions or interaction_factory()

        """A function for outputting data for post-mortem analysis."""
        self.outputs: Callable = (
            cast(Callable, ofactory()) if outputs is None else outputs
        )

        """Computes production as used to return the supply to the MCA.

        It can be anything registered with
        :py:func:`@register_production<muse.production.register_production>`.
        """
        self.supply_prod = supply_prod

        """Full supply, consumption and costs data for the most recent year."""
        self.output_data: xr.Dataset

        """Commodities that the sector is in charge of producing."""
        self.commodities: list[str] = commodities

    def next(
        self,
        mca_market: xr.Dataset,
    ) -> xr.Dataset:
        """Advance sector by one time period.

        Args:
            mca_market:
                Market with ``demand``, ``supply``, and ``prices``.

        Returns:
            A market containing the ``supply`` offered by the sector, it's attendant
            ``consumption`` of fuels and materials and the associated ``costs``.
        """
        from logging import getLogger

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return xr.Dataset(dict(x=x)).groupby("region").sum("asset").x

        # Time period from the market object
        assert len(mca_market.year) == 2
        current_year, investment_year = map(int, mca_market.year.values)
        getLogger(__name__).info(
            f"Running {self.name} for years {current_year} to {investment_year}"
        )

        # Agent interactions
        self.interactions(list(self.agents))

        # Convert market to sector timeslicing
        mca_market = self.convert_to_sector_timeslicing(mca_market)

        # Select appropriate data from the market
        market = mca_market.sel(
            commodity=self.technologies.commodity.values,
            region=self.technologies.region,
        )

        # Select technology data from the investment year
        techs = self.technologies.sel(year=investment_year, drop=True)

        # Perform investments
        for subsector in self.subsectors:
            subsector.invest(technologies=techs, market=market)

        # Full output data
        supply, consume, costs = self.market_variables(market, self.technologies)
        self.output_data = xr.Dataset(
            dict(
                supply=supply,
                consumption=consume,
                costs=costs,
            )
        )

        # Output data for MCA (aggregated over assets)
        if len(supply.region.dims) == 0:
            output_data = self.output_data.sum("asset")
            output_data = output_data.expand_dims(region=[output_data.region.values])
        else:
            output_data = xr.Dataset(
                dict(
                    supply=group_assets(supply),
                    consumption=group_assets(consume),
                    costs=costs,
                )
            )

        result = output_data.copy(deep=True)
        if "dst_region" in result:
            exclude = ["dst_region", "commodity", "year", "timeslice"]
            prices = market.prices.expand_dims(dst_region=market.prices.region.values)
            sup, prices = xr.broadcast(result.supply, prices)
            sup = sup.fillna(0.0)
            con, prices = xr.broadcast(result.consumption, prices)
            con = con.fillna(0.0)
            supply = result.supply.sum("region").rename(dst_region="region")
            consumption = con.sum("dst_region")
            assert len(supply.region) == len(consumption.region)

            # Need to reindex costs to avoid nans for non-producing regions
            costs0, prices = xr.broadcast(result.costs, prices, exclude=exclude)
            # Fulfil nans with price values
            costs0 = costs0.reindex_like(prices).fillna(prices)
            costs0 = costs0.where(costs0 > 0, prices)
            # Find where sup >0 (exporter)
            # Importers have nans and average over exporting price
            costs = ((costs0 * sup) / sup.sum("dst_region")).fillna(
                costs0.mean("region")
            )

            # Take average over dst regions
            costs = costs.where(costs > 0, prices).mean("dst_region")

            result = xr.Dataset(
                dict(supply=supply, consumption=consumption, costs=costs)
            )

        # Convert result to global timeslicing scheme
        return self.convert_to_global_timeslicing(result)

    def save_outputs(self, year: int) -> None:
        """Calls the outputs function with the current output data."""
        self.outputs(self.output_data, self.capacity, year=year)

    def market_variables(self, market: xr.Dataset, technologies: xr.Dataset) -> Any:
        """Computes resulting market: production, consumption, and costs."""
        from muse.costs import levelized_cost_of_energy, supply_cost
        from muse.quantities import capacity_to_service_demand, consumption
        from muse.utilities import broadcast_over_assets, interpolate_capacity

        years = market.year.values
        capacity = interpolate_capacity(self.capacity, year=years)

        # Select technology data for each asset
        # Each asset uses the technology data from the year it was installed
        technodata = broadcast_over_assets(
            technologies, capacity, installed_as_year=True
        )

        # Select relevant investment year prices for each asset
        prices = broadcast_over_assets(market.prices.isel(year=1), capacity)

        # Calculate supply
        supply = self.supply_prod(
            market=market,
            capacity=capacity,
            technologies=technodata,
            timeslice_level=self.timeslice_level,
        )

        # Calculate consumption
        consume = consumption(
            technologies=technodata,
            production=supply,
            prices=prices,
            timeslice_level=self.timeslice_level,
        )

        # Calculate LCOE
        # We select data for the second year, which corresponds to the investment year
        # We base LCOE only on the portion of capacity that is actually used (#728)
        utilized_capacity = capacity_to_service_demand(
            demand=supply.isel(year=1),
            technologies=technodata,
            timeslice_level=self.timeslice_level,
        )
        lcoe = levelized_cost_of_energy(
            prices=prices,
            technologies=technodata,
            capacity=utilized_capacity,
            production=supply.isel(year=1),
            consumption=consume.isel(year=1),
            method="annual",
        )

        # Calculate new commodity prices
        costs = supply_cost(supply, lcoe, asset_dim="asset")

        return supply, consume, costs

    def convert_to_sector_timeslicing(self, market: xr.Dataset) -> xr.Dataset:
        """Converts market data to sector timeslicing."""
        supply = compress_timeslice(
            market["supply"], level=self.timeslice_level, operation="sum"
        )
        consumption = compress_timeslice(
            market["consumption"], level=self.timeslice_level, operation="sum"
        )
        prices = compress_timeslice(
            market["prices"], level=self.timeslice_level, operation="mean"
        )
        return xr.Dataset(dict(supply=supply, consumption=consumption, prices=prices))

    def convert_to_global_timeslicing(self, market: xr.Dataset) -> xr.Dataset:
        """Converts market data to global timeslicing."""
        supply = expand_timeslice(market["supply"], operation="distribute")
        consumption = expand_timeslice(market["consumption"], operation="distribute")
        costs = expand_timeslice(market["costs"], operation="broadcast")
        return xr.Dataset(dict(supply=supply, consumption=consumption, costs=costs))

    @property
    def capacity(self) -> xr.DataArray:
        """Aggregates capacity across agents.

        The capacities are aggregated leaving only two
        dimensions: asset (technology, installation date,
        region), year.
        """
        from muse.utilities import interpolate_capacity, reduce_assets

        traded = [
            u.assets.capacity
            for u in self.agents
            if "dst_region" in u.assets.capacity.dims
        ]
        nontraded = [
            u.assets.capacity
            for u in self.agents
            if "dst_region" not in u.assets.capacity.dims
        ]

        # Only nontraded assets
        if not traded:
            full_list = [
                list(nontraded[i].year.values)
                for i in range(len(nontraded))
                if "year" in nontraded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            capacity = reduce_assets(
                [
                    u.assets.capacity
                    for u in self.agents
                    if "dst_region" not in u.assets.capacity.dims
                ]
            )
            return interpolate_capacity(capacity, year=years)

        # Only traded assets
        elif not nontraded:
            full_list = [
                list(traded[i].year.values)
                for i in range(len(traded))
                if "year" in traded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            capacity = reduce_assets(
                [
                    u.assets.capacity
                    for u in self.agents
                    if "dst_region" in u.assets.capacity.dims
                ]
            )
            return interpolate_capacity(capacity, year=years)

        # Both traded and nontraded assets
        else:
            traded_results = reduce_assets(traded)
            nontraded_results = reduce_assets(nontraded)
            capacity = reduce_assets(
                [
                    traded_results,
                    nontraded_results
                    * (nontraded_results.region == traded_results.dst_region),
                ]
            )
            return interpolate_capacity(capacity, year=years)

    @property
    def agents(self) -> Iterator[AbstractAgent]:
        """Iterator over all agents in the sector."""
        for subsector in self.subsectors:
            yield from subsector.agents
