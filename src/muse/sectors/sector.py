from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import (
    Any,
    Callable,
    cast,
)

import pandas as pd
import xarray as xr

from muse.agents import AbstractAgent
from muse.production import PRODUCTION_SIGNATURE
from muse.sectors.abstract import AbstractSector
from muse.sectors.register import register_sector
from muse.sectors.subsector import Subsector


@register_sector(name="default")
class Sector(AbstractSector):  # type: ignore
    """Base class for all sectors."""

    @classmethod
    def factory(cls, name: str, settings: Any) -> Sector:
        from muse.interactions import factory as interaction_factory
        from muse.outputs.sector import factory as ofactory
        from muse.production import factory as pfactory
        from muse.readers import read_timeslices
        from muse.readers.toml import read_technodata
        from muse.utilities import nametuple_to_dict

        sector_settings = getattr(settings.sectors, name)._asdict()
        for attribute in ("name", "type", "priority", "path"):
            sector_settings.pop(attribute, None)

        timeslices = read_timeslices(
            sector_settings.pop("timeslice_levels", None)
        ).get_index("timeslice")

        technologies = read_technodata(settings, name, settings.time_framework)

        if "subsectors" not in sector_settings:
            raise RuntimeError(f"Missing 'subsectors' section in sector {name}")
        if len(sector_settings["subsectors"]._asdict()) == 0:
            raise RuntimeError(f"Empty 'subsectors' section in sector {name}")
        subsectors = [
            Subsector.factory(
                subsec_settings,
                technologies,
                regions=settings.regions,
                current_year=int(min(settings.time_framework)),
                name=subsec_name,
            )
            for subsec_name, subsec_settings in sector_settings.pop("subsectors")
            ._asdict()
            .items()
        ]
        are_disjoint_commodities = sum(len(s.commodities) for s in subsectors) == len(
            set().union(*(set(s.commodities) for s in subsectors))  # type: ignore
        )
        if not are_disjoint_commodities:
            raise RuntimeError("Subsector commodities are not disjoint")

        outputs = ofactory(*sector_settings.pop("outputs", []), sector_name=name)

        supply_args = sector_settings.pop(
            "supply", sector_settings.pop("dispatch_production", {})
        )
        if isinstance(supply_args, str):
            supply_args = {"name": supply_args}
        else:
            supply_args = nametuple_to_dict(supply_args)
        supply = pfactory(**supply_args)

        interactions = interaction_factory(sector_settings.pop("interactions", None))

        for attr in ("technodata", "commodities_out", "commodities_in"):
            sector_settings.pop(attr, None)
        return cls(
            name,
            technologies,
            subsectors=subsectors,
            timeslices=timeslices,
            supply_prod=supply,
            outputs=outputs,
            interactions=interactions,
            **sector_settings,
        )

    def __init__(
        self,
        name: str,
        technologies: xr.Dataset,
        subsectors: Sequence[Subsector] = [],
        timeslices: pd.MultiIndex | None = None,
        technodata_timeslices: xr.Dataset = None,
        interactions: Callable[[Sequence[AbstractAgent]], None] | None = None,
        interpolation: str = "linear",
        outputs: Callable | None = None,
        supply_prod: PRODUCTION_SIGNATURE | None = None,
    ):
        from muse.interactions import factory as interaction_factory
        from muse.outputs.sector import factory as ofactory
        from muse.production import maximum_production

        self.name: str = name
        """Name of the sector."""
        self.subsectors: Sequence[Subsector] = list(subsectors)
        """Subsectors controlled by this object."""
        self.technologies: xr.Dataset = technologies
        """Parameters describing the sector's technologies."""
        self.timeslices: pd.MultiIndex | None = timeslices
        """Timeslice at which this sector operates.

        If None, it will operate using the timeslice of the input market.
        """
        self.interpolation: Mapping[str, Any] = {
            "method": interpolation,
            "kwargs": {"fill_value": "extrapolate"},
        }
        """Interpolation method and arguments when computing years."""
        if interactions is None:
            interactions = interaction_factory()
        self.interactions = interactions
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
        self.outputs: Callable = (
            cast(Callable, ofactory()) if outputs is None else outputs
        )
        """A function for outputting data for post-mortem analysis."""
        self.supply_prod = (
            supply_prod if supply_prod is not None else maximum_production
        )
        """ Computes production as used to return the supply to the MCA.

        It can be anything registered with
        :py:func:`@register_production<muse.production.register_production>`.
        """
        self.output_data: xr.Dataset
        """Full supply, consumption and costs data for the most recent year."""

    @property
    def forecast(self):
        """Maximum forecast horizon across agents.

        If no agents with a "forecast" attribute are found, defaults to 5. It cannot be
        lower than 1 year.
        """
        forecasts = [
            getattr(agent, "forecast")
            for agent in self.agents
            if hasattr(agent, "forecast")
        ]
        if len(forecasts) == 0:
            return 5
        return max(1, max(forecasts))

    def next(
        self,
        mca_market: xr.Dataset,
        time_period: int | None = None,
        current_year: int | None = None,
    ) -> xr.Dataset:
        """Advance sector by one time period.

        Args:
            mca_market:
                Market with ``demand``, ``supply``, and ``prices``.
            time_period:
                Length of the time period in the framework. Defaults to the range of
                ``mca_market.year``.
            current_year: Current year of the simulation

        Returns:
            A market containing the ``supply`` offered by the sector, it's attendant
            ``consumption`` of fuels and materials and the associated ``costs``.
        """
        from logging import getLogger

        def group_assets(x: xr.DataArray) -> xr.DataArray:
            return xr.Dataset(dict(x=x)).groupby("region").sum("asset").x

        if time_period is None:
            time_period = int(mca_market.year.max() - mca_market.year.min())
        if current_year is None:
            current_year = int(mca_market.year.min())
        getLogger(__name__).info(f"Running {self.name} for year {current_year}")

        # > to sector timeslice
        market = self.convert_market_timeslice(
            mca_market.sel(
                commodity=self.technologies.commodity, region=self.technologies.region
            ).interp(
                year=sorted(
                    {
                        current_year,
                        current_year + time_period,
                        current_year + self.forecast,
                    }
                ),
                **self.interpolation,
            ),
            self.timeslices,
        )
        # > agent interactions
        self.interactions(list(self.agents))
        # > investment
        years = sorted(
            set(
                market.year.data.tolist()
                + self.capacity.installed.data.tolist()
                + self.technologies.year.data.tolist()
            )
        )
        technologies = self.technologies.interp(year=years, **self.interpolation)

        for subsector in self.subsectors:
            subsector.invest(
                technologies, market, time_period=time_period, current_year=current_year
            )

        # Full output data
        supply, consume, costs = self.market_variables(market, technologies)
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
        result = self.convert_market_timeslice(result, mca_market.timeslice)
        result["comm_usage"] = technologies.comm_usage.sel(commodity=result.commodity)
        result.set_coords("comm_usage")
        return result

    def save_outputs(self) -> None:
        """Calls the outputs function with the current output data."""
        self.outputs(self.output_data, self.capacity)

    def market_variables(self, market: xr.Dataset, technologies: xr.Dataset) -> Any:
        """Computes resulting market: production, consumption, and costs."""
        from muse.commodities import is_pollutant
        from muse.costs import annual_levelized_cost_of_energy, supply_cost
        from muse.quantities import consumption
        from muse.timeslices import QuantityType, convert_timeslice
        from muse.utilities import broadcast_techs

        years = market.year.values
        capacity = self.capacity.interp(year=years, **self.interpolation)

        supply = self.supply_prod(
            market=market, capacity=capacity, technologies=technologies
        )

        if "timeslice" in market.prices.dims and "timeslice" not in supply.dims:
            supply = convert_timeslice(supply, market.timeslice, QuantityType.EXTENSIVE)

        consume = consumption(technologies, supply, market.prices)

        technodata = cast(xr.Dataset, broadcast_techs(technologies, supply))
        costs = supply_cost(
            supply.where(~is_pollutant(supply.comm_usage), 0),
            annual_levelized_cost_of_energy(
                prices=market.prices.sel(region=supply.region), technologies=technodata
            ),
            asset_dim="asset",
        )

        return supply, consume, costs

    @property
    def capacity(self) -> xr.DataArray:
        """Aggregates capacity across agents.

        The capacities are aggregated leaving only two
        dimensions: asset (technology, installation date,
        region), year.
        """
        from muse.utilities import filter_input, reduce_assets

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
        if not traded:
            full_list = [
                list(nontraded[i].year.values)
                for i in range(len(nontraded))
                if "year" in nontraded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            nontraded = [
                filter_input(u.assets.capacity, year=years)
                for u in self.agents
                if "dst_region" not in u.assets.capacity.dims
            ]
            return reduce_assets(nontraded)
        if not nontraded:
            full_list = [
                list(traded[i].year.values)
                for i in range(len(traded))
                if "year" in traded[i].dims
            ]
            flat_list = [item for sublist in full_list for item in sublist]
            years = sorted(list(set(flat_list)))
            traded = [
                filter_input(u.assets.capacity, year=years)
                for u in self.agents
                if "dst_region" in u.assets.capacity.dims
            ]
            return reduce_assets(traded)
        traded_results = reduce_assets(traded)
        nontraded_results = reduce_assets(nontraded)
        return reduce_assets(
            [
                traded_results,
                nontraded_results
                * (nontraded_results.region == traded_results.dst_region),
            ]
        )

    @property
    def agents(self) -> Iterator[AbstractAgent]:
        """Iterator over all agents in the sector."""
        for subsector in self.subsectors:
            yield from subsector.agents

    @staticmethod
    def convert_market_timeslice(
        market: xr.Dataset,
        timeslice: pd.MultiIndex,
        intensive: str | tuple[str] = "prices",
    ) -> xr.Dataset:
        """Converts market from one to another timeslice."""
        from muse.timeslices import QuantityType, convert_timeslice

        if isinstance(intensive, str):
            intensive = (intensive,)

        timesliced = {d for d in market.data_vars if "timeslice" in market[d].dims}
        intensives = convert_timeslice(
            market[list(timesliced.intersection(intensive))],
            timeslice,
            QuantityType.INTENSIVE,
        )
        extensives = convert_timeslice(
            market[list(timesliced.difference(intensives.data_vars))],
            timeslice,
            QuantityType.EXTENSIVE,
        )
        others = market[list(set(market.data_vars).difference(timesliced))]
        return xr.merge([intensives, extensives, others])
