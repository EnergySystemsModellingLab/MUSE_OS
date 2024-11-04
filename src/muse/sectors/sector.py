from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import (
    Any,
    Callable,
    cast,
)

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
        from muse.readers.toml import read_technodata
        from muse.utilities import nametuple_to_dict

        # Read sector settings
        sector_settings = getattr(settings.sectors, name)._asdict()
        for attribute in ("name", "type", "priority", "path"):
            sector_settings.pop(attribute, None)
        if "subsectors" not in sector_settings:
            raise RuntimeError(f"Missing 'subsectors' section in sector {name}")
        if len(sector_settings["subsectors"]._asdict()) == 0:
            raise RuntimeError(f"Empty 'subsectors' section in sector {name}")

        # Read technologies
        technologies = read_technodata(settings, name, settings.time_framework)

        # Create subsectors
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

        # Check that subsector commodities are disjoint
        are_disjoint_commodities = sum(len(s.commodities) for s in subsectors) == len(
            set().union(*(set(s.commodities) for s in subsectors))  # type: ignore
        )
        if not are_disjoint_commodities:
            raise RuntimeError("Subsector commodities are not disjoint")

        # Create outputs
        outputs = ofactory(*sector_settings.pop("outputs", []), sector_name=name)

        supply_args = sector_settings.pop(
            "supply", sector_settings.pop("dispatch_production", {})
        )
        if isinstance(supply_args, str):
            supply_args = {"name": supply_args}
        else:
            supply_args = nametuple_to_dict(supply_args)
        supply = pfactory(**supply_args)

        # Create interactions
        interactions = interaction_factory(sector_settings.pop("interactions", None))

        # Create sector
        for attr in (
            "technodata",
            "commodities_out",
            "commodities_in",
            "technodata_timeslices",
        ):
            sector_settings.pop(attr, None)
        return cls(
            name,
            technologies,
            subsectors=subsectors,
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

        It cannot be lower than 1 year.
        """
        forecasts = [getattr(agent, "forecast") for agent in self.agents]
        return max(1, max(forecasts))

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

        time_period = int(mca_market.year.max() - mca_market.year.min())
        current_year = int(mca_market.year.min())
        getLogger(__name__).info(f"Running {self.name} for year {current_year}")

        # Agent interactions
        self.interactions(list(self.agents))

        # Select appropriate data from the market
        market = mca_market.sel(
            commodity=self.technologies.commodity, region=self.technologies.region
        )

        # Investments
        for subsector in self.subsectors:
            subsector.invest(
                self.technologies,
                market,
                time_period=time_period,
                current_year=current_year,
            )

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
        result = self.convert_market_timeslice(result, mca_market.timeslice)
        result["comm_usage"] = self.technologies.comm_usage.sel(
            commodity=result.commodity
        )
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
        from muse.utilities import broadcast_techs

        years = market.year.values
        capacity = self.capacity.interp(year=years, **self.interpolation)

        # Calculate supply
        supply = self.supply_prod(
            market=market, capacity=capacity, technologies=technologies
        )

        # Calculate consumption
        consume = consumption(technologies, supply, market.prices)

        # Calculate commodity prices
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

        capacities = [u.assets.capacity for u in self.agents]
        all_years = sorted({year for capa in capacities for year in capa.year.values})
        capacities = [filter_input(capa, year=all_years) for capa in capacities]
        return reduce_assets(capacities)

    @property
    def agents(self) -> Iterator[AbstractAgent]:
        """Iterator over all agents in the sector."""
        for subsector in self.subsectors:
            yield from subsector.agents
