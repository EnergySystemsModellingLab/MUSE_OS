from __future__ import annotations

from typing import (
    Any,
    Callable,
<<<<<<< HEAD
    Iterator,
=======
    List,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    Mapping,
    Optional,
    Sequence,
    Text,
    Tuple,
    Union,
    cast,
)

<<<<<<< HEAD
import pandas as pd
import xarray as xr

from muse.agents import AbstractAgent
from muse.production import PRODUCTION_SIGNATURE
from muse.sectors.abstract import AbstractSector
from muse.sectors.register import register_sector
from muse.sectors.subsector import Subsector
=======
from pandas import MultiIndex
from xarray import DataArray, Dataset

from muse.agent import AgentBase
from muse.demand_share import DEMAND_SHARE_SIGNATURE
from muse.production import PRODUCTION_SIGNATURE
from muse.sectors.abstract import AbstractSector
from muse.sectors.register import register_sector
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


@register_sector(name="default")
class Sector(AbstractSector):  # type: ignore
    """Base class for all sectors."""

    @classmethod
    def factory(cls, name: Text, settings: Any) -> Sector:
<<<<<<< HEAD
        from muse.readers import read_timeslices
        from muse.readers.toml import read_technodata
=======
        from muse.readers import read_timeslices, read_technologies
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        from muse.utilities import nametuple_to_dict
        from muse.outputs.sector import factory as ofactory
        from muse.production import factory as pfactory
        from muse.interactions import factory as interaction_factory
<<<<<<< HEAD
=======
        from muse.demand_share import factory as share_factory
        from muse.agent import agents_factory
        from logging import getLogger
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

        sector_settings = getattr(settings.sectors, name)._asdict()
        for attribute in ("name", "type", "priority", "path"):
            sector_settings.pop(attribute, None)

        timeslices = read_timeslices(
            sector_settings.pop("timeslice_levels", None)
        ).get_index("timeslice")

<<<<<<< HEAD
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
        are_disjoint_commodities = sum((len(s.commodities) for s in subsectors)) == len(
            set().union(*(set(s.commodities) for s in subsectors))  # type: ignore
        )
        if not are_disjoint_commodities:
            raise RuntimeError("Subsector commodities are not disjoint")

        outputs = ofactory(*sector_settings.pop("outputs", []), sector_name=name)

=======
        # We get and filter the technologies
        technologies = read_technologies(
            sector_settings.pop("technodata"),
            sector_settings.pop("commodities_out"),
            sector_settings.pop("commodities_in"),
            commodities=settings.global_input_files.global_commodities,
        )
        ins = (technologies.fixed_inputs > 0).any(("year", "region", "technology"))
        outs = (technologies.fixed_outputs > 0).any(("year", "region", "technology"))
        techcomms = technologies.commodity[ins | outs]
        technologies = technologies.sel(commodity=techcomms, region=settings.regions)

        # Finally, we create the agents
        agents = agents_factory(
            sector_settings.pop("agents"),
            sector_settings.pop("existing_capacity"),
            technologies=technologies,
            regions=settings.regions,
            year=min(settings.time_framework),
        )

        # make sure technologies includes the requisite years
        maxyear = max(a.forecast for a in agents) + max(settings.time_framework)
        if technologies.year.max() < maxyear:
            msg = "Forward-filling technodata to fit simulation timeframe"
            getLogger(__name__).info(msg)
            years = technologies.year.data.tolist() + [maxyear]
            technologies = technologies.sel(year=years, method="ffill")
            technologies["year"] = "year", years
        minyear = min(settings.time_framework)
        if technologies.year.min() > minyear:
            msg = "Back-filling technodata to fit simulation timeframe"
            getLogger(__name__).info(msg)
            years = [minyear] + technologies.year.data.tolist()
            technologies = technologies.sel(year=years, method="bfill")
            technologies["year"] = "year", years

        outputs = ofactory(*sector_settings.pop("outputs", []), sector_name=name)

        production_args = sector_settings.pop(
            "production", sector_settings.pop("investment_production", {})
        )
        if isinstance(production_args, Text):
            production_args = {"name": production_args}
        else:
            production_args = nametuple_to_dict(production_args)
        production = pfactory(**production_args)

>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        supply_args = sector_settings.pop(
            "supply", sector_settings.pop("dispatch_production", {})
        )
        if isinstance(supply_args, Text):
            supply_args = {"name": supply_args}
        else:
            supply_args = nametuple_to_dict(supply_args)
        supply = pfactory(**supply_args)

        interactions = interaction_factory(sector_settings.pop("interactions", None))

<<<<<<< HEAD
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
=======
        demand_share = share_factory(sector_settings.pop("demand_share", None))

        return cls(
            name,
            technologies,
            agents,
            timeslices=timeslices,
            production=production,
            supply_prod=supply,
            outputs=outputs,
            interactions=interactions,
            demand_share=demand_share,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            **sector_settings,
        )

    def __init__(
        self,
        name: Text,
<<<<<<< HEAD
        technologies: xr.Dataset,
        subsectors: Sequence[Subsector] = [],
        timeslices: Optional[pd.MultiIndex] = None,
        interactions: Optional[Callable[[Sequence[AbstractAgent]], None]] = None,
        interpolation: Text = "linear",
        outputs: Optional[Callable] = None,
        supply_prod: Optional[PRODUCTION_SIGNATURE] = None,
=======
        technologies: Dataset,
        agents: Sequence[AgentBase] = [],
        timeslices: Optional[MultiIndex] = None,
        interactions: Optional[Callable[[Sequence[AgentBase]], None]] = None,
        interpolation: Text = "linear",
        outputs: Optional[Callable] = None,
        production: Optional[PRODUCTION_SIGNATURE] = None,
        supply_prod: Optional[PRODUCTION_SIGNATURE] = None,
        demand_share: Optional[DEMAND_SHARE_SIGNATURE] = None,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    ):
        from muse.production import maximum_production
        from muse.outputs.sector import factory as ofactory
        from muse.interactions import factory as interaction_factory
<<<<<<< HEAD

        self.name: Text = name
        """Name of the sector."""
        self.subsectors: Sequence[Subsector] = list(subsectors)
        """Subsectors controlled by this object."""
        self.technologies: xr.Dataset = technologies
        """Parameters describing the sector's technologies."""
        self.timeslices: Optional[pd.MultiIndex] = timeslices
=======
        from muse.demand_share import factory as share_factory

        self.name: Text = name
        """Name of the sector."""
        self.agents: List[AgentBase] = list(agents)
        """Agents controlled by this object."""
        self.technologies: Dataset = technologies
        """Parameters describing the sector's technologies."""
        self.timeslices: Optional[MultiIndex] = timeslices
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        """Timeslice at which this sector operates.

        If None, it will operate using the timeslice of the input market.
        """
        self.interpolation: Mapping[Text, Any] = {
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
        """A function for outputing data for post-mortem analysis."""
<<<<<<< HEAD
=======
        self.production = production if production is not None else maximum_production
        """ Computes production as used for investment demands.

        It can be anything registered with
        :py:func:`@register_production<muse.production.register_production>`.
        """
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        self.supply_prod = (
            supply_prod if supply_prod is not None else maximum_production
        )
        """ Computes production as used to return the supply to the MCA.

        It can be anything registered with
        :py:func:`@register_production<muse.production.register_production>`.
        """
<<<<<<< HEAD
=======
        if demand_share is None:
            demand_share = share_factory()
        self.demand_share = demand_share
        """Method defining how to split the input demand amongst agents.

        This is a function registered by :py:func:`@register_demand_share
        <muse.demand_share.register_demand_share>`.
        """
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

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

<<<<<<< HEAD
    def next(
        self,
        mca_market: xr.Dataset,
        time_period: Optional[int] = None,
        current_year: Optional[int] = None,
    ) -> xr.Dataset:
=======
    def next(self, mca_market: Dataset, time_period: Optional[int] = None) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        """Advance sector by one time period.

        Args:
            mca_market:
                Market with ``demand``, ``supply``, and ``prices``.
            time_period:
                Length of the time period in the framework. Defaults to the range of
                ``mca_market.year``.

        Returns:
            A market containing the ``supply`` offered by the sector, it's attendant
            ``consumption`` of fuels and materials and the associated ``costs``.
        """
        from logging import getLogger

        if time_period is None:
            time_period = int(mca_market.year.max() - mca_market.year.min())
<<<<<<< HEAD
        if current_year is None:
            current_year = int(mca_market.year.min())
        getLogger(__name__).info(f"Running {self.name} for year {current_year}")
=======
        getLogger(__name__).info(f"Running {self.name} for year {time_period}")
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

        # > to sector timeslice
        market = self.convert_market_timeslice(
            mca_market.sel(
                commodity=self.technologies.commodity, region=self.technologies.region
            ).interp(
                year=sorted(
                    {
<<<<<<< HEAD
                        current_year,
                        current_year + time_period,
                        current_year + self.forecast,
=======
                        int(mca_market.year.min()),
                        int(mca_market.year.min()) + time_period,
                        int(mca_market.year.min()) + self.forecast,
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
                    }
                ),
                **self.interpolation,
            ),
            self.timeslices,
        )
        # > agent interactions
<<<<<<< HEAD
        self.interactions(list(self.agents))
=======
        self.interactions(self.agents)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        # > investment
        years = sorted(
            set(
                market.year.data.tolist()
                + self.capacity.installed.data.tolist()
                + self.technologies.year.data.tolist()
            )
        )
        technologies = self.technologies.interp(year=years, **self.interpolation)
<<<<<<< HEAD
        for subsector in self.subsectors:
            subsector.invest(
                technologies, market, time_period=time_period, current_year=current_year
            )
        # > output to mca
        output_data = self.market_variables(market, technologies)
        # < output to mca
        self.outputs(output_data, self.capacity, technologies)
        # > to mca timeslices
        if len(output_data.region.dims) == 0:
            result = output_data.sum("asset")
            result = result.expand_dims(region=[result.region.values])
        else:
            result = output_data.groupby("region").sum("asset")
        result = self.convert_market_timeslice(result, mca_market.timeslice)
=======
        self.investment(market, technologies, time_period=time_period)
        # > output to mca
        result = self.market_variables(market, technologies)
        # < output to mca
        self.outputs(result, self.capacity, technologies)
        # > to mca timeslices
        result = self.convert_market_timeslice(
            result.groupby("region").sum("asset"), mca_market.timeslice
        )
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        result["comm_usage"] = technologies.comm_usage.sel(commodity=result.commodity)
        result.set_coords("comm_usage")
        # < to mca timeslices
        return result

<<<<<<< HEAD
    def market_variables(
        self, market: xr.Dataset, technologies: xr.Dataset
    ) -> xr.Dataset:
=======
    def market_variables(self, market: Dataset, technologies: Dataset) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        """Computes resulting market: production, consumption, and costs."""
        from muse.quantities import (
            consumption,
            supply_cost,
            annual_levelized_cost_of_energy,
        )
        from muse.commodities import is_pollutant
<<<<<<< HEAD
        from muse.utilities import broadcast_techs
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

        years = market.year.values
        capacity = self.capacity.interp(year=years, **self.interpolation)

<<<<<<< HEAD
        result = xr.Dataset()
=======
        result = Dataset()
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        result["supply"] = self.supply_prod(
            market=market, capacity=capacity, technologies=technologies
        )
        result["consumption"] = consumption(technologies, result.supply, market.prices)
<<<<<<< HEAD
        technodata = cast(xr.Dataset, broadcast_techs(technologies, result.supply))
        result["costs"] = supply_cost(
            result.supply.where(~is_pollutant(result.comm_usage), 0),
            annual_levelized_cost_of_energy(
                market.prices.sel(region=result.region), technodata
            ),
            asset_dim=None,
        )
        return result

    @property
    def capacity(self) -> xr.DataArray:
=======
        result["costs"] = supply_cost(
            result.supply.where(~is_pollutant(result.comm_usage), 0),
            annual_levelized_cost_of_energy(market.prices, technologies),
        ).sum("technology")
        return result

    def investment(
        self, market: Dataset, technologies: Dataset, time_period: Optional[int] = None
    ) -> None:
        """Computes demand share for each agent and run investment."""
        from logging import getLogger

        if time_period is None:
            time_period = int(market.year.max() - market.year.min())

        shares = self.demand_share(  # type: ignore
            self.agents,
            market,
            technologies,
            current_year=market.year.min(),
            forecast=self.forecast,
        )
        capacity = self.capacity.interp(
            year=market.year,
            method=self.interpolation["method"],
            kwargs={"fill_value": 0.0},
        )
        agent_market = market.copy()
        agent_market["capacity"] = self.asset_capacity(capacity)

        for agent in self.agents:
            assert market.year.min() == getattr(agent, "year", market.year.min())
            if shares[agent.uuid].size == 0:
                getLogger(__name__).critical(
                    "Demand share is empty, no investment needed "
                    f"for {agent.category} agent {agent.name} "
                    f"of {self.name} sector in year {int(agent_market.year.min())}."
                )
            elif shares[agent.uuid].sum() < 1e-12:
                getLogger(__name__).critical(
                    "No demand, no investment needed for "
                    f"for {agent.category} agent {agent.name} "
                    f"of {self.name} sector in year {int(agent_market.year.min())}."
                )

            agent.next(
                technologies, agent_market, shares[agent.uuid], time_period=time_period
            )

    @property
    def capacity(self) -> DataArray:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        """Aggregates capacity across agents.

        The capacities are aggregated leaving only two
        dimensions: asset (technology, installation date,
        region), year.
        """
        from muse.utilities import reduce_assets

        return reduce_assets([u.assets.capacity for u in self.agents])

<<<<<<< HEAD
    @property
    def agents(self) -> Iterator[AbstractAgent]:
        """Iterator over all agents in the sector."""
        for subsector in self.subsectors:
            yield from subsector.agents

    @staticmethod
    def convert_market_timeslice(
        market: xr.Dataset,
        timeslice: pd.MultiIndex,
        intensive: Union[Text, Tuple[Text]] = "prices",
    ) -> xr.Dataset:
        """Converts market from one to another timeslice."""
=======
    def _trajectory(self, market: Dataset, capacity: DataArray, technologies: Dataset):
        from muse.quantities import supply

        production = self.production(
            market=market, capacity=capacity, technologies=technologies
        )
        supp = supply(production, market.consumption, technologies).sum("asset")
        return (market.consumption - supp).clip(min=0)

    def decommissioning_demand(
        self, capacity: DataArray, technologies: Dataset, year: int
    ) -> DataArray:
        from muse.quantities import decommissioning_demand

        return decommissioning_demand(
            technologies, capacity, [int(year), int(year) + self.forecast]
        )

    def asset_capacity(self, capacity: DataArray) -> DataArray:
        from muse.utilities import reduce_assets, coords_to_multiindex

        capa = reduce_assets(capacity, ("region", "technology"))
        return cast(
            DataArray, coords_to_multiindex(capa, "asset").unstack("asset").fillna(0)
        )

    @staticmethod
    def convert_market_timeslice(
        market: Dataset,
        timeslice: MultiIndex,
        intensive: Union[Text, Tuple[Text]] = "prices",
    ) -> Dataset:
        """Converts market from one to another timeslice."""
        from xarray import merge
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        from muse.timeslices import convert_timeslice, QuantityType

        if isinstance(intensive, Text):
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
<<<<<<< HEAD
        return xr.merge([intensives, extensives, others])
=======
        return merge([intensives, extensives, others])
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
