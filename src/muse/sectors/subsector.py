from __future__ import annotations

from typing import (
    Any,
    Callable,
    Hashable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Text,
    Tuple,
    Union,
)

import xarray as xr

from muse.agents import Agent


class Subsector:
    def __init__(
        self,
        agents: Sequence[Agent],
        commodities: Sequence[Text],
        demand_share: Optional[Callable] = None,
        constraints: Optional[Callable] = None,
        forecast: int = 5,
    ):
        from muse import demand_share as ds, constraints as cs

        self.agents: Sequence[Agent] = list(agents)
        self.commodities: List[Text] = list(commodities)
        self.demand_share = demand_share or ds.factory()
        self.constraints = constraints or cs.factory()
        self.forecast = forecast

    def invest(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        time_period: int = 5,
        current_year: Optional[int] = None,
    ) -> None:
        if current_year is None:
            current_year = market.year.min()
        lp_problem = self.aggregate_lp(
            technologies, market, time_period, current_year=current_year
        )
        if lp_problem is None:
            return
        solution = self.solve(*lp_problem)
        self.assign_back_to_agents(solution)

    def solve(self, cost: xr.Dataset, constraints: Sequence[xr.Dataset]) -> xr.Dataset:
        raise NotImplementedError()

    def assign_back_to_agents(self, solution: xr.Dataset):
        raise NotImplementedError()

    def aggregate_lp(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        time_period: int = 5,
        current_year: Optional[int] = None,
    ) -> Optional[Tuple[xr.Dataset, Sequence[xr.Dataset]]]:
        from muse.utilities import agent_concatenation, reduce_assets

        if current_year is None:
            current_year = market.year.min()

        demands = self.demand_share(
            self.agents,
            market,
            technologies,
            current_year=current_year,
            forecast=self.forecast,
        )
        agent_market = market.copy()
        assets = agent_concatenation(
            {agent.uuid: agent.assets for agent in self.agents}
        )
        agent_market["capacity"] = reduce_assets(
            assets.capacity, coords=("region", "technology")
        ).interp(year=market.year, method="linear", kwargs={"fill_value": 0.0})

        agent_lps: MutableMapping[Hashable, xr.Dataset] = {}
        for agent in self.agents:
            if "agent" in demands.coords:
                share = demands.sel(asset=demands.agent == agent.uuid)
            else:
                share = demands
            result = agent.next(
                technologies, agent_market, share, time_period=time_period
            )
            if result is not None:
                agent_lps[agent.uuid] = result

        if len(agent_lps) == 0:
            return None

        lps = agent_concatenation(agent_lps)
        constraints = self.constraints(
            demand=demands,
            assets=assets,
            search_space=lps.search_space,
            market=market,
            technologies=technologies,
            year=current_year,
        )
        return lps.decision, constraints

    @classmethod
    def factory(
        cls,
        settings: Any,
        technologies: xr.Dataset,
        regions: Optional[Sequence[Text]] = None,
        current_year: Optional[int] = None,
    ) -> Subsector:
        from muse.agents import agents_factory
        from muse.demand_share import factory as share_factory
        from muse.constraints import factory as constraints_factory

        agents = agents_factory(
            settings.agents,
            settings.existing_capacity,
            technologies=technologies,
            regions=regions,
            year=current_year or int(technologies.year.min()),
            investment=getattr(settings, "lpsolver", "adhoc"),
        )

        if hasattr(settings, "commodities"):
            commodities = settings.commodities
        else:
            commodities = aggregate_enduses(
                [agent.assets for agent in agents], technologies
            )
        if len(commodities) == 0:
            raise RuntimeError("Subsector commodities cannot be empty")

        demand_share = share_factory(getattr(settings, "demand_share", None))
        constraints = constraints_factory(getattr(settings, "constraints", None))
        forecast = getattr(settings, "forecast", 5)

        return cls(agents, commodities, demand_share, constraints, forecast)


def aggregate_enduses(
    assets: Sequence[Union[xr.Dataset, xr.DataArray]], technologies: xr.Dataset
) -> Sequence[Text]:
    """Aggregate enduse commodities for input assets.

    This function is meant as a helper to figure out the commodities attached to a group
    of agents.
    """
    from muse.commodities import is_enduse

    techs = set.union(*(set(data.technology.values) for data in assets))
    outputs = technologies.fixed_outputs.sel(
        commodity=is_enduse(technologies.comm_usage), technology=list(techs)
    )
    return outputs.commodity.sel(
        commodity=outputs.any([u for u in outputs.dims if u != "commodity"])
    )
