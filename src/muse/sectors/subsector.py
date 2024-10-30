from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Any,
    Callable,
)

import numpy as np
import xarray as xr

from muse.agents import Agent


class Subsector:
    """Agent group servicing a subset of the sectorial commodities."""

    def __init__(
        self,
        agents: Sequence[Agent],
        commodities: Sequence[str],
        demand_share: Callable | None = None,
        constraints: Callable | None = None,
        investment: Callable | None = None,
        name: str = "subsector",
        forecast: int = 5,
    ):
        from muse import constraints as cs
        from muse import demand_share as ds
        from muse import investments as iv

        self.agents: Sequence[Agent] = list(agents)
        self.commodities: list[str] = list(commodities)
        self.demand_share = demand_share or ds.factory()
        self.constraints = constraints or cs.factory()
        self.investment = investment or iv.factory()
        self.forecast = forecast
        self.name = name

    def invest(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        time_period: int,
        current_year: int,
    ) -> None:
        # Agent housekeeping
        for agent in self.agents:
            agent.asset_housekeeping()

        # Perform the investments
        self.aggregate_lp(technologies, market, time_period, current_year=current_year)

    def aggregate_lp(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        time_period,
        current_year,
    ) -> None:
        from muse.utilities import agent_concatenation, reduce_assets

        # Split demand across agents
        demands = self.demand_share(
            self.agents,
            market,
            technologies,
            current_year=current_year,
            forecast=self.forecast,
        )

        # Concatenate assets
        assets = agent_concatenation(
            {agent.uuid: agent.assets for agent in self.agents}
        )

        # Calculate existing capacity
        agent_market = market.copy()
        agent_market["capacity"] = (
            reduce_assets(assets.capacity, coords=("region", "technology"))
            .interp(year=market.year, method="linear", kwargs={"fill_value": 0.0})
            .swap_dims(dict(asset="technology"))
        )

        # Increment each agent (perform investments)
        for agent in self.agents:
            if "agent" in demands.coords:
                share = demands.sel(asset=demands.agent == agent.uuid)
            else:
                share = demands
            agent.next(technologies, agent_market, share, time_period=time_period)

    @classmethod
    def factory(
        cls,
        settings: Any,
        technologies: xr.Dataset,
        regions: Sequence[str] | None = None,
        current_year: int | None = None,
        name: str = "subsector",
    ) -> Subsector:
        from muse import constraints as cs
        from muse import demand_share as ds
        from muse import investments as iv
        from muse.agents import agents_factory
        from muse.commodities import is_enduse
        from muse.readers.toml import undo_damage

        # Raise error for renamed asset_threshhold parameter (PR #447)
        if hasattr(settings, "asset_threshhold"):
            msg = "Invalid parameter asset_threshhold. Did you mean asset_threshold?"
            raise ValueError(msg)

        agents = agents_factory(
            settings.agents,
            settings.existing_capacity,
            technologies=technologies,
            regions=regions,
            year=current_year or int(technologies.year.min()),
            asset_threshold=getattr(settings, "asset_threshold", 1e-12),
            # only used by self-investing agents
            investment=getattr(settings, "lpsolver", "adhoc"),
            forecast=getattr(settings, "forecast", 5),
            constraints=getattr(settings, "constraints", ()),
        )
        # technologies can have nans where a commodity
        # does not apply to a technology at all
        # (i.e. hardcoal for a technology using hydrogen)

        # check that all regions have technologies with at least one end-use output
        for a in agents:
            techs = a.filter_input(technologies, region=a.region)
            outputs = techs.fixed_outputs.sel(
                commodity=is_enduse(technologies.comm_usage)
            )
            msg = f"Subsector with {techs.technology.values[0]} for region {a.region} has no output commodities"  # noqa: E501

            if len(outputs) == 0:
                raise RuntimeError(msg)

            if np.sum(outputs) == 0.0:
                raise RuntimeError(msg)

        if hasattr(settings, "commodities"):
            commodities = settings.commodities
        else:
            commodities = aggregate_enduses(
                [agent.assets for agent in agents], technologies
            )

        # len(commodities) == 0 may happen only if
        # we run only one region or all regions have no outputs
        msg = f"Subsector with {techs.technology.values[0]} has no output commodities"
        if len(commodities) == 0:
            raise RuntimeError(msg)

        demand_share = ds.factory(undo_damage(getattr(settings, "demand_share", None)))
        constraints = cs.factory(getattr(settings, "constraints", None))
        # only used by non-self-investing agents
        investment = iv.factory(getattr(settings, "lpsolver", "scipy"))
        forecast = getattr(settings, "forecast", 5)

        return cls(
            agents=agents,
            commodities=commodities,
            demand_share=demand_share,
            constraints=constraints,
            investment=investment,
            forecast=forecast,
            name=name,
        )


def aggregate_enduses(
    assets: Sequence[xr.Dataset | xr.DataArray], technologies: xr.Dataset
) -> Sequence[str]:
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
    ).values.tolist()
