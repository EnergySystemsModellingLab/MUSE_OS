from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Any,
    Callable,
)

import numpy as np
import xarray as xr

from muse.agents import Agent
from muse.timeslices import drop_timeslice


class Subsector:
    """Agent group servicing a subset of the sectorial commodities."""

    def __init__(
        self,
        agents: Sequence[Agent],
        commodities: Sequence[str],
        demand_share: Callable,
        constraints: Callable,
        investment: Callable,
        name: str = "subsector",
        expand_market_prices: bool = False,
        timeslice_level: str | None = None,
    ):
        self.agents: Sequence[Agent] = list(agents)
        self.commodities: list[str] = list(commodities)
        self.demand_share = demand_share
        self.constraints = constraints
        self.investment = investment
        self.name = name
        self.expand_market_prices = expand_market_prices
        self.timeslice_level = timeslice_level
        """Whether to expand prices to include destination region.

        If ``True``, the input market prices are expanded of the missing "dst_region"
        dimension by setting them to the maximum between the source and destination
        region.
        """

    def invest(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
    ) -> None:
        assert "year" not in technologies.dims
        assert len(market.year) == 2

        # Expand prices to include destination region (for trade models)
        if self.expand_market_prices:
            market = market.copy()
            market["prices"] = drop_timeslice(
                np.maximum(market.prices, market.prices.rename(region="dst_region"))
            )

        # Agent housekeeping
        for agent in self.agents:
            agent.asset_housekeeping()

        # Perform the investments
        self.aggregate_lp(technologies, market)

    def aggregate_lp(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
    ) -> None:
        assert "year" not in technologies.dims
        assert len(market.year) == 2

        # Select commodity demands for the subsector
        demands = market.consumption.sel(commodity=self.commodities)

        # Remove commodities that have no demand in the investment year
        mask = (demands.isel(year=1, drop=True) > 0).any(dim=["timeslice", "region"])
        demands = demands.sel(commodity=mask)

        # Split demand across agents
        demands = self.demand_share(
            agents=self.agents,
            demand=demands,
            technologies=technologies,
            timeslice_level=self.timeslice_level,
        )

        if "dst_region" in demands.dims:
            msg = """
                dst_region found in demand dimensions. This is unexpected. Demands
                should only have a region dimension rather both a source and destination
                dimension.
            """
            raise ValueError(msg)

        # Increment each agent (perform investments)
        for agent in self.agents:
            if "agent" in demands.coords:
                share = demands.sel(asset=demands.agent == agent.uuid)
            else:
                share = demands
            agent.next(technologies=technologies, market=market, demand=share)

    @classmethod
    def factory(
        cls,
        settings: Any,
        technologies: xr.Dataset,
        regions: Sequence[str] | None = None,
        current_year: int | None = None,
        name: str = "subsector",
        timeslice_level: str | None = None,
    ) -> Subsector:
        from muse import constraints as cs
        from muse import demand_share as ds
        from muse import investments as iv
        from muse.agents import InvestingAgent, agents_factory
        from muse.commodities import is_enduse
        from muse.readers import read_csv, read_existing_trade, read_initial_capacity

        # Read existing capacity or existing trade file
        # Have to peek at the file to determine what format the data is in
        # TODO: ideally would be more explicit about this. Consider changing
        # the parameter name in the settings file
        df = read_csv(settings.existing_capacity)
        if "year" not in df.columns:
            existing_capacity = read_initial_capacity(settings.existing_capacity)
        else:
            existing_capacity = read_existing_trade(settings.existing_capacity)

        # Create agents
        agents = agents_factory(
            settings.agents,
            capacity=existing_capacity,
            technologies=technologies,
            regions=regions,
            year=current_year or int(technologies.year.min()),
            asset_threshold=getattr(settings, "asset_threshold", 1e-12),
            # only used by self-investing agents
            investment=getattr(settings, "lpsolver", "scipy"),
            constraints=getattr(settings, "constraints", ()),
            timeslice_level=timeslice_level,
        )
        # technologies can have nans where a commodity
        # does not apply to a technology at all
        # (i.e. hardcoal for a technology using hydrogen)

        # check that all regions have technologies with at least one end-use output
        # TODO: move this check to the input layer
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

        # Get list of commodities for the subsector
        if hasattr(settings, "commodities"):
            commodities = settings.commodities
        else:
            # If commodities aren't explicitly specified, we infer the commodities from
            # the existing capacity file
            commodities = aggregate_enduses(
                technologies.sel(technology=existing_capacity.technology.values)
            )

        # len(commodities) == 0 may happen only if
        # we run only one region or all regions have no outputs
        msg = f"Subsector with {techs.technology.values[0]} has no output commodities"
        if len(commodities) == 0:
            raise RuntimeError(msg)

        demand_share = ds.factory(getattr(settings, "demand_share", "standard_demand"))
        constraints = cs.factory(getattr(settings, "constraints", None))
        # only used by non-self-investing agents
        investment = iv.factory(getattr(settings, "lpsolver", "scipy"))

        expand_market_prices = getattr(settings, "expand_market_prices", None)
        if expand_market_prices is None:
            expand_market_prices = "dst_region" in technologies.dims and not any(
                isinstance(u, InvestingAgent) for u in agents
            )

        return cls(
            agents=agents,
            commodities=commodities,
            demand_share=demand_share,
            constraints=constraints,
            investment=investment,
            name=name,
            expand_market_prices=expand_market_prices,
            timeslice_level=timeslice_level,
        )


def aggregate_enduses(technologies: xr.Dataset) -> list[str]:
    """Aggregate enduse commodities for a set of technologies.

    Returns a list of all enduse commodities associated with the technologies in the
    input dataset. Enduse commodities are determined using based on the `comm_usage`
    attribute of the technologies, using the `is_enduse` function from the
    `muse.commodities` module.
    """
    from muse.commodities import is_enduse

    # We select enduse commodities with positive fixed outputs
    outputs = technologies.fixed_outputs
    enduse_output = outputs.any(
        [u for u in outputs.dims if u != "commodity"]
    ) * is_enduse(technologies.comm_usage)

    return technologies.commodity.values[enduse_output].tolist()
