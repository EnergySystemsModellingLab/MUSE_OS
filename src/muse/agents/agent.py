"""Holds all building agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import xarray as xr

from muse.timeslices import drop_timeslice


class AbstractAgent(ABC):
    """Base class for all agents."""

    tolerance = 1e-12
    """tolerance criteria for floating point comparisons."""

    def __init__(
        self,
        name: str = "Agent",
        region: str = "",
        assets: xr.Dataset | None = None,
        category: str | None = None,
        quantity: float | None = 1,
        timeslice_level: str | None = None,
    ):
        """Creates a standard MUSE agent.

        Arguments:
            name: Name of the agent, used for cross-refencing external tables
            region: Region where the agent operates, used for cross-referencing
                external tables.
            assets: dataset holding information about the assets owned by this
                instance. The information should not be anything describing the
                technologies themselves, but rather the stock of assets held by
                the agent.
            category: optional value that could be used to classify different agents
                together.
            quantity: optional value to classify different agents' share of the
                population.
            timeslice_level: the timeslice level over which investments/production
                will be optimized (e.g "hour", "day"). If None, the agent will use the
                finest timeslice level.
        """
        from uuid import uuid4

        super().__init__()
        self.name = name
        """Name associated with the agent."""
        self.region = region
        """Region the agent operates in."""
        self.assets = assets if assets is not None else xr.Dataset()
        """Current stock of technologies."""
        self.uuid = uuid4()
        """A unique identifier for the agent."""
        self.category = category
        """Attribute to classify different sets of agents."""
        self.quantity = quantity
        """Attribute to classify different agents' share of the population."""
        self.timeslice_level = timeslice_level
        """Timeslice level for the agent."""

    def filter_input(
        self,
        dataset: xr.Dataset | xr.DataArray,
        **kwargs,
    ) -> xr.Dataset | xr.DataArray:
        """Filter inputs for usage in agent.

        For instance, filters down to agent's region, etc.
        """
        if "region" in dataset.dims and "region" not in kwargs:
            kwargs["region"] = self.region
        return dataset.sel(**kwargs)

    @abstractmethod
    def next(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
    ) -> None:
        """Increments agent to the next time point (e.g. performing investments).

        Performs investments to meet demands, and increments agent.year to the
        investment year.

        Arguments:
            technologies: dataset of technology parameters for the investment year
            market: market dataset covering the current year and investment year
            demand: data array of demand for the investment year
        """

    def __repr__(self):
        return (
            f"<{self.region}:({self.name}, {self.category}) "
            f"- {self.__class__.__name__} at "
            f"{hex(id(self))}>"
        )


class Agent(AbstractAgent):
    """Standard agent that does not perform investments."""

    def __init__(
        self,
        name: str = "Agent",
        region: str = "USA",
        assets: xr.Dataset | None = None,
        search_rules: Callable | None = None,
        objectives: Callable | None = None,
        decision: Callable | None = None,
        year: int = 2010,
        maturity_threshold: float = 0,
        housekeeping: Callable | None = None,
        merge_transform: Callable | None = None,
        demand_threshold: float | None = None,
        category: str | None = None,
        asset_threshold: float = 1e-4,
        quantity: float | None = 1,
        spend_limit: int = 0,
        timeslice_level: str | None = None,
        **kwargs,
    ):
        """Creates a standard agent.

        Arguments:
            name: Name of the agent, used for cross-refencing external tables
            region: Region where the agent operates, used for cross-referencing
                external tables.
            assets: Current stock of technologies.
            search_rules: method used to filter the search space
            objectives: One or more objectives by which to decide next investments.
            decision: single decision objective from one or more objectives.
            year: year the agent is created / current year
            maturity_threshold: threshold when filtering replacement
                technologies with respect to market share
            housekeeping: transform applied to the assets at the start of
                iteration. Defaults to doing nothing.
            merge_transform: transform merging current and newly invested assets
                together. Defaults to replacing old assets completely.
            demand_threshold: criteria below which the demand is zero.
            category: optional attribute that could be used to classify
                different agents together.
            asset_threshold: Threshold below which assets are not added.
            quantity: different agents' share of the population
            spend_limit: The cost above which agents will not invest
            timeslice_level: the timeslice level over which the agent invesments will
                be optimized (e.g "hour", "day"). If None, the agent will use the finest
                timeslice level.
            **kwargs: Extra arguments
        """
        from muse.decisions import factory as decision_factory
        from muse.filters import factory as filter_factory
        from muse.hooks import asset_merge_factory, housekeeping_factory
        from muse.objectives import factory as objectives_factory

        super().__init__(
            name=name,
            region=region,
            assets=assets,
            category=category,
            quantity=quantity,
            timeslice_level=timeslice_level,
        )

        """ Current year. Incremented by one every time next is called."""
        self.year = year

        """Search rule(s) determining potential replacement technologies.

        This is a string referring to a filter, or a sequence of strings
        referring to multiple filters, applied one after the other.  Any
        function registered via `muse.filters.register_filter` can be
        used to filter the search space.
        """
        if search_rules is None:
            search_rules = filter_factory()
        self.search_rules: Callable = search_rules

        """ Market share threshold.

        Threshold when and if filtering replacement technologies with respect
        to market share.
        """
        self.maturity_threshold = maturity_threshold

        self.spend_limit = spend_limit

        """One or more objectives by which to decide next investments."""
        if objectives is None:
            objectives = objectives_factory()
        self.objectives = objectives

        """Creates single decision objective from one or more objectives."""
        if decision is None:
            decision = decision_factory()
        self.decision = decision

        """Transforms applied on the assets at the start of each iteration.

        It could mean keeping the assets as are, or removing assets with no
        capacity in the current year and beyond, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_initial_asset_transform`.
        """
        if housekeeping is None:
            housekeeping = housekeeping_factory()
        self._housekeeping = housekeeping

        """Transforms applied on the old and new assets.

        It could mean using only the new assets, or merging old and new, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_final_asset_transform`.
        """
        if merge_transform is None:
            merge_transform = asset_merge_factory()
        self.merge_transform = merge_transform

        """Threshold below which the demand share is zero.

        This criteria avoids fulfilling demand for very small values. If None,
        then the criteria is not applied.
        """
        self.demand_threshold = demand_threshold

        """Threshold below which assets are not added."""
        self.asset_threshold = asset_threshold

    def asset_housekeeping(self):
        """Reduces memory footprint of assets.

        Performs tasks such as:

        - remove empty assets
        - remove years prior to current
        """
        # TODO: move this into search and make sure filters, demand_share and
        #  what not use assets from search. That would remove another bit of
        #  state.
        self.assets = self._housekeeping(self, self.assets)

    def next(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
    ) -> None:
        investment_year = int(market.year[1])
        self.year = investment_year


class InvestingAgent(Agent):
    """Agent that performs investment for itself."""

    def __init__(
        self,
        *args,
        constraints: Callable | None = None,
        investment: Callable | None = None,
        **kwargs,
    ):
        """Creates an investing agent.

        Arguments:
            *args: See :py:class:`~muse.agents.agent.Agent`
            constraints: Set of constraints limiting investment
            investment: A function to perform investments
            **kwargs: See :py:class:`~muse.agents.agent.Agent`
        """
        from muse.constraints import factory as csfactory
        from muse.investments import factory as ifactory

        super().__init__(*args, **kwargs)

        self.invest = investment or ifactory()
        """Method to use when fulfilling demand from rated set of techs."""
        self.constraints = constraints or csfactory()
        """Creates a set of constraints limiting investment."""

    def next(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
    ) -> None:
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to
        invest in and by how much.

        This function will modify `self.assets` and increment `self.year`.
        Other attributes are left unchanged. Arguments to the function are
        never modified.
        """
        from logging import getLogger

        from muse.utilities import interpolate_capacity, reduce_assets

        # Check inputs
        assert len(market.year) == 2
        assert "year" not in technologies.dims
        assert "year" not in demand.dims

        # Time period
        current_year, investment_year = map(int, market.year.values)
        assert current_year == self.year

        # Skip forward if demand is zero
        if demand.size == 0 or demand.sum() < 1e-12:
            self.year = investment_year
            return None

        # Calculate the search space
        search_space = (
            self.search_rules(self, demand, technologies=technologies, market=market)
            .fillna(0)
            .astype(int)
        )

        # Select technologies in the search space
        technologies = technologies.sel(
            technology=technologies.technology.isin(search_space.replacement)
        )

        # Skip forward if the search space is empty
        if any(u == 0 for u in search_space.shape):
            getLogger(__name__).critical("Search space is empty")
            self.year = investment_year
            return None

        # Calculate the decision metric
        decision = self.compute_decision(technologies, market, demand, search_space)
        search = xr.Dataset(dict(search_space=search_space, decision=decision))
        if "timeslice" in search.dims:
            search["demand"] = drop_timeslice(demand)
        else:
            search["demand"] = demand

        # Filter assets with demand
        not_assets = [u for u in search.demand.dims if u != "asset"]
        condtechs = (
            search.demand.sum(not_assets) > getattr(self, "tolerance", 1e-8)
        ).values
        search = search.sel(asset=condtechs)

        # Calculate capacity in current and investment year
        capacity = interpolate_capacity(
            reduce_assets(self.assets.capacity, coords=("technology", "region")),
            year=[current_year, investment_year],
        )

        # Calculate constraints
        constraints = self.constraints(
            demand=search.demand,
            capacity=capacity,
            search_space=search.search_space,
            technologies=technologies,
            timeslice_level=self.timeslice_level,
        )

        # Calculate investments
        investments = self.invest(
            search=search[["search_space", "decision"]],
            technologies=technologies,
            constraints=constraints,
            commodities=list(demand.commodity.values),
            timeslice_level=self.timeslice_level,
        )

        # Add investments
        self.add_investments(
            technologies=technologies,
            investments=investments,
            investment_year=investment_year,
        )

        # Increment the year
        self.year = investment_year

    def compute_decision(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
        search_space: xr.DataArray,
    ) -> xr.DataArray:
        # Check inputs
        assert "year" not in technologies.dims
        assert "year" not in demand.dims
        assert "year" not in search_space.dims
        assert len(market.year) == 2

        # Filter technologies according to the search space and region
        techs = self.filter_input(
            technologies,
            technology=search_space.replacement,
        ).drop_vars("technology")

        # Reduce dimensions of the demand array
        reduced_demand = demand.sel(
            {
                k: search_space[k]
                for k in set(demand.dims).intersection(search_space.dims)
            }
        )

        # Filter prices according to the region
        prices = self.filter_input(market.prices)

        # Select prices for the investment year
        investment_year_prices = prices.isel(year=1)

        # Compute the objectives
        objectives = self.objectives(
            technologies=techs,
            demand=reduced_demand,
            prices=investment_year_prices,
            timeslice_level=self.timeslice_level,
        )

        # Compute the decision metric
        decision = self.decision(objectives)
        return decision

    def add_investments(
        self,
        technologies: xr.Dataset,
        investments: xr.DataArray,
        investment_year: int,
    ) -> None:
        """Add new assets to the agent."""
        assert "year" not in technologies.dims

        # Calculate retirement profile of new assets
        new_capacity = self.retirement_profile(
            technologies, investments, investment_year
        )
        if new_capacity is None:
            return
        new_capacity = new_capacity.drop_vars(
            set(new_capacity.coords) - set(self.assets.coords)
        )
        new_assets = xr.Dataset(dict(capacity=new_capacity))

        # Merge new assets with existing assets
        self.assets = self.merge_transform(self.assets, new_assets)

    def retirement_profile(
        self,
        technologies: xr.Dataset,
        investments: xr.DataArray,
        investment_year: int,
    ) -> xr.DataArray | None:
        from muse.investments import cliff_retirement_profile

        assert "year" not in technologies.dims

        # Sum investments
        if "asset" in investments.dims:
            investments = investments.sum("asset")
        if "agent" in investments.dims:
            investments = investments.squeeze("agent", drop=True)

        # Filter out investments below the threshold
        investments = investments.sel(
            replacement=(investments > self.asset_threshold).any(
                [d for d in investments.dims if d != "replacement"]
            )
        )
        if investments.size == 0:
            return None

        # Calculate the retirement profile for new investments
        # Note: technical life must be at least the length of the time period
        lifetime = self.filter_input(
            technologies.technical_life,
            technology=investments.replacement,
        )
        profile = cliff_retirement_profile(
            lifetime,
            investment_year=investment_year,
        )
        if "dst_region" in investments.coords:
            investments = investments.reindex_like(profile, method="ffill")

        # Apply the retirement profile to the investments
        new_assets = (investments * profile).rename(replacement="asset")
        new_assets["installed"] = "asset", [investment_year] * len(new_assets.asset)

        # The new assets have picked up quite a few coordinates along the way.
        # we try and keep only those that were there originally.
        if set(new_assets.dims) != set(self.assets.dims):
            new, old = new_assets.dims, self.assets.dims
            raise RuntimeError(f"Asset dimensions do not match: {new} vs {old}")
        return new_assets
