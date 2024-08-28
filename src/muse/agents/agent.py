"""Holds all building agents."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable, Optional, Union

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
        assets: Optional[xr.Dataset] = None,
        interpolation: str = "linear",
        category: Optional[str] = None,
        quantity: Optional[float] = 1,
    ):
        """Creates a standard MUSE agent.

        Arguments:
            name: Name of the agent, used for cross-refencing external tables
            region: Region where the agent operates, used for cross-referencing
                external tables.
            interpolation: interpolation method. see `xarray.interp`.
            assets: dataset holding information about the assets owned by this
                instance. The information should not be anything describing the
                technologies themselves, but rather the stock of assets held by
                the agent.
            category: optional value that could be used to classify different agents
                together.
            quantity: optional value to classify different agents' share of the
                population.
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
        self.interpolation = interpolation
        """Interpolation method."""
        self.category = category
        """Attribute to classify different sets of agents."""
        self.quantity = quantity
        """Attribute to classify different agents' share of the population."""

    def filter_input(
        self,
        dataset: Union[xr.Dataset, xr.DataArray],
        year: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Filter inputs for usage in agent.

        For instance, filters down to agent's region, etc.
        """
        from muse.utilities import filter_input

        if "region" in dataset.dims and "region" not in kwargs:
            kwargs["region"] = self.region
        return filter_input(dataset, year=year, **kwargs)

    @abstractmethod
    def next(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
        time_period: int = 1,
    ):
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to invest in
        and by how much.
        """
        pass

    def __repr__(self):
        return (
            f"<{self.region}:({self.name}, {self.category}) "
            f"- {self.__class__.__name__} at "
            f"{hex(id(self))}>"
        )


class Agent(AbstractAgent):
    """Agent that is capable of computing a search-space and a cost metric.

    This agent will not perform any investment itself.
    """

    def __init__(
        self,
        name: str = "Agent",
        region: str = "USA",
        assets: Optional[xr.Dataset] = None,
        interpolation: str = "linear",
        search_rules: Optional[Callable] = None,
        objectives: Optional[Callable] = None,
        decision: Optional[Callable] = None,
        year: int = 2010,
        maturity_threshold: float = 0,
        forecast: int = 5,
        housekeeping: Optional[Callable] = None,
        merge_transform: Optional[Callable] = None,
        demand_threshold: Optional[float] = None,
        category: Optional[str] = None,
        asset_threshold: float = 1e-4,
        quantity: Optional[float] = 1,
        spend_limit: int = 0,
        **kwargs,
    ):
        """Creates a standard buildings agent.

        Arguments:
            name: Name of the agent, used for cross-refencing external tables
            region: Region where the agent operates, used for cross-referencing
                external tables.
            assets: Current stock of technologies.
            interpolation: interpolation method. see `xarray.interp`.
            search_rules: method used to filter the search space
            objectives: One or more objectives by which to decide next investments.
            decision: single decision objective from one or more objectives.
            year: year the agent is created / current year
            maturity_threshold: threshold when filtering replacement
                technologies with respect to market share
            forecast: Number of years the agent will forecast
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
            interpolation=interpolation,
            category=category,
            quantity=quantity,
        )

        self.year = year
        """ Current year.

        The year is incremented by one every time next is called.
        """
        self.forecast = forecast
        """Number of years to look into the future for forecating purposed."""
        if search_rules is None:
            search_rules = filter_factory()
        self.search_rules: Callable = search_rules
        """Search rule(s) determining potential replacement technologies.

        This is a string referring to a filter, or a sequence of strings
        referring to multiple filters, applied one after the other.  Any
        function registered via `muse.filters.register_filter` can be
        used to filter the search space.
        """
        self.maturity_threshold = maturity_threshold
        """ Market share threshold.

        Threshold when and if filtering replacement technologies with respect
        to market share.
        """
        self.spend_limit = spend_limit

        if objectives is None:
            objectives = objectives_factory()
        self.objectives = objectives
        """One or more objectives by which to decide next investments."""
        if decision is None:
            decision = decision_factory()
        self.decision = decision
        """Creates single decision objective from one or more objectives."""
        if housekeeping is None:
            housekeeping = housekeeping_factory()
        self._housekeeping = housekeeping
        """Transforms applied on the assets at the start of each iteration.

        It could mean keeping the assets as are, or removing assets with no
        capacity in the current year and beyond, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_initial_asset_transform`.
        """
        if merge_transform is None:
            merge_transform = asset_merge_factory()
        self.merge_transform = merge_transform
        """Transforms applied on the old and new assets.

        It could mean using only the new assets, or merging old and new, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_final_asset_transform`.
        """
        self.demand_threshold = demand_threshold
        """Threshold below which the demand share is zero.

        This criteria avoids fulfilling demand for very small values. If None,
        then the criteria is not applied.
        """
        self.asset_threshold = asset_threshold
        """Threshold below which assets are not added."""

    @property
    def forecast_year(self):
        """Year to consider when forecasting."""
        return self.year + self.forecast

    def asset_housekeeping(self):
        """Reduces memory footprint of assets.

        Performs tasks such as:

        - remove empty assets
        - remove years prior to current
        - interpolate current year and forecasted year
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
        time_period: int = 1,
    ) -> Optional[xr.Dataset]:
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to
        invest in and by how much.

        This function will modify `self.assets` and increment `self.year`.
        Other attributes are left unchanged. Arguments to the function are
        never modified.
        """
        from logging import getLogger

        # dataset with intermediate computational results from search
        # makes it easier to pass intermediate results to functions, as well as
        # filter them when inside a function
        if demand.size == 0 or demand.sum() < 1e-12:
            self.year += time_period
            return None

        search_space = (
            self.search_rules(self, demand, technologies, market).fillna(0).astype(int)
        )

        if any(u == 0 for u in search_space.shape):
            getLogger(__name__).critical("Search space is empty")
            self.year += time_period
            return None

        # Filter technologies according to the search space, forecast year and region
        techs = self.filter_input(
            technologies,
            technology=search_space.replacement,
            year=self.forecast_year,
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

        # Compute the objective
        decision = self._compute_objective(
            technologies=techs, demand=reduced_demand, prices=prices
        )

        self.year += time_period
        return xr.Dataset(dict(search_space=search_space, decision=decision))

    def _compute_objective(
        self,
        technologies: xr.Dataset,
        demand: xr.DataArray,
        prices: xr.DataArray,
    ) -> xr.DataArray:
        objectives = self.objectives(
            technologies=technologies, demand=demand, prices=prices
        )
        decision = self.decision(objectives)
        return decision

    def add_investments(
        self,
        technologies: xr.Dataset,
        investments: xr.DataArray,
        current_year: int,
        time_period: int,
    ):
        """Add new assets to the agent."""
        new_capacity = self.retirement_profile(
            technologies, investments, current_year, time_period
        )

        if new_capacity is None:
            return
        new_capacity = new_capacity.drop_vars(
            set(new_capacity.coords) - set(self.assets.coords)
        )
        new_assets = xr.Dataset(dict(capacity=new_capacity))

        self.assets = self.merge_transform(self.assets, new_assets)

    def retirement_profile(
        self,
        technologies: xr.Dataset,
        investments: xr.DataArray,
        current_year: int,
        time_period: int,
    ) -> Optional[xr.DataArray]:
        from muse.investments import cliff_retirement_profile

        if "asset" in investments.dims:
            investments = investments.sum("asset")
        if "agent" in investments.dims:
            investments = investments.squeeze("agent", drop=True)
        investments = investments.sel(
            replacement=(investments > self.asset_threshold).any(
                [d for d in investments.dims if d != "replacement"]
            )
        )
        if investments.size == 0:
            return None

        # figures out the retirement profile for the new investments
        lifetime = self.filter_input(
            technologies.technical_life,
            year=current_year,
            technology=investments.replacement,
        )
        profile = cliff_retirement_profile(
            lifetime.clip(min=time_period),
            current_year=current_year + time_period,
            protected=max(self.forecast - time_period - 1, 0),
        )
        if "dst_region" in investments.coords:
            investments = investments.reindex_like(profile, method="ffill")

        new_assets = (investments * profile).rename(replacement="asset")

        new_assets["installed"] = "asset", [current_year] * len(new_assets.asset)

        # The new assets have picked up quite a few coordinates along the way.
        # we try and keep only those that were there originally.
        if set(new_assets.dims) != set(self.assets.dims):
            new, old = new_assets.dims, self.assets.dims
            raise RuntimeError(f"Asset dimensions do not match: {new} vs {old}")
        return new_assets


class InvestingAgent(Agent):
    """Agent that performs investment for itself."""

    def __init__(
        self,
        *args,
        constraints: Optional[Callable] = None,
        investment: Optional[Callable] = None,
        **kwargs,
    ):
        """Creates a standard buildings agent.

        Arguments:
            *args: See :py:class:`~muse.agents.agent.Agent`
            constraints: Set of constraints limiting investment
            investment: A function to perform investments
            **kwargs: See :py:class:`~muse.agents.agent.Agent`
        """
        from muse.constraints import factory as csfactory
        from muse.investments import factory as ifactory

        super().__init__(*args, **kwargs)

        if investment is None:
            investment = ifactory()
        self.invest = investment
        """Method to use when fulfilling demand from rated set of techs."""
        if not callable(constraints):
            constraints = csfactory()
        self.constraints = constraints
        """Creates a set of constraints limiting investment."""

    def next(
        self,
        technologies: xr.Dataset,
        market: xr.Dataset,
        demand: xr.DataArray,
        time_period: int = 1,
    ):
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to
        invest in and by how much.

        This function will modify `self.assets` and increment `self.year`.
        Other attributes are left unchanged. Arguments to the function are
        never modified.
        """
        current_year = self.year
        search = super().next(technologies, market, demand, time_period=time_period)
        if search is None:
            return None

        if "timeslice" in search.dims:
            search["demand"] = drop_timeslice(demand)
        else:
            search["demand"] = demand
        not_assets = [u for u in search.demand.dims if u != "asset"]
        condtechs = (
            search.demand.sum(not_assets) > getattr(self, "tolerance", 1e-8)
        ).values
        search = search.sel(asset=condtechs)
        constraints = self.constraints(
            search.demand,
            self.assets,
            search.search_space,
            market,
            technologies,
            year=current_year,
        )

        investments = self.invest(
            search[["search_space", "decision"]],
            technologies,
            constraints,
            year=current_year,
        )

        self.add_investments(
            technologies,
            investments,
            current_year=self.year - time_period,
            time_period=time_period,
        )
