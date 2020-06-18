"""Holds all building agents."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Text, Union

from xarray import DataArray, Dataset

from muse.defaults import DEFAULT_SECTORS_DIRECTORY


class AgentBase(ABC):
    """Base class for all agents."""

    tolerance = 1e-12
    """tolerance criteria for floating point comparisons."""

    def __init__(
        self,
        name: Text = "Agent",
        region: Text = "",
        assets: Optional[Dataset] = None,
        interpolation: Text = "linear",
        category: Optional[Text] = None,
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
            category: optional attribute that could be used to classify
                different agents together.
        """
        from uuid import uuid4

        super().__init__()
        self.name = name
        """ Name associated with the agent """
        self.region = region
        """ Region the agent operates in """
        self.assets = assets if assets is not None else Dataset()
        """Current stock of technologies."""
        self.uuid = uuid4()
        """A unique identifier for the agent."""
        self.interpolation = interpolation
        """Interpolation method."""
        self.category = category
        """Attribute to classify different sets of agents."""

    def filter_input(
        self,
        dataset: Union[Dataset, DataArray],
        year: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
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
        technologies: Dataset,
        market: Dataset,
        demand: DataArray,
        time_period: int = 1,
    ):
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to invest in
        and by how much.
        """
        pass

    def __repr__(self):
        return (
            f"<{self.region}:({self.name}, {self.category}) - object at "
            f"{hex(id(self))}>"
        )


class Agent(AgentBase):
    """Base class for buildings agents."""

    def __init__(
        self,
        name: Text = "Agent",
        region: Text = "USA",
        assets: Optional[Dataset] = None,
        interpolation: Text = "linear",
        search_rules: Optional[Callable] = None,
        objectives: Optional[Callable] = None,
        decision: Optional[Callable] = None,
        investment: Optional[Callable] = None,
        year: int = 2010,
        maturity_threshhold: float = 0,
        forecast: int = 5,
        housekeeping: Optional[Callable] = None,
        merge_transform: Optional[Callable] = None,
        demand_threshhold: Optional[float] = None,
        category: Optional[Text] = None,
        **kwargs,
    ):
        """Creates a standard buildings agent.

        Arguments:
            assets: Current stock of technologies.
            name: Name of the agent, used for cross-refencing external tables
            region: Region where the agent operates, used for cross-referencing
                external tables.
            search_rules: method used to filter the search space
            maturity_threshhold: threshhold when filtering replacement
                technologies with respect to market share
            year: year the agent is created / current year
            forecast: Number of years the agent will forecast
            housekeeping: transform applied to the assets at the start of
                iteration. Defaults to doing nothing.
            merge_transform: transform merging current and newly invested assets
                together. Defaults to replacing old assets completly.
            demand_threshhold: criteria below which the demand is zero.
            category: optional attribute that could be used to classify
                different agents together.
        """
        from muse.hooks import housekeeping_factory, asset_merge_factory
        from muse.filters import factory as filter_factory
        from muse.investments import factory as ifactory
        from muse.objectives import factory as objectives_factory
        from muse.decisions import factory as decision_factory

        super().__init__(
            name=name,
            region=region,
            assets=assets,
            interpolation=interpolation,
            category=category,
        )

        self.year = year
        """ Current year.

        The year is incremented by one everytime next is called.
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
        self.maturity_threshhold = maturity_threshhold
        """ Market share threshhold.

        Threshhold when and if filtering replacement technologies with respect
        to market share.
        """
        if objectives is None:
            objectives = objectives_factory()
        self.objectives = objectives
        """One or more objectives by which to decide next investments."""
        if decision is None:
            decision = decision_factory()
        self.decision = decision
        """Creates single decision objective from one or more objectives."""
        if investment is None:
            investment = ifactory()
        self.invest = investment
        """Method to use when fulfilling demand from rated set of techs."""
        if housekeeping is None:
            housekeeping = housekeeping_factory()
        self.housekeeping = housekeeping
        """Tranforms applied on the assets at the start of each iteration.

        It could mean keeping the assets as are, or removing assets with no
        capacity in the current year and beyond, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_initial_asset_transform`.
        """
        if merge_transform is None:
            merge_transform = asset_merge_factory()
        self.merge_transform = merge_transform
        """Tranforms applied on the old and new assets.

        It could mean using only the new assets, or merging old and new, etc...
        It can be any function registered with
        :py:func:`~muse.hooks.register_final_asset_transform`.
        """
        self.demand_threshhold = demand_threshhold
        """Threshhold below which the demand share is zero.

        This criteria avoids fulfilling demand for very small values. If None,
        then the criteria is not applied.
        """

    @property
    def forecast_year(self):
        """Year to consider when forecasting."""
        return self.year + self.forecast

    def next(
        self,
        technologies: Dataset,
        market: Dataset,
        demand: DataArray,
        time_period: int = 1,
    ):
        """Iterates agent one turn.

        The goal is to figure out from market variables which technologies to
        invest in and by how much.

        This function will modify `self.assets` and increment `self.year`.
        Other attributes are left unchanged. Arguments to the function are
        never modified.
        """
        from logging import getLogger

        # interpolate current year and forecasted year
        # remove empty assets
        # remove years prior to current
        # TODO: move this into search and make sure filters, demand_share and
        #  what not use assets from search. That would remove another bit of
        #  state.
        self.assets = self.housekeeping(self, self.assets)
        # dataset with intermediate computational results from search
        # makes it easier to pass intermediate results to functions, as well as
        # filter them when inside a function
        search = Dataset()
        if demand.size == 0 or demand.sum() < 1e-12:
            self.year += time_period
            return

        search["space"] = self.search_rules(self, demand, technologies, market)
        search["space"] = search["space"].fillna(0).astype(int)
        if any(u == 0 for u in search.space.shape):
            getLogger(__name__).critical("Search space is empty")
        search["decision"] = self._compute_objective(
            demand, search.space, technologies, market
        )

        new_assets = self._compute_new_assets(demand, search, technologies, time_period)

        # add invested capacity to current assets
        self.assets = self.merge_transform(
            self.assets, Dataset({"capacity": new_assets})
        )

        self.year += time_period

    def max_capacity_expansion(
        self,
        technologies: Dataset,
        technology: Optional[DataArray] = None,
        time_period: int = 1,
    ) -> DataArray:
        r"""Maximum capacity expansion.

        Limits by how much the capacity of each technology owned by an agent can grow in
        a given year. This is a constraint on the agent's ability to invest in a
        technology.

        Let :math:`L_t^r(y)` be the total capacity limit for a given year, technology,
        and region. :math:`G_t^r(y)` is the maximum growth. And :math:`W_t^r(y)` is
        the maximum additional capacity. :math:`y=y_0` is the current year and
        :math:`y=y_1` is the year marking the end of the investment period.

        Let :math:`\mathcal{A}^{i, r}_{t, \iota}(y)` be the current assets, before
        invesment, and let :math:`\Delta\mathcal{A}^{i,r}_t` be the future investements.
        The the constraint on agent :math:`i` are given as:

        .. math::

            L_t^r(y_0) - \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_1)
                \geq \Delta\mathcal{A}^{i,r}_t

            (y_1 - y_0 + 1) G_t^r(y_0) \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_0)
                - \sum_\iota \mathcal{A}^{i, r}_{t, \iota}(y_1)
                \geq \Delta\mathcal{A}^{i,r}_t

            (y_1 - y_0)W_t^r(y_0) \geq  \Delta\mathcal{A}^{i,r}_t

        The three constraints are combined into a single one which is returned as the
        maximum capacity expansion, :math:`\Gamma_t^{r, i}`. The maximum capacity
        expansion cannot impose negative investments:

        .. math::

            \Gamma_t^{r, i} \geq 0
        """
        if technology is None:
            technology = technologies.technology

        techs = self.filter_input(
            technologies[
                ["max_capacity_addition", "max_capacity_growth", "total_capacity_limit"]
            ],
            year=self.year,
            technology=technology,
        )
        assert isinstance(techs, Dataset)

        if len(self.assets.technology) != 0:
            capacity = (
                self.assets.capacity.groupby("technology")
                .sum("asset")
                .interp(year=[self.year, self.forecast_year], method=self.interpolation)
                .rename(technology=technology.dims[0])
                .reindex_like(technology)
                .fillna(0)
            )
        else:
            capacity = (
                self.assets.capacity.sum("asset")
                .interp(year=[self.year, self.forecast_year], method=self.interpolation)
                .fillna(0)
            )

        add_cap = techs.max_capacity_addition * time_period

        limit = techs.total_capacity_limit
        forecasted = capacity.sel(year=self.forecast_year, drop=True)
        total_cap = (limit - forecasted).clip(min=0).rename("total_cap")

        max_growth = techs.max_capacity_growth
        initial = capacity.sel(year=self.year, drop=True)
        growth_cap = initial * (max_growth * time_period + 1) - forecasted

        zero_cap = add_cap.where(add_cap < total_cap, total_cap)
        with_growth = zero_cap.where(zero_cap < growth_cap, growth_cap)
        result = with_growth.where(initial > 0, zero_cap)
        return result.rename("maximum capacity expansion")

    def _compute_objective(
        self,
        demand: DataArray,
        search_space: DataArray,
        technologies: Dataset,
        market: Dataset,
    ) -> DataArray:
        objectives = self.objectives(self, demand, search_space, technologies, market)
        result = self.decision(objectives)
        return result.rank("replacement")

    def _compute_new_assets(
        self,
        demand: DataArray,
        search: Dataset,
        technologies: Dataset,
        time_period: int,
    ) -> DataArray:
        """Computes investment and retirement profile."""
        from muse.investments import cliff_retirement_profile

        max_cap = self.max_capacity_expansion(
            technologies, technology=search.replacement, time_period=time_period
        ).drop_vars("technology")

        investments = self.invest(demand, search, max_cap, technologies, year=self.year)
        investments = investments.sum("asset")
        investments = investments.where(investments > self.tolerance, 0)

        # figures out the retirement profile for the new investments
        lifetime = self.filter_input(
            technologies.technical_life,
            year=self.year,
            technology=investments.replacement,
        )
        profile = cliff_retirement_profile(
            lifetime.clip(min=time_period),
            current_year=self.year + time_period,
            protected=max(self.forecast - time_period - 1, 0),
        )

        new_assets = (investments * profile).rename(replacement="asset")
        new_assets["installed"] = "asset", [self.year] * len(new_assets.asset)

        # The new assets have picked up quite a few coordinates along the way.
        # we try and keep only those that were there originally.
        if set(new_assets.dims) != set(self.assets.dims):
            new, old = new_assets.dims, self.assets.dims
            raise RuntimeError(f"Asset dimensions do not match: {new} vs {old}")
        return new_assets.drop_vars(set(new_assets.coords) - set(self.assets.coords))


def create_retrofit_agent(
    technologies: Dataset,
    capacity: DataArray,
    share: Text,
    year: int,
    region: Text,
    interpolation: Text = "linear",
    search_rules="all",
    housekeeping: Union[Text, Mapping, Callable] = "clean",
    merge_transform: Union[Text, Mapping, Callable] = "merge",
    objectives: Union[
        Callable, Text, Mapping, Sequence[Union[Text, Mapping]]
    ] = "fixed_costs",
    decision: Union[Callable, Text, Mapping] = "mean",
    **kwargs,
):
    """Creates retrofit agent from muse primitives."""
    from logging import getLogger
    from muse.filters import factory as filter_factory
    from muse.hooks import housekeeping_factory, asset_merge_factory
    from muse.objectives import factory as objectives_factory
    from muse.decisions import factory as decision_factory

    if "region" in capacity.dims:
        capacity = capacity.sel(region=region)
    if "region" in technologies.dims:
        technologies = technologies.sel(region=region)

    shares = technologies[share].sel(technology=capacity.technology)
    if "region" in shares.dims:
        shares = shares.sel(region=region)
    if "year" in shares.dims:
        shares = shares.interp({"year": year}, method=interpolation)

    existing = capacity.interp({"year": year}, method=interpolation)

    techs = ((existing > 0) & (shares > 0)).values
    assets = Dataset({"capacity": (capacity * shares).sel(asset=techs).copy()})

    if isinstance(search_rules, Text):
        search_rules = [search_rules]
    if len(search_rules) == 0 or search_rules[-1] != "compress":
        search_rules.append("compress")
    if len(search_rules) < 2 or search_rules[-2] != "with_asset_technology":
        search_rules.insert(-1, "with_asset_technology")

    if not callable(housekeeping):
        housekeeping = housekeeping_factory(housekeeping)
    if not callable(merge_transform):
        merge_transform = asset_merge_factory(merge_transform)
    if not callable(objectives):
        objectives = objectives_factory(objectives)
    if not callable(decision):
        name = decision if isinstance(decision, Text) else decision["name"]
        unusual = {"lexo", "lexical_comparison", "epsilon_constaints", "epsilon"}
        if name in unusual:
            msg = (
                f"Decision method is unusual for a retrofit agent."
                f"Expected retro_{name} rather than {name}."
            )
            getLogger(__name__).warning(msg)
        decision = decision_factory(decision)
    assert callable(decision)

    return Agent(
        assets=assets,
        region=region,
        search_rules=filter_factory(search_rules),
        housekeeping=housekeeping,
        merge_transform=merge_transform,
        objectives=objectives,
        decision=decision,
        year=year,
        **kwargs,
    )


def create_newcapa_agent(
    capacity: DataArray,
    year: int,
    region: Text,
    interpolation: Text = "linear",
    search_rules="all",
    housekeeping: Union[Text, Mapping, Callable] = "noop",
    merge_transform: Union[Text, Mapping, Callable] = "new",
    quantity: float = 0.3,
    objectives: Union[
        Callable, Text, Mapping, Sequence[Union[Text, Mapping]]
    ] = "fixed_costs",
    decision: Union[Callable, Text, Mapping] = "mean",
    **kwargs,
):
    """Creates newcapa agent from muse primitives."""
    from xarray import zeros_like
    from muse.hooks import housekeeping_factory, asset_merge_factory
    from muse.filters import factory as filter_factory
    from muse.registration import name_variations
    from muse.objectives import factory as objectives_factory
    from muse.decisions import factory as decision_factory

    if "region" in capacity.dims:
        capacity = capacity.sel(region=region)

    existing = capacity.interp(year=year, method=interpolation) > 0
    assert set(existing.dims) == {"asset"}
    years = [capacity.year.min().values, capacity.year.max().values]
    assets = Dataset()
    assets["capacity"] = zeros_like(capacity.sel(asset=existing.values, year=years))

    if isinstance(search_rules, Text):
        search_rules = [search_rules]
    # ensure newcapa agents do not use currently_existing_tech filter, since it would
    # turn off all replacement techs
    variations = set(name_variations("existing")).union(
        name_variations("currently_existing_tech")
    )
    search_rules = [
        "currently_referenced_tech" if name in variations else name
        for name in search_rules
    ]
    if len(search_rules) == 0 or search_rules[-1] != "compress":
        search_rules.append("compress")

    if not callable(housekeeping):
        housekeeping = housekeeping_factory(housekeeping)
    if not callable(merge_transform):
        merge_transform = asset_merge_factory(merge_transform)
    if not callable(objectives):
        objectives = objectives_factory(objectives)
    if not callable(decision):
        decision = decision_factory(decision)

    result = Agent(
        assets=assets,
        region=region,
        search_rules=filter_factory(search_rules),
        housekeeping=housekeeping,
        merge_transform=merge_transform,
        objectives=objectives,
        decision=decision,
        year=year,
        **kwargs,
    )
    result.quantity = quantity  # type: ignore
    return result


def create_agent(agent_type: Text, **kwargs) -> Agent:
    method = {"retrofit": create_retrofit_agent, "newcapa": create_newcapa_agent}[
        agent_type.lower()
    ]
    return method(**kwargs)  # type: ignore


def factory(
    existing_capacity_path: Optional[Union[Path, Text]] = None,
    agent_parameters_path: Optional[Union[Path, Text]] = None,
    technodata_path: Optional[Union[Path, Text]] = None,
    sector: Optional[Text] = None,
    sectors_directory: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
    baseyear: int = 2010,
) -> List[Agent]:
    """Reads list of agents from standard MUSE input files."""
    from logging import getLogger
    from textwrap import dedent
    from copy import deepcopy
    from muse.readers import (
        read_technodictionary,
        read_initial_capacity,
        read_csv_agent_parameters,
    )
    from muse.readers.csv import find_sectors_file

    if sector is None:
        assert existing_capacity_path is not None
        assert agent_parameters_path is not None
        assert technodata_path is not None

    if existing_capacity_path is None:
        existing_capacity_path = find_sectors_file(
            "Existing%s.csv" % sector, sector, sectors_directory
        )
    if agent_parameters_path is None:
        agent_parameters_path = find_sectors_file(
            "BuildingAgent%s.csv" % sector, sector, sectors_directory
        )
    if technodata_path is None:
        technodata_path = find_sectors_file(
            "technodata%s.csv" % sector, sector, sectors_directory
        )

    params = read_csv_agent_parameters(agent_parameters_path)
    techno = read_technodictionary(technodata_path)
    capa = read_initial_capacity(existing_capacity_path)

    result = []
    for param in params:
        if param["agent_type"] == "retrofit":
            param["technologies"] = techno.sel(region=param["region"])
        param["category"] = param["agent_type"]
        param["capacity"] = deepcopy(capa.sel(region=param["region"]))
        param["year"] = baseyear
        result.append(create_agent(**param))

    nregs = len({u.region for u in result})
    types = [u.name for u in result]
    msg = dedent(
        """\
        Read agents for sector {name} from:
            - agent parameter file {para}
            - technologies data file {tech}
            - initial capacity file {ini}

        Found {n} agents across {nregs} regions, with:
        """.format(
            n=len(result),
            name=sector,
            para=agent_parameters_path,
            tech=technodata_path,
            ini=existing_capacity_path,
            nregs=nregs,
        )
    )
    for t in set(types):
        n = types.count(t)
        msg += "    - {n} {t} agent{plural}\n".format(
            n=n, t=t, plural="" if n == 1 else "s"
        )
    getLogger(__name__).info(msg)
    return result


def agents_factory(
    params_or_path: Union[Text, Path, List],
    capacity: Union[DataArray, Text, Path],
    technologies: Dataset,
    regions: Optional[Sequence[Text]] = None,
    year: Optional[int] = None,
    **kwargs,
) -> List[Agent]:
    """Creates a list of agents for the chosen sector."""
    from logging import getLogger
    from copy import deepcopy
    from muse.readers import read_initial_capacity, read_csv_agent_parameters

    if isinstance(params_or_path, (Text, Path)):
        params = read_csv_agent_parameters(params_or_path)
    else:
        params = params_or_path
    if isinstance(capacity, (Text, Path)):
        capacity = read_initial_capacity(capacity)
    assert isinstance(capacity, DataArray)
    if year is None:
        year = int(capacity.year.min())

    result = []
    for param in params:
        if regions is not None and param["region"] not in regions:
            continue
        if param["agent_type"] == "retrofit":
            param["technologies"] = technologies.sel(region=param["region"])
        param["category"] = param["agent_type"]

        # We deepcopy the capacity  as it changes every iteration and needs to be
        # a separate object
        param["capacity"] = deepcopy(capacity.sel(region=param["region"]))
        param["year"] = year
        param.update(kwargs)
        result.append(create_agent(**param))

    nregs = len({u.region for u in result})
    types = [u.name for u in result]
    msg = f"Found {len(result)} agents across {nregs} regions, with:"
    for t in set(types):
        n = types.count(t)
        msg += "    - {n} {t} agent{plural}\n".format(
            n=n, t=t, plural="" if n == 1 else "s"
        )
    getLogger(__name__).info(msg)
    return result
