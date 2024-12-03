"""Holds all building agents."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Union

import xarray as xr

from muse.agents.agent import Agent, InvestingAgent
from muse.errors import AgentShareNotDefined, TechnologyNotDefined


def create_standard_agent(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    year: int,
    region: str,
    share: Optional[str] = None,
    interpolation: str = "linear",
    **kwargs,
):
    """Creates standard (noninvesting) agent from muse primitives."""
    from muse.filters import factory as filter_factory

    if share is not None:
        capacity = _shared_capacity(
            technologies, capacity, region, share, year, interpolation=interpolation
        )
    else:
        existing = capacity.interp(year=year, method=interpolation) > 0
        existing = existing.any([u for u in existing.dims if u != "asset"])
        years = [capacity.year.min().values, capacity.year.max().values]
        capacity = xr.zeros_like(capacity.sel(asset=existing.values, year=years))
    assets = xr.Dataset(dict(capacity=capacity))
    kwargs = _standardize_inputs(**kwargs)

    return Agent(
        assets=assets,
        region=region,
        search_rules=filter_factory(kwargs.pop("search_rules", None)),
        year=year,
        **kwargs,
    )


def create_retrofit_agent(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    share: str,
    year: int,
    region: str,
    interpolation: str = "linear",
    decision: Union[Callable, str, Mapping] = "mean",
    **kwargs,
):
    """Creates retrofit agent from muse primitives."""
    from logging import getLogger

    from muse.filters import factory as filter_factory

    if not callable(decision):
        name = decision if isinstance(decision, str) else decision["name"]
        unusual = {"lexo", "lexical_comparison", "epsilon_constaints", "epsilon"}
        if name in unusual:
            msg = (
                f"Decision method is unusual for a retrofit agent."
                f"Expected retro_{name} rather than {name}."
            )
            getLogger(__name__).warning(msg)

    assets = _shared_capacity(
        technologies, capacity, region, share, year, interpolation=interpolation
    )

    kwargs = _standardize_investing_inputs(decision=decision, **kwargs)

    search_rules = kwargs.pop("search_rules")

    if len(search_rules) < 2 or search_rules[-2] != "with_asset_technology":
        search_rules.insert(-1, "with_asset_technology")

    return InvestingAgent(
        assets=xr.Dataset(dict(capacity=assets)),
        region=region,
        search_rules=filter_factory(search_rules),
        year=year,
        **kwargs,
    )


def create_newcapa_agent(
    capacity: xr.DataArray,
    year: int,
    region: str,
    share: str,
    search_rules: Union[str, Sequence[str]] = "all",
    interpolation: str = "linear",
    merge_transform: Union[str, Mapping, Callable] = "new",
    quantity: float = 0.3,
    housekeeping: Union[str, Mapping, Callable] = "noop",
    retrofit_present: bool = True,
    **kwargs,
):
    """Creates newcapa agent from muse primitives.

    If there are no retrofit agents present in the sector, then the newcapa agent need
    to be initialised with the initial capacity of the sector.
    """
    from muse.filters import factory as filter_factory
    from muse.registration import name_variations

    if "region" in capacity.dims:
        capacity = capacity.sel(region=region)

    existing = capacity.interp(year=year, method=interpolation) > 0
    assert set(existing.dims) == {"asset"}
    years = [capacity.year.min().values, capacity.year.max().values]

    assets = xr.Dataset()
    if retrofit_present:
        assets["capacity"] = xr.zeros_like(
            capacity.sel(asset=existing.values, year=years)
        )
    else:
        technologies = kwargs["technologies"]
        assets["capacity"] = _shared_capacity(
            technologies, capacity, region, share, year, interpolation=interpolation
        )
        merge_transform = "merge"

    kwargs = _standardize_investing_inputs(
        search_rules=search_rules,
        housekeeping=housekeeping,
        merge_transform=merge_transform,
        **kwargs,
    )

    # ensure newcapa agents do not use currently_existing_tech filter, since it would
    # turn off all replacement techs
    variations = set(name_variations("existing")).union(
        name_variations("currently_existing_tech")
    )
    search_rules = [
        "currently_referenced_tech" if name in variations else name
        for name in kwargs.pop("search_rules")
    ]

    if not retrofit_present:
        if "with_asset_technology" not in search_rules:
            search_rules.insert(-1, "with_asset_technology")

    result = InvestingAgent(
        assets=assets,
        region=region,
        search_rules=filter_factory(search_rules),
        year=year,
        **kwargs,
    )
    result.quantity = quantity  # type: ignore

    return result


def create_agent(agent_type: str, **kwargs) -> Agent:
    method = {
        "retrofit": create_retrofit_agent,
        "newcapa": create_newcapa_agent,
        "agent": create_standard_agent,
        "default": create_standard_agent,
        "standard": create_standard_agent,
    }[agent_type.lower()]
    return method(**kwargs)  # type: ignore


def agents_factory(
    params_or_path: Union[str, Path, list],
    capacity: Union[xr.DataArray, str, Path],
    technologies: xr.Dataset,
    regions: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    **kwargs,
) -> list[Agent]:
    """Creates a list of agents for the chosen sector."""
    from copy import deepcopy
    from logging import getLogger

    from muse.readers import read_csv_agent_parameters, read_initial_assets

    if isinstance(params_or_path, (str, Path)):
        params = read_csv_agent_parameters(params_or_path)
    else:
        params = params_or_path
    if isinstance(capacity, (str, Path)):
        capacity = read_initial_assets(capacity)
    assert isinstance(capacity, xr.DataArray)
    if year is None:
        year = int(capacity.year.min())

    if regions and "region" in capacity.dims:
        capacity = capacity.sel(region=regions)
    if regions and "dst_region" in capacity.dims:
        capacity = capacity.sel(dst_region=regions)
        if capacity.dst_region.size == 1:
            capacity = capacity.squeeze("dst_region", drop=True)
    result = []

    retrofit_present = False
    for param in params:
        retrofit_present = retrofit_present or param["agent_type"] == "retrofit"

    for param in params:
        if regions is not None and param["region"] not in regions:
            continue
        param["technologies"] = technologies.sel(region=param["region"])
        param["category"] = param["agent_type"]

        # We deepcopy the capacity  as it changes every iteration and needs to be
        # a separate object
        param["capacity"] = deepcopy(capacity.sel(region=param["region"]))
        param["year"] = year
        param.update(kwargs)
        result.append(create_agent(**param, retrofit_present=retrofit_present))

    nregs = len({u.region for u in result})
    types = [u.name for u in result]
    msg = f"Found {len(result)} agents across {nregs} regions" + (
        "," if len(result) == 0 else ", with:\n"
    )
    for t in set(types):
        n = types.count(t)
        msg += "    - {n} {t} agent{plural}\n".format(
            n=n, t=t, plural="" if n == 1 else "s"
        )
    getLogger(__name__).info(msg)
    return result


def _shared_capacity(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    region: str,
    share: str,
    year: int,
    interpolation: str = "linear",
) -> xr.DataArray:
    if "region" in capacity.dims:
        capacity = capacity.sel(region=region)
    if "region" in technologies.dims:
        technologies = technologies.sel(region=region)

    try:
        shares = technologies[share]
    except KeyError:
        raise AgentShareNotDefined

    try:
        shares = shares.sel(technology=capacity.technology)
    except KeyError:
        raise TechnologyNotDefined

    if "region" in shares.dims:
        shares = shares.sel(region=region)
    if "year" in shares.dims:
        shares = shares.interp({"year": year}, method=interpolation)

    existing = capacity.interp({"year": year}, method=interpolation)

    techs = (existing > 0) & (shares > 0)
    techs = techs.any([u for u in techs.dims if u != "asset"])
    if not any(techs):
        return (capacity * shares).copy()
    return (capacity * shares).sel(asset=techs.values).copy()


def _standardize_inputs(
    housekeeping: Union[str, Mapping, Callable] = "clean",
    merge_transform: Union[str, Mapping, Callable] = "merge",
    objectives: Union[
        Callable, str, Mapping, Sequence[Union[str, Mapping]]
    ] = "fixed_costs",
    decision: Union[Callable, str, Mapping] = "mean",
    **kwargs,
):
    from muse.decisions import factory as decision_factory
    from muse.hooks import asset_merge_factory, housekeeping_factory
    from muse.objectives import factory as objectives_factory

    if not callable(housekeeping):
        housekeeping = housekeeping_factory(housekeeping)
    if not callable(merge_transform):
        merge_transform = asset_merge_factory(merge_transform)
    if not callable(objectives):
        objectives = objectives_factory(objectives)
    if not callable(decision):
        decision = decision_factory(decision)

    kwargs["housekeeping"] = housekeeping
    kwargs["merge_transform"] = merge_transform
    kwargs["objectives"] = objectives
    kwargs["decision"] = decision
    return kwargs


def _standardize_investing_inputs(
    search_rules: Optional[Union[str, Sequence[str]]] = None,
    investment: Union[Callable, str, Mapping] = "scipy",
    constraints: Optional[
        Union[Callable, str, Mapping, Sequence[Union[str, Mapping]]]
    ] = None,
    **kwargs,
) -> dict[str, Any]:
    from muse.constraints import factory as constraints_factory
    from muse.investments import factory as investment_factory

    kwargs = _standardize_inputs(**kwargs)
    if search_rules is None:
        search_rules = list()
    if isinstance(search_rules, str):
        search_rules = [u.strip() for u in search_rules.split("->")]
    search_rules = list(search_rules)
    if len(search_rules) == 0 or search_rules[-1] != "compress":
        search_rules.append("compress")
    kwargs["search_rules"] = search_rules
    if not callable(investment):
        kwargs["investment"] = investment_factory(investment)
    if not callable(constraints):
        kwargs["constraints"] = constraints_factory(constraints)
    return kwargs
