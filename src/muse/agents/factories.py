"""Holds all building agents."""
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Text, Union

import xarray as xr

from muse.agents.agent import Agent, InvestingAgent
from muse.defaults import DEFAULT_SECTORS_DIRECTORY


def create_retrofit_agent(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
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
    investment: Union[Callable, Text, Mapping] = "adhoc",
    **kwargs,
):
    """Creates retrofit agent from muse primitives."""
    from logging import getLogger
    from muse.filters import factory as filter_factory
    from muse.hooks import housekeeping_factory, asset_merge_factory
    from muse.objectives import factory as objectives_factory
    from muse.decisions import factory as decision_factory
    from muse.investments import factory as investment_factory

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
    assets = xr.Dataset({"capacity": (capacity * shares).sel(asset=techs).copy()})

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
    if not callable(investment):
        investment = investment_factory(investment)

    return InvestingAgent(
        assets=assets,
        region=region,
        search_rules=filter_factory(search_rules),
        housekeeping=housekeeping,
        merge_transform=merge_transform,
        objectives=objectives,
        decision=decision,
        investment=investment,
        year=year,
        **kwargs,
    )


def create_newcapa_agent(
    capacity: xr.DataArray,
    year: int,
    region: Text,
    search_rules="all",
    interpolation: Text = "linear",
    merge_transform: Union[Text, Mapping, Callable] = "new",
    quantity: float = 0.3,
    objectives: Union[
        Callable, Text, Mapping, Sequence[Union[Text, Mapping]]
    ] = "fixed_costs",
    decision: Union[Callable, Text, Mapping] = "mean",
    investment: Union[Callable, Text, Mapping] = "adhoc",
    housekeeping: Union[Text, Mapping, Callable] = "noop",
    **kwargs,
):
    """Creates newcapa agent from muse primitives."""
    from muse.hooks import housekeeping_factory, asset_merge_factory
    from muse.filters import factory as filter_factory
    from muse.registration import name_variations
    from muse.objectives import factory as objectives_factory
    from muse.decisions import factory as decision_factory
    from muse.investments import factory as investment_factory

    if "region" in capacity.dims:
        capacity = capacity.sel(region=region)

    existing = capacity.interp(year=year, method=interpolation) > 0
    assert set(existing.dims) == {"asset"}
    years = [capacity.year.min().values, capacity.year.max().values]
    assets = xr.Dataset()
    assets["capacity"] = xr.zeros_like(capacity.sel(asset=existing.values, year=years))

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
    if not callable(investment):
        investment = investment_factory(investment)

    result = InvestingAgent(
        assets=assets,
        region=region,
        search_rules=filter_factory(search_rules),
        housekeeping=housekeeping,
        merge_transform=merge_transform,
        objectives=objectives,
        decision=decision,
        investment=investment,
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
    capacity: Union[xr.DataArray, Text, Path],
    technologies: xr.Dataset,
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
    assert isinstance(capacity, xr.DataArray)
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
