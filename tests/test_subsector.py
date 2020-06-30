from typing import Text

import xarray as xr
from pytest import fixture


@fixture
def model() -> Text:
    return "medium"


@fixture
def technologies(model) -> xr.Dataset:
    from muse import examples

    return examples.sector("residential", model=model).technologies


@fixture
def market(model) -> xr.Dataset:
    from muse import examples

    return examples.residential_market(model)


@fixture
def rg():
    from numpy.random import default_rng

    return default_rng()


def test_subsector_aggregation(market, model, technologies):
    from copy import deepcopy
    from muse import examples
    from muse.sectors.subsector import Subsector
    from muse.commodities import is_enduse

    agents = examples.agents("residential", model)
    techs = set.union(*(set(agent.assets.technology.values) for agent in agents))
    outputs = technologies.fixed_outputs.sel(
        commodity=is_enduse(technologies.comm_usage), technology=list(techs)
    )
    commodities = outputs.commodity.sel(
        commodity=outputs.any([u for u in outputs.dims if u != "commodity"])
    )
    market = market.sel(
        commodity=technologies.commodity, region=technologies.region
    ).interp(year=[2020, 2025])

    subsector = Subsector(agents, commodities)
    initial_agents = deepcopy(agents)
    assert {agent.year for agent in agents} == {int(market.year.min())}
    assert subsector.aggregate_lp(technologies, market) is None
    assert {agent.year for agent in agents} == {int(market.year.min() + 5)}
    for initial, final in zip(initial_agents, agents):
        assert initial.assets.sum() != final.assets.sum()
