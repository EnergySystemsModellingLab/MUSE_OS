from typing import Sequence, Text

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


def test_subsector_investing_aggregation(market, model, technologies):
    from copy import deepcopy
    from muse import examples
    from muse.sectors.subsector import Subsector, aggregate_enduses

    agents = examples.agents("residential", model)
    commodities = aggregate_enduses((agent.assets for agent in agents), technologies)
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


def test_subsector_noninvesting_aggregation(market, model, technologies, tmp_path):
    """Create some default agents and run subsector.

    Mostly a smoke test to check the returns look about right, with the right type and
    containing "agent" dimensions.
    """
    from copy import deepcopy
    from muse import examples, readers
    from muse.agents.factories import create_agent
    from muse.sectors.subsector import Subsector, aggregate_enduses
    from muse import demand_share as ds, constraints as cs

    examples.copy_model(model, tmp_path)
    path = tmp_path / "model" / "technodata" / "Agents.csv"
    params = readers.read_csv_agent_parameters(path)
    capa = readers.read_initial_capacity(path.with_name("residential") / "Existing.csv")
    for param in params:
        if param["agent_type"] == "retrofit":
            param["capacity"] = deepcopy(capa.sel(region=param["region"]))
        else:
            param["capacity"] = xr.zeros_like(capa.sel(region=param["region"]))
        param["agent_type"] = "default"
        param["category"] = "trade"
        param["year"] = 2020
        param["search_rules"] = "from_assets -> compress -> reduce_assets"
        param.pop("quantity")

    agents = [create_agent(technologies=technologies, **param) for param in params]
    commodities = aggregate_enduses((agent.assets for agent in agents), technologies)

    subsector = Subsector(
        agents,
        commodities,
        demand_share=ds.factory("market_demand"),
        constraints=cs.factory("demand"),
    )

    market = market.sel(
        commodity=technologies.commodity, region=technologies.region
    ).interp(year=[2020, 2025])
    assert all(agent.year == 2020 for agent in agents)
    result = subsector.aggregate_lp(technologies, market)
    assert result is not None
    assert len(result) == 2

    lpcosts, lpconstraints = result
    assert isinstance(lpcosts, xr.DataArray)
    assert "agent" in lpcosts.coords
    assert isinstance(lpconstraints, Sequence)
    assert len(lpconstraints) == 1
    assert all((isinstance(u, xr.Dataset) for u in lpconstraints))
    # makes sure agent investment got called
    assert all(agent.year == 2025 for agent in agents)


def test_factory_smoke_test(model, technologies, tmp_path):
    from muse import examples
    from muse.readers import read_settings
    from muse.sectors.subsector import Subsector

    examples.copy_model(model, tmp_path)
    settings = read_settings(tmp_path / "model" / "settings.toml")

    subsector = Subsector.factory(settings.sectors.residential, technologies)

    assert isinstance(subsector, Subsector)
    assert len(subsector.agents) == 2
