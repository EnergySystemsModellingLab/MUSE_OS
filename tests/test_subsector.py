from copy import deepcopy

import xarray as xr
from pytest import fixture

from muse import constraints as cs
from muse import demand_share as ds
from muse import examples
from muse import investments as iv
from muse.agents.factories import create_agent
from muse.readers import read_agent_parameters, read_initial_capacity
from muse.readers.toml import read_settings
from muse.sectors.subsector import Subsector, aggregate_enduses


@fixture
def model() -> str:
    return "medium"


@fixture
def technologies(model) -> xr.Dataset:
    return examples.sector("residential", model=model).technologies


@fixture
def market(model) -> xr.Dataset:
    return examples.residential_market(model)


@fixture
def base_market(technologies, market):
    """Common market setup used across tests."""
    return market.sel(
        commodity=technologies.commodity, region=technologies.region
    ).interp(year=[2020, 2025])


@fixture
def agent_params(model, tmp_path, technologies):
    """Common agent parameters setup."""
    examples.copy_model(model, tmp_path)
    path = tmp_path / "model" / "Agents.csv"
    params = read_agent_parameters(path)
    capa = read_initial_capacity(path.with_name("residential") / "ExistingCapacity.csv")

    for param in params:
        param.update(
            {
                "capacity": deepcopy(capa.sel(region=param["region"]))
                if param["agent_type"] == "retrofit"
                else xr.zeros_like(capa.sel(region=param["region"])),
                "agent_type": "default",
                "category": "trade",
                "year": 2020,
                "search_rules": "from_assets -> compress -> reduce_assets",
                "objectives": "ALCOE",
                "decision": {"name": "mean", "parameters": ("ALCOE", False, 1)},
            }
        )
        param.pop("quantity", None)
        param.pop("share", None)

    return params


def test_subsector_investing_aggregation():
    model_list = ["default", "medium"]
    sector_list = ["residential", "power", "gas"]

    for model in model_list:
        mca = examples.model(model, test=True)
        for sname in sector_list:
            agents = list(examples.sector(sname, model).agents)
            sector = next(sector for sector in mca.sectors if sector.name == sname)
            technologies = sector.technologies
            commodities = aggregate_enduses(technologies)
            market = mca.market.sel(
                commodity=technologies.commodity, region=technologies.region
            ).interp(year=[2020, 2025])

            initial_agents = deepcopy(agents)
            subsector = Subsector(
                agents,
                commodities,
                demand_share=ds.factory("standard_demand"),
                constraints=cs.factory("demand"),
                investment=iv.factory("scipy"),
            )

            assert {agent.year for agent in agents} == {int(market.year.min())}
            subsector.aggregate_lp(technologies.sel(year=2020), market)
            assert {agent.year for agent in agents} == {int(market.year.min() + 5)}

            for initial, final in zip(initial_agents, agents):
                assert initial.assets.sum() != final.assets.sum()


def test_subsector_noninvesting_aggregation(base_market, agent_params, technologies):
    """Test non-investing aggregation with default agents."""
    agents = [
        create_agent(technologies=technologies, **param) for param in agent_params
    ]
    commodities = aggregate_enduses(technologies)

    subsector = Subsector(
        agents,
        commodities,
        demand_share=ds.factory("unmet_forecasted_demand"),
        constraints=cs.factory("demand"),
        investment=iv.factory("scipy"),
    )

    assert all(agent.year == 2020 for agent in agents)
    subsector.aggregate_lp(technologies.sel(year=2020), base_market)


def test_factory_smoke_test(model, technologies, tmp_path):
    examples.copy_model(model, tmp_path)
    settings = read_settings(tmp_path / "model" / "settings.toml")

    subsector = Subsector.factory(
        settings.sectors.residential.subsectors.all, technologies
    )

    assert isinstance(subsector, Subsector)
    assert len(subsector.agents) == 1
