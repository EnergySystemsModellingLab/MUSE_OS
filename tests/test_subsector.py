import xarray as xr
from pytest import fixture


@fixture
def model() -> str:
    return "medium"


@fixture
def technologies(model) -> xr.Dataset:
    from muse import examples

    return examples.sector("residential", model=model).technologies


@fixture
def market(model) -> xr.Dataset:
    from muse import examples

    return examples.residential_market(model)


def test_subsector_investing_aggregation():
    from copy import deepcopy

    from muse import examples
    from muse.sectors.subsector import Subsector, aggregate_enduses

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
            subsector = Subsector(agents, commodities)
            initial_agents = deepcopy(agents)
            assert {agent.year for agent in agents} == {int(market.year.min())}
            subsector.aggregate_lp(technologies.sel(year=2020), market)
            assert {agent.year for agent in agents} == {int(market.year.min() + 5)}
            for initial, final in zip(initial_agents, agents):
                assert initial.assets.sum() != final.assets.sum()


def test_subsector_noninvesting_aggregation(market, model, technologies, tmp_path):
    """Create some default agents and run subsector.

    Mostly a smoke test to check the returns look about right, with the right type and
    containing "agent" dimensions.
    """
    from copy import deepcopy

    from muse import constraints as cs
    from muse import demand_share as ds
    from muse import examples, readers
    from muse.agents.factories import create_agent
    from muse.sectors.subsector import Subsector, aggregate_enduses

    examples.copy_model(model, tmp_path)
    path = tmp_path / "model" / "Agents.csv"
    params = readers.read_csv_agent_parameters(path)
    capa = readers.read_initial_assets(
        path.with_name("residential") / "ExistingCapacity.csv"
    )

    for param in params:
        if param["agent_type"] == "retrofit":
            param["capacity"] = deepcopy(capa.sel(region=param["region"]))
        else:
            param["capacity"] = xr.zeros_like(capa.sel(region=param["region"]))

        if "share" in param:
            del param["share"]

        param["agent_type"] = "default"
        param["category"] = "trade"
        param["year"] = 2020
        param["search_rules"] = "from_assets -> compress -> reduce_assets"
        param["objectives"] = "ALCOE"
        param["decision"]["parameters"] = ("ALCOE", False, 1)
        param.pop("quantity")
    agents = [create_agent(technologies=technologies, **param) for param in params]
    commodities = aggregate_enduses(technologies)

    subsector = Subsector(
        agents,
        commodities,
        demand_share=ds.factory("unmet_forecasted_demand"),
        constraints=cs.factory("demand"),
    )

    market = market.sel(
        commodity=technologies.commodity, region=technologies.region
    ).interp(year=[2020, 2025])
    assert all(agent.year == 2020 for agent in agents)
    subsector.aggregate_lp(technologies.sel(year=2020), market)


def test_factory_smoke_test(model, technologies, tmp_path):
    from muse import examples
    from muse.readers.toml import read_settings
    from muse.sectors.subsector import Subsector

    examples.copy_model(model, tmp_path)
    settings = read_settings(tmp_path / "model" / "settings.toml")

    subsector = Subsector.factory(
        settings.sectors.residential.subsectors.all, technologies
    )

    assert isinstance(subsector, Subsector)
    assert len(subsector.agents) == 1
