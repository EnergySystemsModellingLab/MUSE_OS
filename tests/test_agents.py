"""Test buildings agents."""

from pytest import fixture, mark


@fixture
def assets():
    """Creates assets with believable capacity profile."""
    from numpy import ones
    from numpy.random import choice, randint
    from xarray import Dataset

    from muse.utilities import avoid_repetitions

    result = Dataset()
    result["year"] = "year", range(2010, 2031)
    assets = set((randint(2010, 2020), choice(list("technology"))) for i in range(30))
    result["installed"] = "asset", [u[0] for u in assets]
    result["technology"] = "asset", [u[1] for u in assets]
    shape = len(result.year), len(result.asset)
    result["capacity"] = ("year", "asset"), ones(shape, dtype="int")
    result["capacity"] *= randint(1, 100, len(result.asset))

    # Generate max years with correct shape for broadcasting
    max_years = randint(2010, 2030, len(result.asset))
    result["capacity"] = result.capacity.where(
        result.year.values[:, None] <= max_years, 0
    )
    result = result.set_coords(("installed", "technology"))
    return result.sel(year=avoid_repetitions(result.capacity))


@fixture
def create_test_agent():
    """Helper fixture to create and test agents with common assertions."""
    from muse.agents.agent import Agent
    from muse.agents.factories import create_agent

    def _create_and_test(
        agent_type, technologies, stock, agent_args, year=2010, **kwargs
    ):
        agent = create_agent(
            agent_type=agent_type,
            technologies=technologies,
            capacity=stock.capacity,
            year=year,
            **{**agent_args, **kwargs},
        )
        assert isinstance(agent, Agent)
        assert "asset" in agent.assets.dims
        assert "year" in agent.assets.dims
        assert "region" not in agent.assets.dims
        assert "commodity" not in agent.assets.dims
        return agent

    return _create_and_test


def test_create_retrofit(agent_args, technologies, stock, create_test_agent):
    # Test with zero share
    agent_args["share"] = "agent_share_zero"
    agent = create_test_agent("Retrofit", technologies, stock, agent_args)
    assert (agent.assets.capacity == 0).all()
    assert len(agent.assets.asset) != 0

    # Test with non-zero share
    agent_args["share"] = "agent_share"
    agent = create_test_agent("Retrofit", technologies, stock, agent_args)
    assert (agent.assets.capacity != 0).any()


def test_create_newcapa(agent_args, technologies, stock, create_test_agent):
    # Test without retrofit
    agent_args.update({"share": "agent_share_zero", "retrofit_present": False})
    agent = create_test_agent("Newcapa", technologies, stock, agent_args)
    assert (agent.assets.capacity == 0).all()
    assert agent.merge_transform.__name__ == "merge"

    # Test with non-zero share
    agent_args["share"] = "agent_share"
    agent = create_test_agent("Newcapa", technologies, stock, agent_args)
    assert (agent.assets.capacity != 0).any()
    assert agent.merge_transform.__name__ == "merge"

    # Test with retrofit present
    agent_args.update({"share": "agent_share", "retrofit_present": True})
    agent = create_test_agent("Newcapa", technologies, stock, agent_args)
    assert (agent.assets.capacity == 0).all()
    assert agent.merge_transform.__name__ == "new"


def test_issue_835_and_842(agent_args, technologies, stock, create_test_agent):
    agent_args["share"] = "agent_share_zero"
    agent = create_test_agent(
        "Retrofit", technologies, stock, agent_args, search_rules="from_techs->compress"
    )
    assert (agent.assets.capacity == 0).all()
    assert len(agent.assets.asset) != 0


@mark.xfail(reason="Retrofit agents will be deprecated.")
def test_run_retro_agent(retro_agent, technologies, agent_market, demand_share):
    capacity_multiplier = retro_agent.assets.capacity.sum() * 100
    for attr in [
        "total_capacity_limit",
        "max_capacity_addition",
        "max_capacity_growth",
    ]:
        setattr(technologies, attr, capacity_multiplier)

    investment_year = int(agent_market.year[1])
    retro_agent.next(
        technologies.sel(year=investment_year).isel(region=0),
        agent_market.isel(region=0),
        demand_share,
    )


def test_merge_assets(assets):
    from muse.hooks import merge_assets
    from muse.utilities import avoid_repetitions, coords_to_multiindex

    assets = assets.isel(asset=range(5))
    assets = assets.sel(year=avoid_repetitions(assets.capacity))

    n = len(assets.asset)
    current = assets.sel(asset=range(n - 2))
    current = current.sel(year=avoid_repetitions(current.capacity))
    new = assets.sel(asset=range(n - 2, n))
    new = new.sel(year=avoid_repetitions(new.capacity))

    actual = merge_assets(current, new)
    assert (coords_to_multiindex(actual) == coords_to_multiindex(assets)).all()


def test_clean_assets(assets):
    from numpy.random import choice

    from muse.utilities import clean_assets

    current_year = choice(range(assets.year.min().values, assets.year.max().values))
    iempties = assets.asset[range(0, len(assets.asset), 3)].asset
    assets.capacity[:] = 1
    assets.capacity.loc[{"asset": iempties, "year": assets.year >= current_year}] = 0

    cleaned = clean_assets(assets, current_year)
    assert (cleaned.year >= current_year).all()

    empties = set(
        zip(
            assets.sel(asset=iempties).technology.values,
            assets.sel(asset=iempties).installed.values,
        )
    )
    cleanies = set(zip(cleaned.technology.values, cleaned.installed.values))
    originals = set(zip(assets.technology.values, assets.installed.values))
    assert empties.isdisjoint(cleanies)
    assert empties.union(cleanies) == originals
