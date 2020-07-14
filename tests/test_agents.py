"""Test buildings agents."""
from pytest import fixture


@fixture
def assets():
    """Creates assets with believable capacity profile."""
    from numpy.random import choice, randint
    from numpy import ones
    from xarray import Dataset  # noqa
    from muse.utilities import avoid_repetitions

    result = Dataset()
    result["year"] = "year", range(2010, 2031)
    assets = set((randint(2010, 2020), choice(list("technology"))) for i in range(30))
    assets = list(assets)
    result["installed"] = "asset", [u[0] for u in assets]
    result["technology"] = "asset", [u[1] for u in assets]
    shape = len(result.year), len(result.asset)
    result["capacity"] = ("year", "asset"), ones(shape, dtype="int")
    result["maxyears"] = "asset", randint(2010, 2030, len(result.asset))
    result["capa"] = "asset", randint(1, 100, len(result.asset))
    result["capacity"] = result.capacity.where(result.year <= result.maxyears, 0)
    result["capacity"] *= result.capa
    result = result.drop_vars(("maxyears", "capa"))
    result = result.set_coords(("installed", "technology"))
    return result.sel(year=avoid_repetitions(result.capacity))


def test_create_retrofit(agent_args, technologies, stock):
    from muse import create_agent, Agent

    agent_args["share"] = "agent_share_zero"
    agent = create_agent(
        agent_type="Retrofit",
        technologies=technologies,
        capacity=stock.capacity,
        year=2010,
        **agent_args,
    )
    assert isinstance(agent, Agent)
    assert len(agent.assets.capacity) == 0
    assert "asset" in agent.assets.dims and len(agent.assets.asset) == 0
    assert "year" in agent.assets.dims or len(agent.assets.year) > 1
    assert "region" not in agent.assets.dims
    assert "commodity" not in agent.assets.dims

    agent_args["share"] = "agent_share"
    agent = create_agent(
        agent_type="Retrofit",
        technologies=technologies,
        capacity=stock.capacity,
        year=2010,
        **agent_args,
    )
    assert isinstance(agent, Agent)
    assert "asset" in agent.assets.dims
    assert len(agent.assets.capacity) != 0


def test_run_retro_agent(retro_agent, technologies, agent_market, demand_share):
    # make sure capacity limits are no reached
    technologies.total_capacity_limit[:] = retro_agent.assets.capacity.sum() * 100
    technologies.max_capacity_addition[:] = retro_agent.assets.capacity.sum() * 100
    technologies.max_capacity_growth[:] = retro_agent.assets.capacity.sum() * 100

    retro_agent.next(technologies, agent_market, demand_share)


def test_merge_assets(assets):
    from muse.hooks import merge_assets
    from muse.utilities import coords_to_multiindex, avoid_repetitions

    assets = assets.isel(asset=range(5))
    assets = assets.sel(year=avoid_repetitions(assets.capacity))

    n = len(assets.asset)
    current = assets.sel(asset=range(n - 2))
    current = current.sel(year=avoid_repetitions(current.capacity))

    new = assets.sel(asset=range(n - 2, n))
    new = new.sel(year=avoid_repetitions(new.capacity))

    actual = merge_assets(current, new)

    multi_assets = coords_to_multiindex(assets)
    multi_actual = coords_to_multiindex(actual)
    assert (multi_actual == multi_assets).all()


def test_clean_assets(assets):
    from numpy.random import choice
    from muse.utilities import clean_assets

    current_year = choice(range(assets.year.min().values, assets.year.max().values))
    iempties = assets.asset[range(0, len(assets.asset), 3)].asset
    assets.capacity[:] = 1
    assets.capacity.loc[{"asset": iempties, "year": assets.year >= current_year}] = 0

    cleaned = clean_assets(assets, current_year)
    assert (cleaned.year >= current_year).all()

    # fmt: disable
    empties = set(
        zip(
            assets.sel(asset=iempties).technology.values,
            assets.sel(asset=iempties).installed.values,
        )
    )
    # fmt: enable
    cleanies = set(zip(cleaned.technology.values, cleaned.installed.values))
    originals = set(zip(assets.technology.values, assets.installed.values))
    assert empties.isdisjoint(cleanies)
    assert empties.union(cleanies) == originals


def test_initial_assets(tmp_path):
    from muse.examples import copy_model
    from muse.readers.csv import read_initial_assets

    copy_model("default", tmp_path / "default")
    copy_model("trade", tmp_path / "trade")

    def path(x, y):
        return (
            tmp_path / x / "model" / "technodata" / "gas" / f"Existing{y.title()}.csv"
        )

    assets = read_initial_assets(path("default", "capacity"))
    assert set(assets.dims) == {"year", "region", "asset"}

    assets = read_initial_assets(path("trade", "trade"))
    assert set(assets.dims) == {"year", "region", "asset", "dst_region"}
