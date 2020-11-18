"""Test buildings agents."""
<<<<<<< HEAD
from pytest import fixture
=======
from pytest import approx, fixture, mark
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


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
<<<<<<< HEAD
    from muse.agents.factories import create_agent
    from muse.agents.agent import Agent
=======
    from muse import create_agent, Agent
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    agent_args["share"] = "agent_share_zero"
    agent = create_agent(
        agent_type="Retrofit",
        technologies=technologies,
        capacity=stock.capacity,
        year=2010,
<<<<<<< HEAD
        **agent_args,
=======
        **agent_args
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
        **agent_args,
=======
        **agent_args
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    )
    assert isinstance(agent, Agent)
    assert "asset" in agent.assets.dims
    assert len(agent.assets.capacity) != 0


<<<<<<< HEAD
def test_run_retro_agent(retro_agent, technologies, agent_market, demand_share):
    # make sure capacity limits are no reached
    technologies.total_capacity_limit[:] = retro_agent.assets.capacity.sum() * 100
    technologies.max_capacity_addition[:] = retro_agent.assets.capacity.sum() * 100
    technologies.max_capacity_growth[:] = retro_agent.assets.capacity.sum() * 100
=======
@mark.parametrize("time_period", [1, 5, 0.01])
@mark.parametrize("scale_growth", [1, 0.01])
@mark.parametrize("scale_add", [1, 0.01])
@mark.parametrize("scale_limit", [1, 0.01])
def test_max_capacity_expansion(
    retro_agent, technologies, time_period, scale_growth, scale_limit, scale_add
):
    # disable/enable constraints by adding a factor
    technologies.max_capacity_growth[:] *= scale_growth
    technologies.max_capacity_growth[:] *= scale_limit
    technologies.total_capacity_limit[:] *= scale_add

    max_cap = retro_agent.max_capacity_expansion(technologies, time_period=time_period)

    techs = technologies.sel(year=retro_agent.year, region=retro_agent.region)
    assets = (
        retro_agent.assets.groupby("technology")
        .sum("asset")
        .reindex_like(techs.technology)
        .fillna(0)
    )

    current_cap = assets.capacity.sel(year=retro_agent.year)
    forecast_cap = assets.capacity.interp(
        year=retro_agent.forecast_year, method="linear"
    )

    tot_cap_lim = techs.total_capacity_limit
    max_cap_gro = techs.max_capacity_growth
    max_cap_add = techs.max_capacity_addition

    left = tot_cap_lim + 1e-12 >= forecast_cap + max_cap
    right = tot_cap_lim <= forecast_cap + 1e-12
    assert (left | right).all()
    decom = current_cap - forecast_cap
    left = time_period * max_cap_gro * current_cap + decom + 1e-12
    assert (left >= max_cap).where(current_cap > 0, True).all()
    assert (time_period * max_cap_add + 1e-12 >= max_cap).all()
    is_new_tech = ~technologies.technology.isin(retro_agent.assets.technology)
    assert max_cap.sel(technology=is_new_tech).values != approx(0)


# Some random numbers dont result in viable retro agent
def test_run_retro_agent(retro_agent, technologies, agent_market, demand_share):
    from copy import deepcopy

    assets = deepcopy(retro_agent.assets.capacity)

    # make sure capacity limit is not reached
    capa_year = (
        assets.interp(year=retro_agent.forecast_year, method="linear")
        .groupby("technology")
        .sum("asset")
    )
    tot_lim = technologies.total_capacity_limit.sel(
        technology=list(set(assets.technology.values))
    )
    technologies.total_capacity_limit.loc[
        {"technology": list(set(assets.technology.values))}
    ] = tot_lim.where(
        tot_lim > capa_year, (capa_year + 10).sel(technology=tot_lim.technology)
    )
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

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
<<<<<<< HEAD


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
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
