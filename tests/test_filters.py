import numpy as np
import xarray as xr
from pytest import fixture, mark

from muse.filters import factory, register_filter, register_initializer


@fixture
def search_space(retro_agent, technologies):
    technology = technologies.technology
    asset = technology[technology.isin(retro_agent.assets.technology)]
    coords = {"asset": asset.values, "replacement": technology.values}
    return xr.DataArray(
        np.ones(tuple(len(u) for u in coords.values()), dtype=bool),
        coords=coords,
        dims=coords.keys(),
        name="search_space",
    )


@mark.usefixtures("save_registries")
def test_filter_registering():
    from muse.filters import SEARCH_SPACE_FILTERS

    @register_filter
    def a_filter(retro_agent, search_space: xr.DataArray):
        pass

    assert "a_filter" in SEARCH_SPACE_FILTERS
    assert SEARCH_SPACE_FILTERS["a_filter"] is a_filter

    @register_filter(name="something")
    def b_filter(retro_agent, search_space: xr.DataArray):
        pass

    assert "something" in SEARCH_SPACE_FILTERS
    assert SEARCH_SPACE_FILTERS["something"] is b_filter


@mark.usefixtures("save_registries")
def test_filtering():
    @register_initializer
    def start(*args, **kwargs):
        return list(range(5))

    @register_filter
    def first(retro_agent, search_space, switch=True, data=None):
        if switch:
            return search_space[2:]
        return search_space[:2]

    @register_filter
    def second(retro_agent, search_space, switch=True, data=None):
        return [u for u in search_space if u in data]

    sp = start(None, None, None)
    assert factory(["start", "first"])(None, sp) == sp[2:]
    assert factory(["start", "first"])(None, sp, False) == sp[:2]

    assert factory(["start", "second"])(None, sp, data=(1, 3, 5)) == [1, 3]
    assert factory(["start", "first", "second"])(None, sp, data=(1, 3, 5)) == [3]
    assert factory(["start", "first", "second"])(None, sp, False, (1, 3, 5)) == [1]


def test_same_enduse(retro_agent, technologies, search_space):
    from muse.filters import same_enduse
    from muse.commodities import is_enduse

    result = same_enduse(retro_agent, search_space, technologies)
    enduses = is_enduse(technologies.comm_usage)
    finputs = technologies.sel(
        region=retro_agent.region, year=retro_agent.year, commodity=enduses
    )
    finputs = finputs.fixed_outputs > 0

    expected = search_space.copy()
    for asset in result.asset:
        asset_enduses = finputs.sel(technology=asset)
        asset_enduses = set(asset_enduses.commodity.loc[asset_enduses].values)
        for tech in result.replacement:
            tech_enduses = finputs.sel(technology=tech)
            tech_enduses = set(tech_enduses.commodity.loc[tech_enduses].values)
            expected.loc[
                {"replacement": tech, "asset": asset}
            ] = asset_enduses.issubset(tech_enduses)

    assert sorted(result.dims) == sorted(search_space.dims)
    assert (result == expected).all()


def test_similar_tech(retro_agent, search_space, technologies):
    from muse.filters import similar_technology

    actual = similar_technology(retro_agent, search_space, technologies)
    assert sorted(actual.dims) == sorted(search_space.dims)

    tech_type = technologies.tech_type
    for tech in actual.replacement:
        for asset in actual.asset:
            expected = tech_type.loc[tech] == tech_type.loc[asset]
            assert expected == actual.sel(replacement=tech, asset=asset)


def test_similar_fuels(retro_agent, search_space, technologies):
    from muse.filters import same_fuels

    actual = same_fuels(retro_agent, search_space, technologies)
    assert sorted(actual.dims) == sorted(search_space.dims)

    fuel_type = technologies.fuel
    for tech in actual.replacement:
        for asset in actual.asset:
            expected = fuel_type.loc[tech] == fuel_type.loc[asset]
            assert expected == actual.sel(replacement=tech, asset=asset)


def test_currently_existing(retro_agent, search_space, technologies, agent_market, rng):
    from muse.filters import currently_existing_tech

    agent_market.capacity[:] = 0
    actual = currently_existing_tech(
        retro_agent, search_space, technologies, agent_market
    )
    assert sorted(actual.dims) == sorted(search_space.dims)
    assert not actual.any()

    agent_market.capacity[:] = 1
    actual = currently_existing_tech(
        retro_agent, search_space, technologies, agent_market
    )
    assert sorted(actual.dims) == sorted(search_space.dims)
    in_market = search_space.replacement.isin(agent_market.technology)
    assert not actual.sel(replacement=~in_market).any()
    assert actual.sel(replacement=in_market).all()

    techs = rng.choice(
        list(set(agent_market.technology.values)),
        1 + rng.choice(range(len(set(agent_market.technology.values)))),
        replace=False,
    )
    agent_market.capacity[:] = 0
    agent_market.capacity.loc[{"technology": agent_market.technology.isin(techs)}] = 1
    actual = currently_existing_tech(
        retro_agent, search_space, technologies, agent_market
    )
    assert sorted(actual.dims) == sorted(search_space.dims)
    assert not actual.sel(replacement=~in_market).any()
    current_cap = agent_market.capacity.sel(
        year=retro_agent.year, region=retro_agent.region
    ).rename(technology="replacement")
    expected = (current_cap > retro_agent.tolerance).rename("expected")
    assert (actual.sel(replacement=in_market) == expected).all()


@mark.xfail
def test_maturity(retro_agent, search_space, technologies, agent_market):
    from muse.filters import maturity
    from muse.commodities import is_enduse

    enduses = is_enduse(technologies.comm_usage)
    outputs = technologies.fixed_outputs.sel(commodity=enduses, region="USA", year=2010)
    capacity = agent_market.capacity.sel(year=2010, region="USA")
    production = (outputs * capacity).sum("technology")

    # nothing should be true
    retro_agent.maturity_threshhold = 1.1 * (capacity / production).max()
    actual = maturity(retro_agent, search_space, technologies, agent_market)
    assert sorted(actual.dims) == sorted(search_space.dims)
    assert (not actual).all()

    # some should be true - do it with a fully on search space for  simplicity
    retro_agent.maturity_threshhold = 0.8 * (capacity / production).max()
    actual = maturity(
        search_space == retro_agent, search_space, technologies, agent_market
    )
    assert sorted(actual.dims) == sorted(search_space.dims)
    assert actual.any()
    # all should be true
    retro_agent.maturity_threshhold = 0.8 * (capacity / production).min()
    actual = maturity(retro_agent, search_space, technologies, agent_market)
    assert (actual == search_space).any()


def test_init_from_tech(demand_share, technologies, agent_market):
    from collections import namedtuple
    from muse.filters import initialize_from_technologies

    agent = namedtuple("DummyAgent", ["tolerance"])(tolerance=1e-8)

    space = initialize_from_technologies(agent, demand_share, technologies)
    assert set(space.dims) == {"asset", "replacement"}
    assert (space.asset.values == demand_share.asset.values).all()
    assert (space.replacement.values == technologies.technology.values).all()
    assert space.all()


def test_init_from_asset(technologies, rng):
    from collections import namedtuple
    from muse.filters import initialize_from_assets

    technology = rng.choice(technologies.technology, 5)
    installed = rng.choice((2020, 2025), len(technology))
    year = np.arange(2020, 2040, 5)
    capacity = xr.DataArray(
        rng.choice([0, 0, 1, 10], (len(technology), len(year))),
        coords={
            "technology": ("asset", technology),
            "installed": ("asset", installed),
            "region": "USA",
            "year": ("year", year),
        },
        dims=("asset", "year"),
    )
    agent = namedtuple("DummyAgent", ["assets"])(xr.Dataset(dict(capacity=capacity)))

    space = initialize_from_assets(agent, demand=None, technologies=technologies)
    assert set(space.dims) == {"asset", "replacement"}
    assert space.replacement.isin(technologies.technology).all()
    assert technologies.technology.isin(space.replacement).all()
    assert set(space.asset.asset.values) == set(capacity.technology.values)


def test_init_from_asset_no_assets(technologies, rng):
    from collections import namedtuple
    from muse.filters import initialize_from_assets

    agent = namedtuple("DummyAgent", ["assets"])(
        xr.Dataset(dict(capacity=xr.DataArray(0)))
    )

    space = initialize_from_assets(agent, demand=None, technologies=technologies)
    assert set(space.dims) == {"replacement"}
    assert space.replacement.isin(technologies.technology).all()
    assert technologies.technology.isin(space.replacement).all()
