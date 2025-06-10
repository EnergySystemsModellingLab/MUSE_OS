from collections import namedtuple

import numpy as np
import xarray as xr
from pytest import fixture, mark

from muse.commodities import is_enduse, is_fuel
from muse.filters import (
    currently_existing_tech,
    factory,
    initialize_from_assets,
    initialize_from_technologies,
    maturity,
    register_filter,
    register_initializer,
    same_enduse,
    same_fuels,
    similar_technology,
)


# Common test utilities
def assert_tech_comparison(actual, search_space, tech_attr, technologies):
    """Helper for technology comparison tests."""
    assert sorted(actual.dims) == sorted(search_space.dims)
    attr_values = getattr(technologies, tech_attr)
    for tech in actual.replacement:
        for asset in actual.asset:
            expected = attr_values.loc[tech] == attr_values.loc[asset]
            assert expected == actual.sel(replacement=tech, asset=asset)


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


@fixture
def technologies(technologies):
    return technologies.sel(year=2010)


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
    def start(retro_agent, demand, **kwargs):
        return list(range(5))

    @register_filter
    def first(retro_agent, search_space, switch=True, data=None):
        return search_space[2:] if switch else search_space[:2]

    @register_filter
    def second(retro_agent, search_space, switch=True, data=None):
        return [u for u in search_space if u in data]

    sp = start(None, None)
    assert factory(["start", "first"])(None, sp) == sp[2:]
    assert factory(["start", "first"])(None, sp, switch=False) == sp[:2]
    assert factory(["start", "second"])(None, sp, data=(1, 3, 5)) == [1, 3]
    assert factory(["start", "first", "second"])(None, sp, data=(1, 3, 5)) == [3]
    assert factory(["start", "first", "second"])(
        None, sp, switch=False, data=(1, 3, 5)
    ) == [1]


def test_same_enduse(retro_agent, technologies, search_space):
    result = same_enduse(retro_agent, search_space, technologies=technologies)
    enduses = is_enduse(technologies.comm_usage)
    finputs = technologies.sel(region=retro_agent.region, commodity=enduses)
    finputs = finputs.fixed_outputs > 0

    expected = search_space.copy()
    for asset in result.asset:
        asset_enduses = set(
            finputs.sel(technology=asset)
            .commodity.loc[finputs.sel(technology=asset)]
            .values
        )
        for tech in result.replacement:
            tech_enduses = set(
                finputs.sel(technology=tech)
                .commodity.loc[finputs.sel(technology=tech)]
                .values
            )
            expected.loc[{"replacement": tech, "asset": asset}] = (
                asset_enduses.issubset(tech_enduses)
            )

    assert sorted(result.dims) == sorted(search_space.dims)
    assert (result == expected).all()


def test_similar_tech(retro_agent, search_space, technologies):
    actual = similar_technology(retro_agent, search_space, technologies=technologies)
    assert_tech_comparison(actual, search_space, "tech_type", technologies)


def test_similar_fuels(retro_agent, search_space, technologies):
    actual = same_fuels(retro_agent, search_space, technologies=technologies)
    assert sorted(actual.dims) == sorted(search_space.dims)

    # Get the fixed inputs that are fuels
    fuels = is_fuel(technologies.comm_usage)
    finputs = technologies.sel(region=retro_agent.region, commodity=fuels)
    finputs = finputs.fixed_inputs > 0

    expected = search_space.copy()
    for asset in actual.asset:
        asset_fuels = finputs.sel(technology=asset)
        asset_fuels = set(asset_fuels.commodity.loc[asset_fuels].values)
        for tech in actual.replacement:
            tech_fuels = finputs.sel(technology=tech)
            tech_fuels = set(tech_fuels.commodity.loc[tech_fuels].values)
            expected.loc[{"replacement": tech, "asset": asset}] = (
                asset_fuels == tech_fuels
            )

    assert (actual == expected).all()


def test_currently_existing(retro_agent, search_space, technologies, agent_market, rng):
    # Test with zero capacity
    agent_market.capacity[:] = 0
    actual = currently_existing_tech(
        retro_agent, search_space, technologies=technologies, market=agent_market
    )
    assert sorted(actual.dims) == sorted(search_space.dims)
    assert not actual.any()

    # Test with full capacity
    agent_market.capacity[:] = 1
    actual = currently_existing_tech(
        retro_agent, search_space, technologies=technologies, market=agent_market
    )
    in_market = search_space.replacement.isin(agent_market.technology)
    assert not actual.sel(replacement=~in_market).any()
    assert actual.sel(replacement=in_market).all()

    # Test with partial capacity
    techs = rng.choice(
        list(set(agent_market.technology.values)),
        1 + rng.choice(range(len(set(agent_market.technology.values)))),
        replace=False,
    )
    agent_market.capacity[:] = 0
    agent_market.capacity.loc[{"technology": agent_market.technology.isin(techs)}] = 1
    actual = currently_existing_tech(
        retro_agent, search_space, technologies=technologies, market=agent_market
    )

    assert not actual.sel(replacement=~in_market).any()
    current_cap = agent_market.capacity.sel(
        year=retro_agent.year, region=retro_agent.region
    ).rename(technology="replacement")
    expected = (current_cap > retro_agent.tolerance).rename("expected")
    assert (actual.sel(replacement=in_market) == expected).all()


@mark.xfail
def test_maturity(retro_agent, search_space, technologies, agent_market):
    enduses = is_enduse(technologies.comm_usage)
    outputs = technologies.fixed_outputs.sel(commodity=enduses, region="USA", year=2010)
    capacity = agent_market.capacity.sel(year=2010, region="USA")
    production = (outputs * capacity).sum("technology")

    def check_maturity(threshold_factor, expected_result=None):
        retro_agent.maturity_threshold = (
            threshold_factor * (capacity / production).max()
        )
        actual = maturity(retro_agent, search_space, technologies, agent_market)
        assert sorted(actual.dims) == sorted(search_space.dims)
        if expected_result is not None:
            assert (actual == expected_result).all()
        return actual

    # Test different threshold scenarios
    assert not check_maturity(1.1).any()  # Nothing should be true
    assert check_maturity(0.8).any()  # Some should be true
    assert (
        check_maturity(
            0.8 * (capacity / production).min() / (capacity / production).max()
        )
        == search_space
    ).any()


def test_init_from_tech(demand_share, technologies, agent_market):
    agent = namedtuple("DummyAgent", ["tolerance"])(tolerance=1e-8)

    # Test with producing technologies
    space = initialize_from_technologies(agent, demand_share, technologies=technologies)
    assert set(space.dims) == {"asset", "replacement"}
    assert (space.asset.values == demand_share.asset.values).all()
    assert (space.replacement.values == technologies.technology.values).all()
    assert space.all()

    # Test with non-producing technologies
    technologies.fixed_outputs[:] = 0
    space = initialize_from_technologies(agent, demand_share, technologies=technologies)
    assert not space.any()


def test_init_from_asset(technologies, rng):
    # Create test data
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

    # Test with assets
    space = initialize_from_assets(agent, demand=None, technologies=technologies)
    assert set(space.dims) == {"asset", "replacement"}
    assert space.replacement.isin(technologies.technology).all()
    assert technologies.technology.isin(space.replacement).all()
    assert set(space.asset.asset.values) == set(capacity.technology.values)


def test_init_from_asset_no_assets(technologies, rng):
    agent = namedtuple("DummyAgent", ["assets"])(
        xr.Dataset(dict(capacity=xr.DataArray(0)))
    )
    space = initialize_from_assets(agent, demand=None, technologies=technologies)
    assert set(space.dims) == {"replacement"}
    assert space.replacement.isin(technologies.technology).all()
    assert technologies.technology.isin(space.replacement).all()
