from pytest import fixture, mark


@fixture
def capacity_expansion():
    from numpy import arange
    from numpy.random import rand
    from xarray import Dataset
    from muse.investments import CapacityAddition

    data = Dataset()
    data["asset"] = "asset", arange(5, 10)
    data["replacement"] = "replacement", arange(0, 6)
    data["ranks"] = data.asset + data.replacement // 2
    data["ranks"] = data.ranks.rank("replacement").astype(int)
    data["deltas"] = (
        ("asset", "replacement"),
        rand(data.asset.size, data.replacement.size),
    )
    data["deltas"] *= rand(*data.deltas.shape) > 0.25

    return CapacityAddition(data.ranks, data.deltas)


def add_var(coordinates, *dims, factor=100.0):
    from numpy.random import rand

    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


def test_match_demand_smoke_test(
    retro_agent, technologies, demand_share, search_space, timeslice
):
    from numpy.random import randint
    from muse.investments import match_demand
    from muse.timeslices import convert_timeslice
    from muse.commodities import is_enduse
    from xarray import DataArray, Dataset
    from pytest import approx

    # missing entries in the timeslice result in comple behaviour,
    # just making sure this fixture does have missing entries.
    assert convert_timeslice(DataArray(1), timeslice).sum() == approx(1)

    prodparams = retro_agent.filter_input(
        technologies.fixed_outputs,
        year=retro_agent.year,
        technology=search_space.replacement,
    )

    search = Dataset(coords=search_space.coords)
    search["ranks"] = ("asset", "replacement"), randint(1, 5, search_space.shape)
    search["max_capacity"] = ("replacement", randint(0, 100, len(search.replacement)))
    maxdemand = convert_timeslice(
        search.max_capacity * search_space * prodparams, timeslice
    )
    search["demand"] = (
        maxdemand
        * (randint(0, 3, maxdemand.shape) == 0)
        / randint(1, 4, maxdemand.shape)
    ).sum("replacement")
    comm_usage = technologies.comm_usage.sel(commodity=search.commodity)
    search.demand[{"commodity": ~is_enduse(comm_usage)}] = 0

    capacity = match_demand(
        search.ranks,
        search_space,
        technologies,
        [
            Dataset(
                dict(b=search.max_capacity), attrs=dict(name="max capacity expansion")
            ),
            Dataset(dict(b=search.demand), attrs=dict(name="demand")),
        ],
        year=retro_agent.year,
    )
    assert set(capacity.dims) == {"timeslice", "replacement", "asset"}


def test_cliff_retirement_known_profile():
    from muse.investments import cliff_retirement_profile
    from numpy import array
    from xarray import DataArray

    technology = ["a", "b", "c"]
    lifetime = DataArray(
        range(1, 1 + len(technology)),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )

    profile = cliff_retirement_profile(lifetime)
    expected = array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    assert set(profile.dims) == {"year", "technology"}
    assert (profile == expected.T).all()


@mark.parametrize("protected", range(12))
def test_cliff_retirement_random_profile(protected):
    from muse.investments import cliff_retirement_profile
    from numpy.random import randint
    from xarray import DataArray

    technology = list("abcde")

    lifetime = DataArray(
        sorted(randint(1, 10, len(technology))),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )
    effective_lifetime = (protected // lifetime + 1) * lifetime

    current = 5
    profile = cliff_retirement_profile(
        lifetime, current_year=current, protected=protected
    )
    assert profile.year.min() == current
    assert profile.year.max() <= current + effective_lifetime.max() + 1
    assert profile.astype(int).interp(year=current).all()
    assert profile.astype(int).interp(year=current + protected).all()
    assert not profile.astype(int).interp(year=profile.year.max()).any()
