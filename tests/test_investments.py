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


def test_cliff_retirement_known_profile():
    from numpy import array
    from xarray import DataArray

    from muse.investments import cliff_retirement_profile

    technology = ["a", "b", "c"]
    lifetime = DataArray(
        range(1, 1 + len(technology)),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )

    profile = cliff_retirement_profile(technical_life=lifetime)
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
    from numpy.random import randint
    from xarray import DataArray

    from muse.investments import cliff_retirement_profile

    technology = list("abcde")

    lifetime = DataArray(
        sorted(randint(1, 10, len(technology))),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )
    effective_lifetime = (protected // lifetime + 1) * lifetime

    investment_year = 5
    profile = cliff_retirement_profile(
        technical_life=lifetime.clip(min=protected), investment_year=investment_year
    )
    assert profile.year.min() == investment_year
    assert profile.year.max() <= investment_year + effective_lifetime.max() + 1
    assert profile.astype(int).interp(year=investment_year).all()
    assert profile.astype(int).interp(year=investment_year + protected - 1).all()
    assert not profile.astype(int).interp(year=profile.year.max()).any()
