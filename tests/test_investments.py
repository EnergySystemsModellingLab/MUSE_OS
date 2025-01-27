from pytest import mark


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

    profile = cliff_retirement_profile(technical_life=lifetime, investment_year=2020)
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

    investment_year = 2020
    profile = cliff_retirement_profile(
        technical_life=lifetime.clip(min=protected), investment_year=investment_year
    )
    assert profile.year.min() == investment_year
    assert profile.year.max() <= investment_year + effective_lifetime.max() + 1
    assert profile.astype(int).interp(year=investment_year).all()
    assert profile.astype(int).interp(year=investment_year + protected - 1).all()
    assert not profile.astype(int).interp(year=profile.year.max()).any()
