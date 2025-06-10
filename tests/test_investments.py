import numpy as np
import xarray as xr
from pytest import mark

from muse.investments import cliff_retirement_profile


def test_cliff_retirement_known_profile():
    """Test cliff retirement profile with known values and expected output."""
    technology = ["a", "b", "c"]
    lifetime = xr.DataArray(
        np.arange(1, len(technology) + 1),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )

    profile = cliff_retirement_profile(technical_life=lifetime, investment_year=2020)
    expected = np.array(
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
    """Test cliff retirement profile with random lifetimes and protected periods."""
    technology = list("abcde")
    lifetime = xr.DataArray(
        sorted(np.random.randint(1, 10, len(technology))),
        dims="technology",
        coords={"technology": technology},
        name="technical_life",
    )

    investment_year = 2020
    effective_lifetime = (protected // lifetime + 1) * lifetime
    profile = cliff_retirement_profile(
        technical_life=lifetime.clip(min=protected), investment_year=investment_year
    )

    # Verify profile boundaries and properties
    profile_int = profile.astype(int)
    assert profile.year.min() == investment_year
    assert profile.year.max() <= investment_year + effective_lifetime.max() + 1
    assert profile_int.interp(year=investment_year).all()
    assert profile_int.interp(year=investment_year + protected - 1).all()
    assert not profile_int.interp(year=profile.year.max()).any()
