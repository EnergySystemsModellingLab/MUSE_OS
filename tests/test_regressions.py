import numpy as np
import xarray as xr
from pytest import approx, fixture

from muse.regressions import Exponential, Linear


def create_dataset(coords, random_vars):
    """Helper to create test datasets with random variables."""
    ds = xr.Dataset(coords=coords)
    shape = {k: len(v) for k, v in ds.coords.items()}
    dims = tuple(shape.keys())

    for var in random_vars:
        ds[var] = dims, np.random.rand(*shape.values())
    return ds


@fixture
def regression_params():
    coords = {
        "region": ["USA", "ATE"],
        "sector": ["Residential", "Commercial"],
        "commodity": ["algae", "agrires", "coal"],
    }
    random_vars = ["a", "b", "b0", "b1", "c", "w"]
    return create_dataset(coords, random_vars)


@fixture
def drivers():
    coords = {"year": [2010, 2015, 2020], "region": ["USA", "ATE"]}
    ds = xr.Dataset(coords=coords)
    shape = (len(ds.region), len(ds.year))
    ds["gdp"] = ("region", "year"), 1000 * np.random.rand(*shape)
    ds["population"] = ("region", "year"), 10 * np.random.rand(*shape)
    return ds


def test_exponential(regression_params, drivers):
    from numpy import exp
    from xarray import broadcast

    # Prepare regression parameters and create functor
    rp = regression_params.drop_vars(("c", "w", "b0", "b1"))
    functor = Exponential(**rp.data_vars)

    # Calculate expected and actual results
    actual = functor(drivers)
    factor = 1e6 * drivers.population * rp.a
    expected = factor * exp(drivers.population / drivers.gdp * rp.b)
    expected, actual = broadcast(expected, actual)
    assert actual.values == approx(expected.values)

    # Test partial selection
    partial = actual.sel(region="USA")
    a, b = broadcast(partial, functor(drivers, region="USA"))
    assert a.values == approx(b.values)


def test_linear(regression_params, drivers):
    from xarray import DataArray, broadcast

    # Prepare regression parameters and create functor
    rp = regression_params.drop_vars(("c", "w", "b"))
    functor = Linear(**rp.data_vars)

    # Test basic functionality
    actual = functor(drivers, forecast=2)
    offset = drivers.gdp.sel(year=2010) / drivers.population.sel(year=2010)
    expected = rp.a * drivers.population + rp.b0 * (
        drivers.gdp - offset * drivers.population
    )
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)

    # Test with interpolation
    year = [2010, 2012, 2014, 2020]
    scale = rp.b0.where(
        DataArray(year, coords={"year": year}, dims="year") + 2 < 2015, rp.b1
    )
    population = drivers.population.interp(year=year, method="linear")
    gdp = drivers.gdp.interp(year=year, method="linear")
    expected = rp.a * population + scale * (gdp - offset * population)
    actual = functor(drivers, forecast=2, year=year)
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)


if __name__ == "__main__":
    regression_params = regression_params()
    drivers = drivers()
