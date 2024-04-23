from pytest import approx, fixture


@fixture
def regression_params():
    from numpy.random import rand
    from xarray import Dataset

    params = Dataset()
    params["region"] = "region", ["USA", "ATE"]
    params["sector"] = "sector", ["Residential", "Commercial"]
    params["commodity"] = "commodity", ["algae", "agrires", "coal"]

    shape = {k: len(v) for k, v in params.coords.items()}
    params["a"] = tuple(shape.keys()), rand(*shape.values())
    params["b"] = tuple(shape.keys()), rand(*shape.values())
    params["b0"] = tuple(shape.keys()), rand(*shape.values())
    params["b1"] = tuple(shape.keys()), rand(*shape.values())
    params["c"] = tuple(shape.keys()), rand(*shape.values())
    params["w"] = tuple(shape.keys()), rand(*shape.values())
    return params


@fixture
def drivers():
    from numpy.random import rand
    from xarray import Dataset

    drivers = Dataset()
    drivers["year"] = "year", [2010, 2015, 2020]
    drivers["region"] = "region", ["USA", "ATE"]
    drivers["gdp"] = (
        ("region", "year"),
        (1000 * rand(len(drivers.region), len(drivers.year))),
    )
    drivers["population"] = (
        ("region", "year"),
        (10 * rand(len(drivers.region), len(drivers.year))),
    )
    return drivers


def test_exponential(regression_params, drivers):
    from muse.regressions import Exponential
    from numpy import exp
    from xarray import broadcast

    rp = regression_params.drop_vars(("c", "w", "b0", "b1"))
    functor = Exponential(**(rp.data_vars))
    actual = functor(drivers)
    factor = 1e6 * drivers.population * rp.a
    expected = factor * exp(drivers.population / drivers.gdp * rp.b)
    expected, actual = broadcast(expected, actual)
    assert actual.values == approx(expected.values)

    partial = actual.sel(region="USA")
    a, b = broadcast(partial, functor(drivers, region="USA"))
    assert a.values == approx(b.values)


def test_linear(regression_params, drivers):
    from muse.regressions import Linear
    from xarray import DataArray, broadcast

    rp = regression_params.drop_vars(("c", "w", "b"))
    functor = Linear(**rp.data_vars)
    actual = functor(drivers, forecast=2)

    offset = drivers.gdp.sel(year=2010) / drivers.population.sel(year=2010)
    expected = rp.a * drivers.population + rp.b0 * (
        drivers.gdp - offset * drivers.population
    )
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)

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
    from pytest import approx  # noqa

    from muse import DEFAULT_SECTORS_DIRECTORY
    from tests.agents import test_regressions  # noqa

    sectors_dir = DEFAULT_SECTORS_DIRECTORY
    regression_params = test_regressions.regression_params()  # noqa
    drivers = test_regressions.drivers()  # noqa
