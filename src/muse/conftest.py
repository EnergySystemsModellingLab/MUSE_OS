from pytest import fixture


@fixture(autouse=True)
def add_np(doctest_namespace):
    import numpy
    import pandas
    import xarray

    doctest_namespace["np"] = numpy
    doctest_namespace["xr"] = xarray
    doctest_namespace["pd"] = pandas
