import numpy as np
import xarray as xr
from pytest import fixture

from muse import examples


@fixture
def timeslice():
    """Sets up global timeslicing scheme to match the default model."""
    from muse.timeslices import setup_module

    timeslice = """
    [timeslices]
    all-year.all-week.night = 1460
    all-year.all-week.morning = 1460
    all-year.all-week.afternoon = 1460
    all-year.all-week.early-peak = 1460
    all-year.all-week.late-peak = 1460
    all-year.all-week.evening = 1460
    """

    setup_module(timeslice)


@fixture
def model_path(tmp_path, timeslice):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default", path=tmp_path)
    return tmp_path / "model"


@fixture
def timeslice_model_path(tmp_path, timeslice):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default_timeslice", path=tmp_path)
    return tmp_path / "model"


def test_read_global_commodities(model_path):
    from muse.readers.csv import read_global_commodities

    path = model_path / "GlobalCommodities.csv"
    data = read_global_commodities(path)

    assert isinstance(data, xr.Dataset)
    expected_coords = {"commodity"}
    expected_data_vars = {
        "comm_name": "object",
        "comm_type": "object",
        "emmission_factor": "float64",
        "heat_rate": "int64",
        "unit": "object",
    }
    assert set(data.coords) == set(expected_coords)
    assert set(data.data_vars) == set(expected_data_vars)
    for var, expected_type in expected_data_vars.items():
        actual_type = str(data.data_vars[var].dtype)
        assert actual_type == expected_type

    # Check a single coordinate
    actual = data.sel(commodity="electricity").data_vars
    expected = {
        "comm_name": "Electricity",
        "comm_type": "energy",
        "emmission_factor": 0.0,
        "heat_rate": 1,
        "unit": "PJ",
    }
    assert actual == expected


def test_read_presets(model_path):
    from muse.readers.csv import read_presets

    data = read_presets(str(model_path / "residential_presets" / "*.csv"))
    assert isinstance(data, xr.DataArray)

    # Check properties of the data array
    expected_dims = {"year", "commodity", "region", "timeslice"}
    expected_coords = {"year", "commodity", "region", "timeslice"}
    expected_dtype = "float64"
    assert set(data.dims) == set(expected_dims)
    assert set(data.coords) == set(expected_coords)
    assert data.dtype == expected_dtype

    # Check a single coordinate
    assert data.sel(year=2020, commodity="heat", region="R1", timeslice=1) == 1.0


def test_read_initial_market(model_path):
    from muse.readers.csv import read_initial_market

    data = read_initial_market(model_path / "Projections.csv")
    assert isinstance(data, xr.Dataset)

    # Check properties of the dataset
    expected_dims = {"region", "year", "commodity", "timeslice"}
    expected_coords = {
        "region": ("region",),
        "year": ("year",),
        "commodity": ("commodity",),
        "units_prices": ("commodity",),
        "timeslice": ("timeslice",),
        "month": ("timeslice",),
        "day": ("timeslice",),
        "hour": ("timeslice",),
    }
    expected_data_vars = {
        "prices": "float64",
        "exports": "float64",
        "imports": "float64",
        "static_trade": "float64",
    }
    assert set(data.dims) == set(expected_dims)
    assert set(data.data_vars) == set(expected_data_vars)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims
    assert hasattr(data.coords["timeslice"].to_index(), "levels")

    # Check a single coordinate
    actual = data.sel(
        year=2010,
        region="R1",
        commodity="electricity",
        timeslice=("all-year", "all-week", "night"),
    ).data_vars
    expected = {
        "prices": 14.81481472,
        "exports": 0.0,
        "imports": 0.0,
        "static_trade": 0.0,
    }
    for k, v in expected.items():
        assert np.isclose(actual[k].item(), v), (
            f"Expected {k} to be {v}, got {actual[k].item()}"
        )
