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


def test_read_technodictionary(model_path):
    from muse.readers.csv import read_technodictionary

    data = read_technodictionary(model_path / "power" / "Technodata.csv")
    assert isinstance(data, xr.Dataset)

    # Check properties of the dataset
    expected_dims = {"technology", "region"}
    expected_coords = {"technology", "region"}
    expected_data_vars = {
        "cap_par": "float64",
        "cap_exp": "int64",
        "fix_par": "int64",
        "fix_exp": "int64",
        "var_par": "int64",
        "var_exp": "int64",
        "max_capacity_addition": "int64",
        "max_capacity_growth": "float64",
        "total_capacity_limit": "int64",
        "technical_life": "int64",
        "utilization_factor": "float64",
        "scaling_size": "float64",
        "efficiency": "int64",
        "interest_rate": "float64",
        "type": "object",
        "agent1": "int64",
        "tech_type": "<U6",
    }

    assert set(data.dims) == set(expected_dims)
    assert set(data.coords) == set(expected_coords)
    assert set(data.data_vars) == set(expected_data_vars)
    for var, expected_type in expected_data_vars.items():
        actual_type = str(data.data_vars[var].dtype)
        assert actual_type == expected_type, (
            f"Expected {var} to be {expected_type}, got {actual_type}"
        )

    # Check single coordinate
    actual_gas = data.sel(technology="gasCCGT", region="R1").data_vars
    expected_gas = {
        "cap_par": 23.78234399,
        "cap_exp": 1,
        "fix_par": 0,
        "fix_exp": 1,
        "var_par": 0,
        "var_exp": 1,
        "max_capacity_addition": 10,
        "max_capacity_growth": 0.5,
        "total_capacity_limit": 100,
        "technical_life": 35,
        "utilization_factor": 0.9,
        "scaling_size": 1.89e-06,
        "efficiency": 86,
        "interest_rate": 0.1,
        "type": "energy",
        "agent1": 1,
    }
    for k, v in expected_gas.items():
        assert (
            np.isclose(actual_gas[k].item(), v)
            if isinstance(v, float)
            else actual_gas[k].item() == v
        ), f"Expected {k} to be {v}, got {actual_gas[k].item()}"


def test_read_technodata_timeslices(timeslice_model_path):
    from muse.readers.csv import read_technodata_timeslices

    data = read_technodata_timeslices(
        timeslice_model_path / "power" / "TechnodataTimeslices.csv"
    )
    assert isinstance(data, xr.Dataset)

    # Check properties of the dataset
    expected_dims = {"technology", "region", "year", "timeslice"}
    expected_coords = {
        "technology": ("technology",),
        "region": ("region",),
        "year": ("year",),
        "timeslice": ("timeslice",),
        "month": ("timeslice",),
        "day": ("timeslice",),
        "hour": ("timeslice",),
    }
    expected_data_vars = {
        "utilization_factor": "int64",
        "minimum_service_factor": "int64",
    }

    assert set(data.dims) == set(expected_dims)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims
    assert set(data.data_vars) == set(expected_data_vars)
    for var, expected_type in expected_data_vars.items():
        actual_type = str(data.data_vars[var].dtype)
        assert actual_type == expected_type, (
            f"Expected {var} to be {expected_type}, got {actual_type}"
        )

    # Check timeslice structure
    expected_timeslices = [
        ("all-year", "all-week", "night"),
        ("all-year", "all-week", "morning"),
        ("all-year", "all-week", "afternoon"),
        ("all-year", "all-week", "early-peak"),
        ("all-year", "all-week", "late-peak"),
        ("all-year", "all-week", "evening"),
    ]
    actual_timeslices = list(zip(data.month.values, data.day.values, data.hour.values))
    assert actual_timeslices == expected_timeslices

    # Check single coordinate
    actual = data.sel(
        technology="gasCCGT",
        region="R1",
        year=2020,
        timeslice=("all-year", "all-week", "night"),
    ).data_vars
    expected = {"utilization_factor": 1, "minimum_service_factor": 0}
    for k, v in expected.items():
        assert actual[k].item() == v, f"Expected {k} to be {v}, got {actual[k].item()}"


def test_read_io_technodata(model_path):
    from muse.readers.csv import read_io_technodata

    data = read_io_technodata(model_path / "power" / "CommIn.csv")
    assert isinstance(data, xr.Dataset)

    # Check properties of the dataset
    expected_dims = {"technology", "region", "year", "commodity"}
    expected_coords = {
        "technology": ("technology",),
        "region": ("region",),
        "year": ("year",),
        "commodity": ("commodity",),
    }
    expected_data_vars = {
        "fixed": "float64",
        "flexible": "float64",
        "commodity_units": "object",
    }

    assert set(data.dims) == set(expected_dims)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims
    assert set(data.data_vars) == set(expected_data_vars)
    for var, expected_type in expected_data_vars.items():
        actual_type = str(data.data_vars[var].dtype)
        assert actual_type == expected_type, (
            f"Expected {var} to be {expected_type}, got {actual_type}"
        )

    # Check single coordinate
    actual = data.sel(
        technology="gasCCGT", region="R1", year=2020, commodity="gas"
    ).data_vars
    expected = {"fixed": 1.67, "flexible": 0.0, "commodity_units": "PJ/PJ"}
    for k, v in expected.items():
        assert actual[k].item() == v, f"Expected {k} to be {v}, got {actual[k].item()}"


def test_read_initial_assets(model_path):
    from muse.readers.csv import read_initial_assets

    data = read_initial_assets(model_path / "power" / "ExistingCapacity.csv")
    assert isinstance(data, xr.DataArray)

    # Check properties of the DataArray
    expected_dims = {"region", "asset", "year"}
    expected_coords = {
        "region": ("region",),
        "technology": ("asset",),
        "installed": ("asset",),
        "year": ("year",),
    }

    assert set(data.dims) == set(expected_dims)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims

    # Check single coordinate
    assert data.installed.sel(asset=0).item() == 2020
    assert data.technology.sel(asset=0).item() == "gasCCGT"
    capacity = data.sel(region="R1", asset=0, year=2020).item()
    assert capacity == 1


def test_read_csv_agent_parameters(model_path):
    from muse.readers.csv import read_csv_agent_parameters

    data = read_csv_agent_parameters(model_path / "Agents.csv")
    assert isinstance(data, list)
    assert len(data) == 1

    # Check properties of the agent
    agent = data[0]
    expected = {
        "name": "A1",
        "region": "R1",
        "objectives": ["LCOE"],
        "search_rules": "all",
        "decision": {"name": "singleObj", "parameters": [("LCOE", True, 1)]},
        "agent_type": "newcapa",
        "quantity": 1,
        "maturity_threshold": -1,
        "spend_limit": float("inf"),
        "share": "agent1",
    }
    for k, v in expected.items():
        if k == "spend_limit":
            assert np.isinf(agent[k]), f"Expected {k} to be inf, got {agent[k]}"
        else:
            assert agent[k] == v, f"Expected {k} to be {v}, got {agent[k]}"
