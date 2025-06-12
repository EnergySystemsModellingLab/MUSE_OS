import numpy as np
import xarray as xr
from pytest import fixture

from muse import examples

# Common test data
EXPECTED_TIMESLICES = [
    ("all-year", "all-week", "night"),
    ("all-year", "all-week", "morning"),
    ("all-year", "all-week", "afternoon"),
    ("all-year", "all-week", "early-peak"),
    ("all-year", "all-week", "late-peak"),
    ("all-year", "all-week", "evening"),
]
COMMODITIES = ["electricity", "gas", "heat", "wind", "CO2f"]


# Helper functions for common assertions
def assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars):
    """Assert basic dataset structure including dimensions, coordinates and data variables."""  # noqa: E501
    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == set(expected_dims)
    assert set(data.data_vars) == set(expected_data_vars)

    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims


def assert_data_types(data, expected_types):
    """Assert data types of variables match expected types."""
    for var, expected_type in expected_types.items():
        actual_type = str(data.data_vars[var].dtype)
        assert actual_type == expected_type, (
            f"Expected {var} to be {expected_type}, got {actual_type}"
        )


def assert_coordinate_values(data, coordinates: dict[str, list], check_order=False):
    """Assert coordinate values match expected values.

    Args:
        data: xarray Dataset or DataArray
        coordinates: dict mapping coordinate names to expected values
        check_order: whether to check the order of values
    """
    for coord, expected_values in coordinates.items():
        actual_values = data.coords[coord].values.tolist()
        if check_order:
            assert actual_values == expected_values, (
                f"Expected {coord} values to be {expected_values}, got {actual_values}"
            )
        else:
            assert set(actual_values) == set(expected_values), (
                f"Expected {coord} values to be {expected_values}, got {actual_values}"
            )


def assert_single_coordinate(data, selection, expected):
    """Assert values for a single coordinate selection match expected values."""
    actual = data.sel(**selection).data_vars
    for k, v in expected.items():
        if isinstance(v, float):
            assert np.isclose(actual[k].item(), v), (
                f"Expected {k} to be {v}, got {actual[k].item()}"
            )
        else:
            assert actual[k].item() == v, (
                f"Expected {k} to be {v}, got {actual[k].item()}"
            )


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


@fixture
def trade_model_path(tmp_path):
    """Creates temporary folder containing the trade model."""
    examples.copy_model(name="trade", path=tmp_path)
    return tmp_path / "model"


@fixture
def correlation_model_path(tmp_path):
    """Creates temporary folder containing the correlation model."""
    examples.copy_model(name="default_correlation", path=tmp_path)
    return tmp_path / "model"


def test_read_global_commodities(model_path):
    from muse.readers.csv import read_global_commodities

    path = model_path / "GlobalCommodities.csv"
    data = read_global_commodities(path)

    assert_dataset_structure(
        data,
        {"commodity"},
        {"commodity": ("commodity",)},
        {
            "comm_name": "object",
            "comm_type": "object",
            "emmission_factor": "float64",
            "heat_rate": "int64",
            "unit": "object",
        },
    )
    assert_data_types(
        data,
        {
            "comm_name": "object",
            "comm_type": "object",
            "emmission_factor": "float64",
            "heat_rate": "int64",
            "unit": "object",
        },
    )

    # Check coordinates
    expected_coord_values = {"commodity": COMMODITIES}
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {"commodity": "electricity"}
    expected = {
        "comm_name": "Electricity",
        "comm_type": "energy",
        "emmission_factor": 0.0,
        "heat_rate": 1,
        "unit": "PJ",
    }
    assert_single_coordinate(data, coord, expected)


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

    # Check coordinates
    expected_coord_values = {
        "year": [2020, 2050],
        "commodity": COMMODITIES,
        "region": ["R1"],
        "timeslice": list(range(1, 7)),
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    assert data.sel(year=2020, commodity="heat", region="R1", timeslice=1) == 1.0


def test_read_initial_market(model_path):
    from muse.readers.csv import read_initial_market

    data = read_initial_market(model_path / "Projections.csv")

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

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)
    assert hasattr(data.coords["timeslice"].to_index(), "levels")

    # Check coordinates
    expected_coord_values = {
        "region": ["R1"],
        "year": list(range(2010, 2105, 5)),
        "commodity": COMMODITIES,
        "timeslice": EXPECTED_TIMESLICES,
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {
        "year": 2010,
        "region": "R1",
        "commodity": "electricity",
        "timeslice": ("all-year", "all-week", "night"),
    }
    expected = {
        "prices": 14.81481472,
        "exports": 0.0,
        "imports": 0.0,
        "static_trade": 0.0,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_technodictionary(model_path):
    from muse.readers.csv import read_technodictionary

    data = read_technodictionary(model_path / "power" / "Technodata.csv")

    # Check properties of the dataset
    expected_dims = {"technology", "region"}
    expected_coords = {"technology": ("technology",), "region": ("region",)}
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

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "technology": ["gasCCGT", "windturbine"],
        "region": ["R1"],
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check coordinate consistency
    for var in data.data_vars:
        if var == "tech_type":
            assert list(data.data_vars[var].coords) == ["technology"]
        else:
            assert data.data_vars[var].coords.equals(data.coords)

    # Check values at a single coordinate
    coord = {"technology": "gasCCGT", "region": "R1"}
    expected = {
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
    assert_single_coordinate(data, coord, expected)


def test_read_technodata_timeslices(timeslice_model_path):
    from muse.readers.csv import read_technodata_timeslices

    data = read_technodata_timeslices(
        timeslice_model_path / "power" / "TechnodataTimeslices.csv"
    )

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

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "technology": ["gasCCGT", "windturbine"],
        "region": ["R1"],
        "year": [2020],
        "timeslice": EXPECTED_TIMESLICES,
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {
        "technology": "gasCCGT",
        "region": "R1",
        "year": 2020,
        "timeslice": ("all-year", "all-week", "night"),
    }
    expected = {"utilization_factor": 1, "minimum_service_factor": 0}
    assert_single_coordinate(data, coord, expected)


def test_read_io_technodata(model_path):
    from muse.readers.csv import read_io_technodata

    data = read_io_technodata(model_path / "power" / "CommIn.csv")

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

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "technology": ["gasCCGT", "windturbine"],
        "region": ["R1"],
        "year": [2020],
        "commodity": COMMODITIES,
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {"technology": "gasCCGT", "region": "R1", "year": 2020, "commodity": "gas"}
    expected = {"fixed": 1.67, "flexible": 0.0, "commodity_units": "PJ/PJ"}
    assert_single_coordinate(data, coord, expected)


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

    # Check coordinates
    expected_coord_values = {
        "region": ["R1"],
        "technology": ["gasCCGT", "windturbine"],
        "installed": [2020, 2020],
        "year": list(range(2020, 2055, 5)),
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    assert data.installed.sel(asset=0).item() == 2020
    assert data.technology.sel(asset=0).item() == "gasCCGT"
    assert data.sel(region="R1", asset=0, year=2020).item() == 1


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


def test_read_existing_trade(trade_model_path):
    from muse.readers.csv import read_trade

    data = read_trade(trade_model_path / "gas" / "ExistingTrade.csv", skiprows=[1])

    assert isinstance(data, xr.DataArray)

    # Check properties of the DataArray
    expected_dims = {"year", "technology", "dst_region", "region"}
    expected_coords = {
        "year": ("year",),
        "technology": ("technology",),
        "dst_region": ("dst_region",),
        "region": ("region",),
    }
    assert set(data.dims) == set(expected_dims)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims

    # Check coordinates
    expected_coord_values = {
        "year": [2010, 2020, 2030, 2040, 2050],
        "technology": ["gassupply1"],
        "dst_region": ["R1", "R2"],
        "region": ["R1", "R2"],
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    assert (
        data.sel(year=2010, technology="gassupply1", dst_region="R1", region="R2") == 0
    )


def test_read_trade_technodata(trade_model_path):
    from muse.readers.csv import read_trade

    data = read_trade(trade_model_path / "gas" / "TradeTechnodata.csv", drop="Unit")

    assert isinstance(data, xr.Dataset)

    # Check properties of the dataset
    expected_dims = {"technology", "dst_region", "region"}
    expected_coords = {
        "technology": ("technology",),
        "dst_region": ("dst_region",),
        "region": ("region",),
    }
    expected_data_vars = {
        "cap_par": "float64",
        "cap_exp": "float64",
        "fix_par": "float64",
        "fix_exp": "float64",
        "max_capacity_addition": "float64",
        "max_capacity_growth": "float64",
        "total_capacity_limit": "float64",
    }

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "technology": ["gassupply1"],
        "dst_region": ["R1", "R2"],
        "region": ["R1", "R2", "R3"],
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {"technology": "gassupply1", "dst_region": "R1", "region": "R1"}
    expected = {
        "cap_par": 3,
        "cap_exp": 1,
        "fix_par": 0.3,
        "fix_exp": 1,
        "max_capacity_addition": 200,
        "max_capacity_growth": 1,
        "total_capacity_limit": 3937.219,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_timeslice_shares(correlation_model_path):
    from muse.readers.csv import read_timeslice_shares

    data = read_timeslice_shares(
        correlation_model_path / "residential_presets" / "TimesliceSharepreset.csv"
    )

    assert isinstance(data, xr.DataArray)

    # Check properties of the DataArray
    expected_dims = {"region", "timeslice", "commodity"}
    expected_coords = {
        "region": ("region",),
        "timeslice": ("timeslice",),
        "commodity": ("commodity",),
    }
    assert set(data.dims) == set(expected_dims)
    for coord, dims in expected_coords.items():
        assert coord in data.coords
        assert data.coords[coord].dims == dims

    # Check coordinates
    expected_coord_values = {
        "region": ["R1"],
        "timeslice": list(range(1, 7)),
        "commodity": COMMODITIES,
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {"region": "R1", "timeslice": 1, "commodity": "heat"}
    assert data.sel(**coord).item() == 0.071


def test_read_macro_drivers(correlation_model_path):
    from muse.readers.csv import read_macro_drivers

    data = read_macro_drivers(
        correlation_model_path / "residential_presets" / "Macrodrivers.csv"
    )

    # Check properties of the dataset
    expected_dims = {"region", "year"}
    expected_coords = {
        "region": ("region",),
        "year": ("year",),
    }
    expected_data_vars = {
        "gdp": "int64",
        "population": "int64",
    }

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "region": ["R1"],
        "year": list(range(2010, 2111)),
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check values at a single coordinate
    coord = {"year": 2010, "region": "R1"}
    expected = {
        "gdp": 1206919,
        "population": 80004200,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_regression_parameters(correlation_model_path):
    from muse.readers.csv import read_regression_parameters

    data = read_regression_parameters(
        correlation_model_path / "residential_presets" / "regressionparameters.csv"
    )

    # Check properties of the dataset
    expected_dims = {"sector", "region", "commodity"}
    expected_coords = {
        "sector": ("sector",),
        "region": ("region",),
        "commodity": ("commodity",),
    }
    expected_data_vars = {
        "GDPexp": "float64",
        "constant": "float64",
        "GDPscaleLess": "float64",
        "GDPscaleGreater": "float64",
        "function_type": "<U16",
    }

    assert_dataset_structure(data, expected_dims, expected_coords, expected_data_vars)
    assert_data_types(data, expected_data_vars)

    # Check coordinates
    expected_coord_values = {
        "sector": ["residential"],
        "region": ["R1"],
        "commodity": ["electricity", "gas", "heat", "CO2f"],
    }
    assert_coordinate_values(data, expected_coord_values)

    # Check function type
    assert data.function_type.sel(sector="residential").item() == "logistic-sigmoid"

    # Check values at a single coordinate
    coord = {"sector": "residential", "region": "R1", "commodity": "heat"}
    expected = {
        "GDPexp": 0.0994,
        "constant": 1.01039e-05,
        "GDPscaleLess": 753.1068725,
        "GDPscaleGreater": 672.9316672,
    }
    assert_single_coordinate(data, coord, expected)
