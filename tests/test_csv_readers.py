from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from pytest import fixture

from muse import examples
from muse.readers.toml import read_settings

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


@dataclass
class CoordinateSchema:
    """Schema for validating xarray coordinates.

    Attributes:
        dims: Tuple of dimension names for this coordinate
        dtype: Expected data type of the coordinate values
    """

    dims: tuple[str, ...]
    dtype: str = None  # Optional expected dtype


@dataclass
class DataArraySchema:
    """Schema for validating xarray DataArrays.

    Attributes:
        dims: Set of dimension names
        coords: Dictionary mapping coordinate names to their schemas
        dtype: Expected data type of the DataArray values
        name: Expected name of the DataArray (e.g. "value")
    """

    dims: set[str]
    coords: dict[str, CoordinateSchema]
    dtype: str
    name: str

    @classmethod
    def from_da(cls, data: xr.DataArray) -> DataArraySchema:
        """Generate a DataArraySchema from an existing DataArray."""
        return cls(
            dims=set(data.dims),
            coords={
                name: CoordinateSchema(dims=coord.dims, dtype=str(coord.dtype))
                for name, coord in data.coords.items()
            },
            dtype=str(data.dtype),
            name=data.name,
        )


@dataclass
class DatasetSchema:
    """Schema for validating xarray Datasets.

    Attributes:
        dims: Set of dimension names
        coords: Dictionary mapping coordinate names to their schemas
        data_vars: Dictionary mapping variable names to their expected dtypes
    """

    dims: set[str]
    coords: dict[str, CoordinateSchema]
    data_vars: dict[str, str]  # var_name -> dtype

    @classmethod
    def from_ds(cls, data: xr.Dataset) -> DatasetSchema:
        """Generate a DatasetSchema from an existing Dataset."""
        return cls(
            dims=set(data.dims),
            coords={
                name: CoordinateSchema(dims=coord.dims, dtype=str(coord.dtype))
                for name, coord in data.coords.items()
            },
            data_vars={name: str(var.dtype) for name, var in data.data_vars.items()},
        )


def assert_coordinate_values(data, coordinates, check_order=False):
    """Assert that coordinate values match expected values.

    Args:
        data: xarray Dataset or DataArray
        coordinates: dict mapping coordinate names to expected values
        check_order: if True, also check that values are in the same order
    """
    for coord_name, expected_values in coordinates.items():
        assert coord_name in data.coords, f"Missing coordinate {coord_name}"
        if check_order:
            assert list(data.coords[coord_name].values) == list(expected_values), (
                f"Expected {coord_name} to have values {expected_values} in order, "
                f"got {list(data.coords[coord_name].values)}"
            )
        else:
            assert set(data.coords[coord_name].values) == set(expected_values), (
                f"Expected {coord_name} to have values {expected_values}, "
                f"got {list(data.coords[coord_name].values)}"
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
def model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default", path=tmp_path)
    path = tmp_path / "model"
    read_settings(path / "settings.toml")  # setup globals
    return path


@fixture
def timeslice_model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default_timeslice", path=tmp_path)
    path = tmp_path / "model"
    read_settings(path / "settings.toml")  # setup globals
    return path


@fixture
def trade_model_path(tmp_path):
    """Creates temporary folder containing the trade model."""
    examples.copy_model(name="trade", path=tmp_path)
    path = tmp_path / "model"
    read_settings(path / "settings.toml")  # setup globals
    return path


@fixture
def correlation_model_path(tmp_path):
    """Creates temporary folder containing the correlation model."""
    examples.copy_model(name="default_correlation", path=tmp_path)
    path = tmp_path / "model"
    read_settings(path / "settings.toml")  # setup globals
    return path


def test_read_global_commodities(model_path):
    from muse.readers.csv import read_global_commodities

    path = model_path / "GlobalCommodities.csv"
    data = read_global_commodities(path)

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"commodity"},
        coords={"commodity": CoordinateSchema(dims=("commodity",), dtype="object")},
        data_vars={
            "commodity_type": "object",
            "unit": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(data, {"commodity": COMMODITIES})

    # Check values at a single coordinate
    coord = {"commodity": "electricity"}
    expected = {
        "commodity_type": "energy",
        "unit": "PJ",
    }
    assert_single_coordinate(data, coord, expected)


def test_read_presets(model_path):
    from muse.readers.csv import read_presets

    data = read_presets(str(model_path / "residential_presets" / "*.csv"))

    # Check data against schema
    expected_schema = DataArraySchema(
        dims={"year", "commodity", "region", "timeslice"},
        coords={
            "year": CoordinateSchema(("year",), dtype="int64"),
            "commodity": CoordinateSchema(("commodity",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
            "timeslice": CoordinateSchema(("timeslice",), dtype="object"),
            "month": CoordinateSchema(("timeslice",), dtype="object"),
            "day": CoordinateSchema(("timeslice",), dtype="object"),
            "hour": CoordinateSchema(("timeslice",), dtype="object"),
        },
        dtype="float64",
        name="value",
    )
    assert DataArraySchema.from_da(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "year": [2020, 2050],
            "commodity": COMMODITIES,
            "region": ["r1"],
            "timeslice": EXPECTED_TIMESLICES,
        },
    )

    # Check values at a single coordinate
    assert (
        data.sel(
            year=2020,
            commodity="heat",
            region="r1",
            timeslice=("all-year", "all-week", "night"),
        )
        == 1.0
    )


def test_read_initial_market(model_path):
    from muse.readers.csv import read_initial_market

    data = read_initial_market(model_path / "Projections.csv", currency="MUS$2010")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"region", "year", "commodity", "timeslice"},
        coords={
            "region": CoordinateSchema(("region",), dtype="object"),
            "year": CoordinateSchema(("year",), dtype="int64"),
            "commodity": CoordinateSchema(("commodity",), dtype="object"),
            "units_prices": CoordinateSchema(("commodity",), dtype="<U11"),
            "timeslice": CoordinateSchema(("timeslice",), dtype="object"),
            "month": CoordinateSchema(("timeslice",), dtype="object"),
            "day": CoordinateSchema(("timeslice",), dtype="object"),
            "hour": CoordinateSchema(("timeslice",), dtype="object"),
        },
        data_vars={
            "prices": "float64",
            "exports": "float64",
            "imports": "float64",
            "static_trade": "float64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "year": list(range(2020, 2055, 5)),
            "commodity": COMMODITIES,
            "units_prices": [
                "MUS$2010/PJ",
                "MUS$2010/PJ",
                "MUS$2010/PJ",
                "MUS$2010/PJ",
                "MUS$2010/kt",
            ],
            "timeslice": EXPECTED_TIMESLICES,
        },
    )

    assert hasattr(data.coords["timeslice"].to_index(), "levels")

    # Check values at a single coordinate
    coord = {
        "year": 2020,
        "region": "r1",
        "commodity": "CO2f",
        "timeslice": ("all-year", "all-week", "night"),
    }
    expected = {
        "prices": 0.08314119,
        "exports": 0.0,
        "imports": 0.0,
        "static_trade": 0.0,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_technodictionary(model_path):
    from muse.readers.csv import read_technodictionary

    data = read_technodictionary(model_path / "power" / "Technodata.csv")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"technology", "region", "year"},
        coords={
            "technology": CoordinateSchema(("technology",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
            "year": CoordinateSchema(("year",), dtype="int64"),
        },
        data_vars={
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
            "minimum_service_factor": "float64",
            "interest_rate": "float64",
            "agent1": "int64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
            "year": [2020],
        },
    )

    # Check coordinate consistency
    for var in data.data_vars:
        if var == "tech_type":
            assert list(data.data_vars[var].coords) == ["technology"]
        else:
            assert data.data_vars[var].coords.equals(data.coords)

    # Check values at a single coordinate
    coord = {"technology": "gasCCGT", "region": "r1", "year": 2020}
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
        "minimum_service_factor": 0.0,
        "interest_rate": 0.1,
        "agent1": 1,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_technodata_timeslices(timeslice_model_path):
    from muse.readers.csv import read_technodata_timeslices

    data = read_technodata_timeslices(
        timeslice_model_path / "power" / "TechnodataTimeslices.csv"
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"technology", "region", "timeslice", "year"},
        coords={
            "technology": CoordinateSchema(("technology",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
            "year": CoordinateSchema(("year",), dtype="int64"),
            "timeslice": CoordinateSchema(("timeslice",), dtype="object"),
            "month": CoordinateSchema(("timeslice",), dtype="object"),
            "day": CoordinateSchema(("timeslice",), dtype="object"),
            "hour": CoordinateSchema(("timeslice",), dtype="object"),
        },
        data_vars={
            "utilization_factor": "float64",
            "minimum_service_factor": "float64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
            "timeslice": EXPECTED_TIMESLICES,
            "year": [2020],
        },
    )

    assert hasattr(data.coords["timeslice"].to_index(), "levels")

    # Check values at a single coordinate
    coord = {
        "technology": "gasCCGT",
        "region": "r1",
        "timeslice": ("all-year", "all-week", "night"),
        "year": 2020,
    }
    expected = {"utilization_factor": 1.0, "minimum_service_factor": 0.0}
    assert_single_coordinate(data, coord, expected)


def test_read_io_technodata(model_path):
    from muse.readers.csv import read_io_technodata

    data = read_io_technodata(model_path / "power" / "CommIn.csv")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"technology", "region", "commodity", "year"},
        coords={
            "technology": CoordinateSchema(("technology",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
            "commodity": CoordinateSchema(("commodity",), dtype="object"),
            "year": CoordinateSchema(("year",), dtype="int64"),
        },
        data_vars={
            "fixed": "float64",
            "flexible": "float64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
            "commodity": COMMODITIES,
            "year": [2020],
        },
    )

    # Check values at a single coordinate
    coord = {"technology": "gasCCGT", "region": "r1", "commodity": "gas", "year": 2020}
    expected = {"fixed": 1.67, "flexible": 0.0}
    assert_single_coordinate(data, coord, expected)


def test_read_initial_capacity(model_path):
    from muse.readers.csv import read_initial_capacity

    data = read_initial_capacity(model_path / "power" / "ExistingCapacity.csv")

    # Check data against schema
    expected_schema = DataArraySchema(
        dims={"region", "asset", "year"},
        coords={
            "region": CoordinateSchema(("region",), dtype="object"),
            "technology": CoordinateSchema(("asset",), dtype="object"),
            "installed": CoordinateSchema(("asset",), dtype="int64"),
            "year": CoordinateSchema(("year",), dtype="int64"),
        },
        dtype="float64",
        name="value",
    )
    assert DataArraySchema.from_da(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "technology": ["gasCCGT", "windturbine"],
            "installed": [2020, 2020],
            "year": list(range(2020, 2055, 5)),
        },
    )

    # Check values at a single coordinate
    assert data.installed.sel(asset=0).item() == 2020
    assert data.technology.sel(asset=0).item() == "gasCCGT"
    assert data.sel(region="r1", asset=0, year=2020).item() == 1


def test_read_agent_parameters(model_path):
    from muse.readers.csv import read_agent_parameters

    data = read_agent_parameters(model_path / "Agents.csv")
    assert isinstance(data, list)
    assert len(data) == 1

    # Check properties of the agent
    agent = data[0]
    assert isinstance(agent, dict)
    expected = {
        "name": "A1",
        "region": "r1",
        "objectives": ["LCOE"],
        "search_rules": "all",
        "decision": {"name": "singleObj", "parameters": [("LCOE", True, 1)]},
        "agent_type": "newcapa",
        "quantity": 1,
        "share": "agent1",
    }
    assert agent == expected


def test_read_existing_trade(trade_model_path):
    from muse.readers.csv import read_existing_trade

    data = read_existing_trade(trade_model_path / "gas" / "ExistingTrade.csv")

    # Check data against schema
    expected_schema = DataArraySchema(
        dims={"region", "dst_region", "year", "asset"},
        coords={
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "dst_region": CoordinateSchema(dims=("dst_region",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
            "technology": CoordinateSchema(dims=("asset",), dtype="object"),
            "installed": CoordinateSchema(dims=("asset",), dtype="int64"),
        },
        dtype="float64",
        name="value",
    )
    assert DataArraySchema.from_da(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "year": [2010, 2020, 2030, 2040, 2050],
            "technology": ["gassupply1"],
            "installed": [2010],
            "dst_region": ["r1", "r2"],
            "region": ["r1", "r2"],
        },
    )

    # Check values at a single coordinate
    assert data.sel(year=2010, asset=0, dst_region="r1", region="r2") == 0


def test_read_trade_technodata(trade_model_path):
    from muse.readers.csv import read_trade_technodata

    data = read_trade_technodata(trade_model_path / "gas" / "TradeTechnodata.csv")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"technology", "dst_region", "region"},
        coords={
            "technology": CoordinateSchema(("technology",), dtype="object"),
            "dst_region": CoordinateSchema(("dst_region",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
        },
        data_vars={
            "cap_par": "float64",
            "fix_par": "float64",
            "max_capacity_addition": "float64",
            "max_capacity_growth": "float64",
            "total_capacity_limit": "float64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "technology": ["gassupply1"],
            "dst_region": ["r1", "r2"],
            "region": ["r1", "r2"],
        },
    )

    # Check values at a single coordinate
    coord = {"technology": "gassupply1", "dst_region": "r1", "region": "r1"}
    expected = {
        "cap_par": 3,
        "fix_par": 0.3,
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

    # Check data against schema
    expected_schema = DataArraySchema(
        dims={"region", "timeslice", "commodity"},
        coords={
            "region": CoordinateSchema(("region",), dtype="object"),
            "commodity": CoordinateSchema(("commodity",), dtype="object"),
            "timeslice": CoordinateSchema(("timeslice",), dtype="object"),
            "month": CoordinateSchema(("timeslice",), dtype="object"),
            "day": CoordinateSchema(("timeslice",), dtype="object"),
            "hour": CoordinateSchema(("timeslice",), dtype="object"),
        },
        dtype="float64",
        name="value",
    )
    assert DataArraySchema.from_da(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "timeslice": EXPECTED_TIMESLICES,
            "commodity": COMMODITIES,
        },
    )

    # Check values at a single coordinate
    coord = {
        "region": "r1",
        "timeslice": ("all-year", "all-week", "night"),
        "commodity": "heat",
    }
    assert data.sel(**coord).item() == 0.071


def test_read_macro_drivers(correlation_model_path):
    from muse.readers.csv import read_macro_drivers

    data = read_macro_drivers(
        correlation_model_path / "residential_presets" / "Macrodrivers.csv"
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"region", "year"},
        coords={
            "region": CoordinateSchema(("region",), dtype="object"),
            "year": CoordinateSchema(("year",), dtype="int64"),
        },
        data_vars={
            "gdp": "int64",
            "population": "int64",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "year": list(range(2010, 2111)),
        },
    )

    # Check values at a single coordinate
    coord = {"year": 2010, "region": "r1"}
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

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"sector", "region", "commodity"},
        coords={
            "sector": CoordinateSchema(("sector",), dtype="object"),
            "region": CoordinateSchema(("region",), dtype="object"),
            "commodity": CoordinateSchema(("commodity",), dtype="object"),
        },
        data_vars={
            "GDPexp": "float64",
            "constant": "float64",
            "GDPscaleLess": "float64",
            "GDPscaleGreater": "float64",
            "function_type": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "sector": ["residential"],
            "region": ["r1"],
            "commodity": COMMODITIES,
        },
    )

    # Check function type (should be one value per sector)
    assert data.function_type.sel(sector="residential").item() == "logistic-sigmoid"

    # Check values at a single coordinate
    coord = {"sector": "residential", "region": "r1", "commodity": "heat"}
    expected = {
        "GDPexp": 0.0994,
        "constant": 1.01039e-05,
        "GDPscaleLess": 753.1068725,
        "GDPscaleGreater": 672.9316672,
        "function_type": "logistic-sigmoid",
    }
    assert_single_coordinate(data, coord, expected)


def test_read_technologies(model_path):
    from muse.readers.csv import read_technologies

    # Read technologies
    data = read_technologies(
        technodata_path=model_path / "power" / "Technodata.csv",
        comm_out_path=model_path / "power" / "CommOut.csv",
        comm_in_path=model_path / "power" / "CommIn.csv",
        time_framework=[2020, 2025, 2030, 2035, 2040, 2045, 2050],
        interpolation_mode="linear",
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"commodity", "technology", "region", "year"},
        coords={
            "technology": CoordinateSchema(dims=("technology",), dtype="object"),
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
        },
        data_vars={
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
            "minimum_service_factor": "float64",
            "interest_rate": "float64",
            "agent1": "int64",
            "fixed_outputs": "float64",
            "fixed_inputs": "float64",
            "flexible_inputs": "float64",
            "unit": "object",
            "comm_usage": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "commodity": ["electricity", "gas", "heat", "wind", "CO2f"],
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
        },
    )


def test_read_technologies__timeslice(timeslice_model_path):
    """Testing the read_technologies function with the timeslice model."""
    from muse.readers.csv import read_technologies

    data = read_technologies(
        technodata_path=timeslice_model_path / "power" / "Technodata.csv",
        comm_out_path=timeslice_model_path / "power" / "CommOut.csv",
        comm_in_path=timeslice_model_path / "power" / "CommIn.csv",
        technodata_timeslices_path=timeslice_model_path
        / "power"
        / "TechnodataTimeslices.csv",
        time_framework=[2020, 2025, 2030, 2035, 2040, 2045, 2050],
        interpolation_mode="linear",
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"technology", "timeslice", "region", "commodity", "year"},
        coords={
            "technology": CoordinateSchema(dims=("technology",), dtype="object"),
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "timeslice": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "month": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "day": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "hour": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
        },
        data_vars={
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
            "interest_rate": "float64",
            "agent1": "int64",
            "fixed_outputs": "float64",
            "fixed_inputs": "float64",
            "flexible_inputs": "float64",
            "utilization_factor": "float64",
            "minimum_service_factor": "float64",
            "unit": "object",
            "comm_usage": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "commodity": ["electricity", "gas", "heat", "wind", "CO2f"],
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
            "timeslice": EXPECTED_TIMESLICES,
            "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
        },
    )


def test_read_technodata(model_path):
    from muse.readers.toml import read_settings, read_technodata

    settings = read_settings(model_path / "settings.toml")
    data = read_technodata(
        settings,
        sector_name="power",
        interpolation_mode="linear",
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"commodity", "technology", "region", "year"},
        coords={
            "technology": CoordinateSchema(dims=("technology",), dtype="object"),
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
        },
        data_vars={
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
            "minimum_service_factor": "float64",
            "interest_rate": "float64",
            "agent1": "int64",
            "fixed_outputs": "float64",
            "fixed_inputs": "float64",
            "flexible_inputs": "float64",
            "unit": "object",
            "comm_usage": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "commodity": ["electricity", "gas", "wind", "CO2f"],
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1"],
            "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
        },
    )


def test_read_technodata__trade(trade_model_path):
    """Testing the read_technodata function with the trade model."""
    from muse.readers.toml import read_settings, read_technodata

    settings = read_settings(trade_model_path / "settings.toml")
    data = read_technodata(
        settings,
        sector_name="power",
        interpolation_mode="linear",
    )

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"dst_region", "commodity", "region", "technology", "year"},
        coords={
            "technology": CoordinateSchema(dims=("technology",), dtype="object"),
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "dst_region": CoordinateSchema(dims=("dst_region",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
        },
        data_vars={
            "cap_exp": "int64",
            "fix_exp": "int64",
            "var_par": "int64",
            "var_exp": "int64",
            "technical_life": "int64",
            "utilization_factor": "float64",
            "minimum_service_factor": "float64",
            "interest_rate": "float64",
            "agent1": "int64",
            "fixed_outputs": "float64",
            "fixed_inputs": "float64",
            "flexible_inputs": "float64",
            "unit": "object",
            "max_capacity_addition": "float64",
            "max_capacity_growth": "float64",
            "total_capacity_limit": "float64",
            "cap_par": "float64",
            "fix_par": "float64",
            "comm_usage": "object",
        },
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "commodity": ["electricity", "gas", "wind", "CO2f"],
            "technology": ["gasCCGT", "windturbine"],
            "region": ["r1", "r2"],
            "dst_region": ["r1", "r2"],
            "year": [2010, 2020, 2025, 2030, 2035],
        },
    )


def test_read_presets_sector(model_path):
    from muse.readers.toml import read_presets_sector, read_settings

    settings = read_settings(model_path / "settings.toml")
    data = read_presets_sector(settings, sector_name="residential_presets")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"year", "region", "timeslice", "commodity"},
        coords={
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "timeslice": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "month": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "day": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "hour": CoordinateSchema(dims=("timeslice",), dtype="object"),
        },
        data_vars={"consumption": "float64", "costs": "float64", "supply": "float64"},
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "year": [2020, 2050],
            "commodity": COMMODITIES,
            "timeslice": EXPECTED_TIMESLICES,
        },
    )

    # Check values at a single coordinate
    coord = {
        "year": 2020,
        "region": "r1",
        "commodity": "heat",
        "timeslice": ("all-year", "all-week", "night"),
    }
    expected = {
        "consumption": 1.0,
        "costs": 0,
        "supply": 0,
    }
    assert_single_coordinate(data, coord, expected)


def test_read_presets_sector__correlation(correlation_model_path):
    """Testing the read_presets_sector function with the correlation model."""
    from muse.readers.toml import read_presets_sector, read_settings

    settings = read_settings(correlation_model_path / "settings.toml")
    data = read_presets_sector(settings, sector_name="residential_presets")

    # Check data against schema
    expected_schema = DatasetSchema(
        dims={"commodity", "year", "timeslice", "region"},
        coords={
            "region": CoordinateSchema(dims=("region",), dtype="object"),
            "commodity": CoordinateSchema(dims=("commodity",), dtype="object"),
            "year": CoordinateSchema(dims=("year",), dtype="int64"),
            "timeslice": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "month": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "day": CoordinateSchema(dims=("timeslice",), dtype="object"),
            "hour": CoordinateSchema(dims=("timeslice",), dtype="object"),
        },
        data_vars={"consumption": "float64", "costs": "float64", "supply": "float64"},
    )
    assert DatasetSchema.from_ds(data) == expected_schema

    # Check coordinate values
    assert_coordinate_values(
        data,
        {
            "region": ["r1"],
            "year": range(2010, 2111),
            "commodity": COMMODITIES,
            "timeslice": EXPECTED_TIMESLICES,
        },
    )

    # Check values at a single coordinate
    coord = {
        "year": 2020,
        "region": "r1",
        "commodity": "heat",
        "timeslice": ("all-year", "all-week", "night"),
    }
    expected = {
        "consumption": 0.8958,
        "costs": 0,
        "supply": 0,
    }
    assert_single_coordinate(data, coord, expected)
