from itertools import chain, permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import toml
import xarray as xr
from pytest import fixture, mark, raises

YEARS = list(range(2010, 2105, 5))
COMMODITIES = ["electricity", "gas", "heat", "wind", "CO2f"]
TIMESLICES = [
    ("all-year", "all-week", hour)
    for hour in ["night", "morning", "afternoon", "early-peak", "late-peak", "evening"]
]


@fixture
def user_data_files(settings: dict) -> None:
    """Creates test files related to user data."""
    for file_key, file_path in settings["global_input_files"].items():
        if file_key == "path" or not file_path:
            continue
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Some data")


@fixture
def plugins(settings: dict, tmp_path) -> Path:
    """Creates test files related to custom modules."""
    plugin = tmp_path / "plugins" / "cat.py"
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_text("my_cat_colour = 'tabby' ")
    settings["plugins"] = str(plugin)
    return plugin


def test_add_known_parameters(settings: dict):
    """Test adding known parameters to settings."""
    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import MissingSettings, add_known_parameters

    defaults = toml.load(DEFAULT_SETTINGS_PATH)
    settings_copy = settings.copy()

    # Test required parameters
    settings_copy.pop("time_framework")
    with raises(MissingSettings):
        add_known_parameters(defaults, settings_copy)

    # Test optional parameters
    settings_copy = settings.copy()
    settings_copy.pop("global_input_files")
    new_settings = add_known_parameters(defaults, settings_copy)
    assert "projections" in new_settings["global_input_files"]


def test_add_unknown_parameters(settings: dict):
    """Test adding unknown parameters to settings."""
    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import add_unknown_parameters

    defaults = toml.load(DEFAULT_SETTINGS_PATH)
    settings_copy = settings.copy()
    settings_copy["my_new_parameter"] = 42

    new_settings = add_unknown_parameters(defaults, settings_copy)
    assert "my_new_parameter" in new_settings


def test_check_log_level(settings: dict):
    """Tests log level validation."""
    from muse.readers.toml import check_log_level

    check_log_level(settings)


def test_check_interpolation_mode(settings: dict):
    """Tests interpolation mode validation."""
    from muse.readers.toml import check_interpolation_mode

    check_interpolation_mode(settings)


def test_check_budget_parameters(settings: dict):
    """Tests budget parameters validation."""
    from muse.readers.toml import check_budget_parameters

    check_budget_parameters(settings)


def test_check_global_data_files(settings: dict, user_data_files):
    """Tests global data files validation."""
    from muse.readers.toml import check_global_data_files

    check_global_data_files(settings)

    # Test file not found error
    proj_file = Path(settings["global_input_files"]["projections"])
    proj_file.rename(proj_file.parent / "my_file")
    with raises(AssertionError):
        check_global_data_files(settings)


def test_check_global_data_dir(settings: dict, user_data_files):
    """Tests global data directory validation."""
    from muse.readers.toml import check_global_data_files

    check_global_data_files(settings)

    # Test directory not found error
    path = Path(settings["global_input_files"]["path"])
    path.rename(path.parent / "my_directory")
    with raises(AssertionError):
        check_global_data_files(settings)


def test_check_plugins(settings: dict, plugins: Path):
    """Tests plugin validation."""
    from muse.readers.toml import IncorrectSettings, check_plugins

    check_plugins(settings)

    # Test plugin not found error
    settings["plugins"] = plugins.parent / f"{plugins.stem}_2{plugins.suffix}"
    with raises(IncorrectSettings):
        check_plugins(settings)


def test_check_iteration_control(settings: dict):
    """Tests iteration control validation."""
    from muse.readers.toml import check_iteration_control

    # Test valid settings
    settings.update({"equilibrium": "off", "maximum_iterations": 7})
    check_iteration_control(settings)
    assert not settings["equilibrium"]

    # Test invalid maximum iterations
    settings.update({"equilibrium": True, "maximum_iterations": -1})
    with raises(AssertionError):
        check_iteration_control(settings)

    # Test invalid tolerance
    settings.update({"maximum_iterations": 5, "tolerance": -1})
    with raises(AssertionError):
        check_iteration_control(settings)


def test_format_paths_cwd():
    from pathlib import Path

    from muse.readers.toml import format_paths

    settings = format_paths({"a_path": "{cwd}/a/b/c"})
    assert str(Path().absolute() / "a" / "b" / "c") == settings["a_path"]


def test_format_paths_default_path():
    from pathlib import Path

    from muse.readers.toml import format_paths

    settings = format_paths({"a_path": "{path}/a/b/c"})
    assert str(Path().absolute() / "a" / "b" / "c") == settings["a_path"]


def test_format_paths_path():
    from pathlib import Path

    from muse.readers.toml import format_paths

    settings = format_paths({"path": "{cwd}/a/b/", "a_path": "{path}/c"})
    assert str(Path().absolute() / "a" / "b" / "c") == settings["a_path"]


def test_split_toml_one_down(tmpdir):
    """Test single level of TOML file inclusion."""
    from toml import dumps

    from muse.readers.toml import read_split_toml

    # Create outer TOML with inner file reference
    outer_content = {
        "path": str(tmpdir),
        "some_section": {"include_path": "{path}/inner.toml"},
        "another_section": {"option_a": "a"},
    }
    (tmpdir / "outer.toml").write(dumps(outer_content))

    # Create inner TOML file
    inner_content = {"some_section": {"my_option": "found it!"}}
    (tmpdir / "inner.toml").write(dumps(inner_content))

    # Test merged result
    result = read_split_toml(tmpdir / "outer.toml")
    assert set(result) == {"path", "some_section", "another_section"}
    assert result["another_section"] == {"option_a": "a"}
    assert result["some_section"] == {"my_option": "found it!"}


def test_split_toml_nested(tmpdir):
    """Test nested TOML file inclusion."""
    from toml import dumps

    from muse.readers.toml import read_split_toml

    # Create outer TOML with nested inner file reference
    outer_content = {
        "path": str(tmpdir),
        "some_section": {"nested": {"include_path": "{path}/inner.toml"}},
        "another_section": {"option_a": "a"},
    }
    (tmpdir / "outer.toml").write(dumps(outer_content))

    # Create inner TOML file
    inner_content = {"nested": {"my_option": "found it!"}}
    (tmpdir / "inner.toml").write(dumps(inner_content))

    # Test merged result
    result = read_split_toml(tmpdir / "outer.toml")
    assert set(result) == {"path", "some_section", "another_section"}
    assert result["another_section"] == {"option_a": "a"}
    assert result["some_section"]["nested"] == {"my_option": "found it!"}


def test_split_toml_errors(tmpdir):
    """Test error cases for TOML file inclusion."""
    from toml import dumps

    from muse.readers.toml import IncorrectSettings, MissingSettings, read_split_toml

    # Test too many options in outer file
    outer_content = {
        "path": str(tmpdir),
        "some_section": {
            "nested": {"include_path": "{path}/inner.toml", "extra": "error"}
        },
    }
    (tmpdir / "outer.toml").write(dumps(outer_content))
    (tmpdir / "inner.toml").write(dumps({"nested": {"my_option": "found it!"}}))
    with raises(IncorrectSettings):
        read_split_toml(tmpdir / "outer.toml")

    # Test too many sections in inner file
    outer_content = {
        "path": str(tmpdir),
        "some_section": {"nested": {"include_path": "{path}/inner.toml"}},
    }
    (tmpdir / "outer.toml").write(dumps(outer_content))
    (tmpdir / "inner.toml").write(
        dumps({"extra": "error", "nested": {"my_option": "found it!"}})
    )
    with raises(IncorrectSettings):
        read_split_toml(tmpdir / "outer.toml")

    # Test incorrect inner section name
    (tmpdir / "inner.toml").write(dumps({"incorrect_name": {"my_option": "found it!"}}))
    with raises(MissingSettings):
        read_split_toml(tmpdir / "outer.toml")


def test_format_path():
    """Test path formatting with different variables."""
    from muse.readers.toml import format_path

    test_paths = {
        "cwd": "current_path",
        "path": "this_path",
        "muse_sectors": "sectors_path",
    }

    for var, value in test_paths.items():
        expected = str(Path(value).absolute() / "{other_param}")
        result = format_path(f"{{{var}}}/{{other_param}}", **{var: value})
        assert result == expected


@mark.parametrize("suffix", [".xlsx", ".csv", ".toml", ".py", ".xls", ".nc"])
def test_suffix_path_formatting(suffix, tmpdir):
    """Test path formatting with different file suffixes."""
    from muse.readers.toml import read_split_toml

    # Test path formatting
    settings = {"this": 0, "plugins": f"{{path}}/thisfile{suffix}"}
    input_file = tmpdir.join("settings.toml")
    with open(input_file, "w") as f:
        toml.dump(settings, f)

    result = read_split_toml(input_file, path=str(tmpdir))
    assert result["plugins"] == str(tmpdir / f"thisfile{suffix}")

    # Test cwd formatting
    settings["plugins"] = [f"{{cwd}}/other/thisfile{suffix}"]
    with open(input_file, "w") as f:
        toml.dump(settings, f)

    result = read_split_toml(input_file, path="hello")
    assert result["plugins"][0] == str(
        (Path() / "other" / f"thisfile{suffix}").absolute()
    )


def test_read_existing_trade(tmp_path):
    """Test reading existing trade data from CSV."""
    from muse.examples import copy_model
    from muse.readers.csv import read_trade

    copy_model("trade", tmp_path)
    path = tmp_path / "model" / "gas" / "ExistingTrade.csv"
    data = read_trade(path, skiprows=[1])

    assert isinstance(data, xr.DataArray)
    assert set(data.dims) == {"year", "technology", "dst_region", "region"}
    assert data.coords["year"].values.tolist() == [2010, 2020, 2030, 2040, 2050]
    assert data.coords["technology"].values.tolist() == ["gassupply1"]
    assert data.coords["dst_region"].values.tolist() == ["R1", "R2"]
    assert data.coords["region"].values.tolist() == ["R1", "R2"]


def test_read_trade_technodata(tmp_path):
    """Test reading trade technodata from CSV."""
    from muse.examples import copy_model
    from muse.readers.csv import read_trade

    copy_model("trade", tmp_path)
    path = tmp_path / "model" / "gas" / "TradeTechnodata.csv"
    data = read_trade(path, drop="Unit")

    expected_vars = {
        "cap_par",
        "cap_exp",
        "fix_par",
        "fix_exp",
        "max_capacity_addition",
        "max_capacity_growth",
        "total_capacity_limit",
    }

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"technology", "dst_region", "region"}
    assert set(data.data_vars) == expected_vars
    assert all(var.dtype == np.float64 for var in data.data_vars.values())
    assert data.coords["dst_region"].values.tolist() == ["R1", "R2"]
    assert data.coords["technology"].values.tolist() == ["gassupply1"]
    assert data.coords["region"].values.tolist() == ["R1", "R2", "R3"]
    assert all(var.coords.equals(data.coords) for var in data.data_vars.values())


@fixture
def default_model(tmp_path):
    from muse.examples import copy_model

    copy_model("default", tmp_path)
    return tmp_path / "model"


def test_read_technodictionary(default_model):
    """Test reading technology dictionary from CSV."""
    from muse.readers.csv import read_technodictionary

    path = default_model / "residential" / "Technodata.csv"
    data = read_technodictionary(path)

    expected_dtypes = {
        "cap_par": np.float64,
        "cap_exp": np.int64,
        "fix_par": np.int64,
        "fix_exp": np.int64,
        "var_par": np.int64,
        "interest_rate": np.float64,
        "type": np.dtype("O"),
        "agent1": np.int64,
        "tech_type": np.dtype("<U6"),
        "efficiency": np.int64,
        "max_capacity_addition": np.int64,
        "max_capacity_growth": np.float64,
        "scaling_size": np.float64,
        "technical_life": np.int64,
        "total_capacity_limit": np.int64,
        "utilization_factor": np.int64,
        "var_exp": np.int64,
    }

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"technology", "region"}
    assert dict(data.dtypes) == expected_dtypes
    assert data.coords["technology"].values.tolist() == ["gasboiler", "heatpump"]
    assert data.coords["region"].values.tolist() == ["R1"]

    # Check coordinate consistency
    for var in data.data_vars:
        if var == "tech_type":
            assert list(data.data_vars[var].coords) == ["technology"]
        else:
            assert data.data_vars[var].coords.equals(data.coords)


def test_read_technodata_timeslices(tmp_path):
    """Test reading technodata timeslices from CSV."""
    from muse.examples import copy_model
    from muse.readers.csv import read_technodata_timeslices
    from muse.timeslices import setup_module

    # Setup
    copy_model("default_timeslice", tmp_path)
    settings_path = tmp_path / "model" / "settings.toml"
    settings = toml.load(settings_path)
    setup_module(settings)

    # Read data
    data_path = tmp_path / "model" / "power" / "TechnodataTimeslices.csv"
    data = read_technodata_timeslices(data_path)

    # Verify structure
    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"technology", "region", "year", "timeslice"}
    assert dict(data.dtypes) == {
        "utilization_factor": np.int64,
        "minimum_service_factor": np.int64,
    }

    # Verify coordinates
    assert data.coords["technology"].values.tolist() == ["gasCCGT", "windturbine"]
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["year"].values.tolist() == [2020]

    # Verify timeslice components
    month_values = ["all-year"] * 6
    day_values = ["all-week"] * 6
    hour_values = [
        "night",
        "morning",
        "afternoon",
        "early-peak",
        "late-peak",
        "evening",
    ]

    assert data.coords["timeslice"].values.tolist() == list(
        zip(month_values, day_values, hour_values)
    )
    assert data.coords["month"].values.tolist() == month_values
    assert data.coords["day"].values.tolist() == day_values
    assert data.coords["hour"].values.tolist() == hour_values


def test_read_io_technodata(default_model):
    """Test reading input/output technodata from CSV."""
    from muse.readers.csv import read_io_technodata

    path = default_model / "residential" / "CommOut.csv"
    data = read_io_technodata(path)

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"technology", "region", "year", "commodity"}
    assert dict(data.dtypes) == {
        "fixed": np.float64,
        "flexible": np.float64,
        "commodity_units": np.dtype("O"),
    }

    # Verify coordinates
    assert data.coords["technology"].values.tolist() == ["gasboiler", "heatpump"]
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["year"].values.tolist() == [2020]
    assert set(data.coords["commodity"].values.tolist()) == set(COMMODITIES)

    # Check coordinate consistency
    assert data.data_vars["fixed"].coords.equals(data.coords)
    assert data.data_vars["flexible"].coords.equals(data.coords)
    assert list(data.data_vars["commodity_units"].coords) == ["commodity"]


def test_read_initial_assets(default_model):
    """Test reading initial assets from CSV."""
    from muse.readers.csv import read_initial_assets

    path = default_model / "residential" / "ExistingCapacity.csv"
    data = read_initial_assets(path)

    assert isinstance(data, xr.DataArray)
    assert set(data.dims) == {"region", "asset", "year"}
    assert data.dtype == np.int64

    # Verify coordinates
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["technology"].values.tolist() == ["gasboiler", "heatpump"]
    assert data.coords["installed"].values.tolist() == [2020, 2020]
    assert data.coords["year"].values.tolist() == list(range(2020, 2055, 5))


def test_global_commodities(default_model):
    """Test reading global commodities from CSV."""
    from muse.readers.csv import read_global_commodities

    path = default_model / "GlobalCommodities.csv"
    data = read_global_commodities(path)

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"commodity"}
    assert dict(data.dtypes) == {
        "comm_name": np.dtype("O"),
        "comm_type": np.dtype("O"),
        "emmission_factor": np.float64,
        "heat_rate": np.int64,
        "unit": np.dtype("O"),
    }

    assert data.coords["commodity"].values.tolist() == COMMODITIES
    assert all(var.coords.equals(data.coords) for var in data.data_vars.values())


def test_read_csv_agent_parameters(default_model):
    """Test reading agent parameters from CSV."""
    from muse.readers.csv import read_csv_agent_parameters

    path = default_model / "Agents.csv"
    data = read_csv_agent_parameters(path)

    expected_agent = {
        "name": "A1",
        "region": "R1",
        "objectives": ["LCOE"],
        "search_rules": "all",
        "decision": {"name": "singleObj", "parameters": [("LCOE", True, 1)]},
        "agent_type": "newcapa",
        "quantity": 1,
        "maturity_threshold": -1,
        "spend_limit": np.inf,
        "share": "agent1",
    }

    assert data == [expected_agent]


def test_read_initial_market(default_model):
    """Test reading initial market data from CSV."""
    from muse.readers.csv import read_initial_market

    path = default_model / "Projections.csv"
    data = read_initial_market(path)

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"region", "year", "commodity", "timeslice"}
    assert dict(data.dtypes) == {
        "prices": np.float64,
        "exports": np.float64,
        "imports": np.float64,
        "static_trade": np.float64,
    }

    # Verify coordinates
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["year"].values.tolist() == YEARS
    assert set(data.coords["commodity"].values.tolist()) == set(COMMODITIES)
    assert (
        data.coords["units_prices"].values.tolist()
        == ["MUS$2010/PJ"] * 3 + ["MUS$2010/kt"] * 2
    )

    # Verify timeslice components
    month_values = ["all-year"] * 6
    day_values = ["all-week"] * 6
    hour_values = [
        "night",
        "morning",
        "afternoon",
        "early-peak",
        "late-peak",
        "evening",
    ]

    assert data.coords["timeslice"].values.tolist() == list(
        zip(month_values, day_values, hour_values)
    )
    assert data.coords["month"].values.tolist() == month_values
    assert data.coords["day"].values.tolist() == day_values
    assert data.coords["hour"].values.tolist() == hour_values

    # Check coordinate consistency
    assert all(var.coords.equals(data.coords) for var in data.data_vars.values())


def test_read_attribute_table(default_model):
    """Test reading attribute table from CSV."""
    from muse.readers.csv import read_attribute_table

    path = default_model / "Projections.csv"
    data = read_attribute_table(path)

    assert isinstance(data, xr.DataArray)
    assert data.dtype == np.float64
    assert set(data.dims) == {"region", "year", "commodity"}

    # Verify coordinates
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["year"].values.tolist() == YEARS
    assert set(data.coords["commodity"].values.tolist()) == set(COMMODITIES)
    assert (
        data.coords["units_commodity_price"].values.tolist()
        == ["MUS$2010/PJ"] * 3 + ["MUS$2010/kt"] * 2
    )


def test_read_presets(default_model):
    """Test reading presets from CSV."""
    from muse.readers.csv import read_presets

    path = default_model / "residential_presets" / "*Consumption.csv"
    data = read_presets(str(path))

    assert isinstance(data, xr.DataArray)
    assert data.dtype == np.float64
    assert set(data.dims) == {"year", "commodity", "region", "timeslice"}

    # Verify coordinates
    assert data.coords["region"].values.tolist() == ["R1"]
    assert data.coords["timeslice"].values.tolist() == list(range(1, 7))
    assert data.coords["year"].values.tolist() == [2020, 2050]
    assert set(data.coords["commodity"].values.tolist()) == set(COMMODITIES)


def test_check_utilization_not_all_zero_success():
    """Test validation of non-zero utilization factors."""
    from muse.readers.csv import _check_utilization_not_all_zero

    df = pd.DataFrame(
        {
            "utilization_factor": [0, 1, 1],
            "technology": ["gas", "gas", "solar"],
            "region": ["GB", "GB", "FR"],
            "year": [2010, 2010, 2011],
        }
    )
    _check_utilization_not_all_zero(df, "file.csv")


def test_check_utilization_not_all_zero_fail():
    """Test validation fails when all utilization factors are zero."""
    from muse.readers.csv import _check_utilization_not_all_zero

    df = pd.DataFrame(
        {
            "utilization_factor": [0, 0, 1],
            "technology": ["gas", "gas", "solar"],
            "region": ["GB", "GB", "FR"],
            "year": [2010, 2010, 2011],
        }
    )
    with raises(ValueError):
        _check_utilization_not_all_zero(df, "file.csv")


def test_check_utilization_in_range_success():
    """Test validation of utilization factors within valid range."""
    from muse.readers.csv import _check_utilization_in_range

    df = pd.DataFrame({"utilization_factor": [0, 1]})
    _check_utilization_in_range(df, "file.csv")


@mark.parametrize(
    "values", chain.from_iterable(permutations((0, bad)) for bad in (-1, 2))
)
def test_check_utilization_in_range_fail(values):
    """Test validation fails for utilization factors outside valid range."""
    from muse.readers.csv import _check_utilization_in_range

    df = pd.DataFrame({"utilization_factor": values})
    with raises(ValueError):
        _check_utilization_in_range(df, "file.csv")


def test_check_utilization_and_minimum_service():
    """Test combined validation of utilization and minimum service factors."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    # Test valid case
    df = pd.DataFrame(
        {
            "utilization_factor": [0, 1],
            "minimum_service_factor": [0, 0],
            "technology": ["tech1", "tech1"],
            "region": ["R1", "R1"],
            "year": [2020, 2020],
        }
    )
    check_utilization_and_minimum_service_factors(df, "file.csv")

    # Test utilization below minimum
    df = pd.DataFrame(
        {
            "utilization_factor": [0, 1],
            "minimum_service_factor": [0.1, 0],
            "technology": ["tech1", "tech1"],
            "region": ["R1", "R1"],
            "year": [2020, 2020],
        }
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(df, "file.csv")

    # Test missing utilization factor
    df = pd.DataFrame(
        {"technology": ["tech1", "tech2"], "region": ["R1", "R2"], "year": [2020, 2021]}
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(df, "file.csv")


@mark.parametrize(
    "values", chain.from_iterable(permutations((0, bad)) for bad in (-1, 2))
)
def test_check_minimum_service_factors_in_range(values):
    """Test validation of minimum service factors within valid range."""
    from muse.readers.csv import _check_minimum_service_factors_in_range

    df = pd.DataFrame({"minimum_service_factor": values})
    with raises(ValueError):
        _check_minimum_service_factors_in_range(df, "file.csv")


@patch("muse.readers.csv._check_utilization_in_range")
@patch("muse.readers.csv._check_utilization_not_all_zero")
@patch("muse.readers.csv._check_utilization_not_below_minimum")
@patch("muse.readers.csv._check_minimum_service_factors_in_range")
def test_check_utilization_and_minimum_service_factors_mocked(*mocks):
    """Test all validation checks are called with correct parameters."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    df = pd.DataFrame(
        {"utilization_factor": [0, 0, 1], "minimum_service_factor": [0, 0, 0]}
    )
    check_utilization_and_minimum_service_factors(df, "file.csv")
    for mock in mocks:
        mock.assert_called_once_with(df, ["file.csv"])


@patch("muse.readers.csv._check_utilization_in_range")
@patch("muse.readers.csv._check_utilization_not_all_zero")
@patch("muse.readers.csv._check_utilization_not_below_minimum")
@patch("muse.readers.csv._check_minimum_service_factors_in_range")
def test_check_utilization_no_min_service(
    min_service_factor_mock, utilization_below_min_mock, *mocks
):
    """Test validation when minimum service factors are not present."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    df = pd.DataFrame({"utilization_factor": [0, 0, 1]})
    check_utilization_and_minimum_service_factors(df, "file.csv")

    for mock in mocks:
        mock.assert_called_once_with(df, ["file.csv"])
    min_service_factor_mock.assert_not_called()
    utilization_below_min_mock.assert_not_called()
