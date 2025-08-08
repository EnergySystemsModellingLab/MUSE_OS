from itertools import chain, permutations
from pathlib import Path

import pandas as pd
import toml
import xarray as xr
from pytest import fixture, mark, raises


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
    with raises(ValueError):
        check_iteration_control(settings)

    # Test invalid tolerance
    settings.update({"maximum_iterations": 5, "tolerance": -1})
    with raises(ValueError):
        check_iteration_control(settings)


def test_format_paths(tmp_path):
    """Test format_paths with various settings structures."""
    from muse.readers.toml import format_paths

    # Flat dict with path-like values
    settings = {
        "input_file": "{path}/data.csv",
        "script": "{cwd}/run.py",
        "not_a_path": "foo.txt",
        "filename": "{cwd}/special",
    }
    path = tmp_path
    cwd = tmp_path.parent
    result = format_paths(settings, path=path, cwd=cwd)
    assert result["input_file"] == (path / "data.csv").absolute()
    assert result["script"] == (cwd / "run.py").absolute()
    assert result["not_a_path"] == "foo.txt"
    assert result["filename"] == (cwd / "special").absolute()

    # Nested dict
    settings = {
        "level1": {
            "input_file": "{path}/nested.csv",
            "filename": "{cwd}/nested.py",
        },
        "other": 123,
    }
    result = format_paths(settings, path=path, cwd=cwd)
    assert result["level1"]["input_file"] == (path / "nested.csv").absolute()
    assert result["level1"]["filename"] == (cwd / "nested.py").absolute()
    assert result["other"] == 123

    # List of paths and non-paths
    settings = {
        "files": [
            "{path}/a.csv",
            "{cwd}/b.py",
            "not_a_path",
            {"input_file": "{cwd}/c.nc"},
        ]
    }
    result = format_paths(settings, path=path, cwd=cwd)
    assert result["files"][0] == (path / "a.csv").absolute()
    assert result["files"][1] == (cwd / "b.py").absolute()
    assert result["files"][2] == "not_a_path"
    assert result["files"][3]["input_file"] == (cwd / "c.nc").absolute()


@mark.parametrize("suffix", [".xlsx", ".csv", ".toml", ".py", ".xls", ".nc"])
def test_suffix_path_formatting(suffix, tmp_path):
    """Test path formatting with different file suffixes."""
    from muse.readers.toml import read_toml

    # Test path formatting
    settings = {"this": 0, "plugins": f"{{path}}/thisfile{suffix}"}
    input_file = tmp_path / "settings.toml"
    input_file.write_text(toml.dumps(settings), encoding="utf-8")

    result = read_toml(input_file, path=tmp_path)
    assert result["plugins"].resolve() == (tmp_path / f"thisfile{suffix}").resolve()

    # Test cwd formatting
    settings["plugins"] = [f"{{cwd}}/other/thisfile{suffix}"]
    input_file.write_text(toml.dumps(settings), encoding="utf-8")

    result = read_toml(input_file, path="hello")
    assert result["plugins"][0].resolve() == (
        (Path.cwd() / "other" / f"thisfile{suffix}").resolve()
    )


def test_check_utilization_and_minimum_service():
    """Test combined validation of utilization and minimum service factors."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    # Test valid case - create dataset with proper dimensions
    ds = xr.Dataset(
        {
            "utilization_factor": xr.DataArray(
                [[0.5, 0.5], [0.3, 0.7]],
                dims=["technology", "timeslice"],
                coords={"technology": ["tech1", "tech2"], "timeslice": [1, 2]},
            ),
            "minimum_service_factor": xr.DataArray(
                [[0.1, 0.1], [0.0, 0.0]],
                dims=["technology", "timeslice"],
                coords={"technology": ["tech1", "tech2"], "timeslice": [1, 2]},
            ),
        }
    )
    check_utilization_and_minimum_service_factors(ds)

    # Test utilization below minimum
    ds = xr.Dataset(
        {
            "utilization_factor": xr.DataArray(
                [[0.5, 0.5], [0.3, 0.7]],
                dims=["technology", "timeslice"],
                coords={"technology": ["tech1", "tech2"], "timeslice": [1, 2]},
            ),
            "minimum_service_factor": xr.DataArray(
                [[0.6, 0.1], [0.0, 0.0]],
                dims=["technology", "timeslice"],
                coords={"technology": ["tech1", "tech2"], "timeslice": [1, 2]},
            ),
        }
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(ds)

    # Test missing utilization factor
    ds = xr.Dataset(
        {
            "technology": xr.DataArray(
                ["tech1", "tech2"],
                dims=["technology"],
                coords={"technology": ["tech1", "tech2"]},
            ),
        }
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(ds)


def test_check_utilization_not_all_zero_fail():
    """Test validation fails when all utilization factors are zero."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    ds = xr.Dataset(
        {
            "utilization_factor": xr.DataArray(
                [[0.0, 0.0], [0.3, 0.7]],
                dims=["technology", "timeslice"],
                coords={"technology": ["tech1", "tech2"], "timeslice": [1, 2]},
            ),
        }
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(ds)


@mark.parametrize(
    "values", chain.from_iterable(permutations((0, bad)) for bad in (-1, 2))
)
def test_check_utilization_in_range_fail(values):
    """Test validation fails for utilization factors outside valid range."""
    from muse.readers.csv import check_utilization_and_minimum_service_factors

    ds = xr.Dataset(
        {
            "utilization_factor": xr.DataArray(
                list(values),
                dims=["timeslice"],
                coords={"timeslice": range(len(values))},
            ),
        }
    )
    with raises(ValueError):
        check_utilization_and_minimum_service_factors(ds)


def test_get_nan_coordinates():
    """Test get_nan_coordinates for various scenarios."""
    from muse.readers.csv import get_nan_coordinates

    # Test 1: Explicit NaN values
    df1 = pd.DataFrame(
        {
            "region": ["R1", "R1", "R2", "R2"],
            "year": [2020, 2021, 2020, 2021],
            "value": [1.0, float("nan"), 3.0, 4.0],
        }
    )
    dataset1 = xr.Dataset.from_dataframe(df1.set_index(["region", "year"]))
    nan_coords1 = get_nan_coordinates(dataset1)
    assert nan_coords1 == [("R1", 2021)]

    # Test 2: Missing coordinate combinations
    df2 = pd.DataFrame(
        {
            "region": ["R1", "R1", "R2"],  # Missing R2-2021
            "year": [2020, 2021, 2020],
            "value": [1.0, 2.0, 3.0],
        }
    )
    dataset2 = xr.Dataset.from_dataframe(df2.set_index(["region", "year"]))
    nan_coords2 = get_nan_coordinates(dataset2)
    assert nan_coords2 == [("R2", 2021)]

    # Test 3: No NaN values
    df3 = pd.DataFrame(
        {
            "region": ["R1", "R1", "R2", "R2"],
            "year": [2020, 2021, 2020, 2021],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    dataset3 = xr.Dataset.from_dataframe(df3.set_index(["region", "year"]))
    nan_coords3 = get_nan_coordinates(dataset3)
    assert nan_coords3 == []
