from pathlib import Path

import toml
import xarray as xr
from pytest import fixture, mark, raises


@fixture
def user_data_files(settings: dict) -> None:
    """Creates the files related to the user data."""

    files = list(settings["global_input_files"].keys())
    files.remove("path")
    for m in files:
        if settings["global_input_files"][m] == "":
            settings["global_input_files"].pop(m)
            continue
        new_file = Path(settings["global_input_files"][m])
        new_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.write_text("Some data")


@fixture
def sectors_files(settings: dict):
    """Creates the files related to the sector."""
    from typing import Text

    for data in settings["sectors"].values():
        for path in data.values():
            if not isinstance(path, (Path, Text)):
                continue
            path = Path(path)
            if path.suffix != ".csv":
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Some data")


@fixture
def plugins(settings: dict, tmp_path) -> Path:
    """Creates the files related to the custom modules."""

    plugin = tmp_path / "plugins" / "cat.py"
    plugin.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_text("my_cat_colour = 'tabby' ")

    settings["plugins"] = str(plugin)
    return plugin


@fixture
def input_file(settings: dict, tmpdir, plugins, user_data_files, sectors_files) -> Path:
    """Creates a whole set of MUSE input files in a temporary directory.

    This fixture creates a temporal directory with all the folders and files required
    for a successful run of the read_settings function.
    """
    # Finally we create the settings file
    input_file = tmpdir.join("settings.toml")
    with open(input_file, "w") as f:
        toml.dump(settings, f)

    return input_file


def test_add_known_parameters(settings: dict):
    """Test the add_known_parameters function.

    This test checks the following things:
        - That missing required parameters raise an exception
        - That missing optional parameters do not get a default value
        - That missing any other parameter gets a default value
    """
    from copy import copy

    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import MissingSettings, add_known_parameters

    defaults = toml.load(DEFAULT_SETTINGS_PATH)

    # REQUIRED parameters raise exception
    missing_settings = copy(settings)
    missing_settings.pop("time_framework")
    with raises(MissingSettings):
        add_known_parameters(defaults, missing_settings)

    # Others get a default value. We check a couple of them
    optional_settings = copy(settings)
    optional_settings.pop("global_input_files")
    new_settings = add_known_parameters(defaults, optional_settings)
    assert "projections" in new_settings["global_input_files"].keys()


def test_add_unknown_parameters(settings: dict):
    """Test the add_unknown_parameters function."""
    from copy import copy

    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import add_unknown_parameters

    defaults = toml.load(DEFAULT_SETTINGS_PATH)

    extra_settings = copy(settings)
    extra_settings["my_new_parameter"] = 42
    new_settings = add_unknown_parameters(defaults, extra_settings)
    assert "my_new_parameter" in new_settings.keys()


def test_check_log_level(settings: dict):
    """Tests the check_interpolation_mode function."""
    from muse.readers.toml import check_log_level

    check_log_level(settings)


def test_check_interpolation_mode(settings: dict):
    """Tests the check_interpolation_mode function."""
    from muse.readers.toml import check_interpolation_mode

    check_interpolation_mode(settings)


def test_check_budget_parameters(settings: dict):
    """Tests the check_budget_parameters function."""
    from muse.readers.toml import check_budget_parameters

    check_budget_parameters(settings)


def test_check_foresight(settings: dict):
    """Tests the check_budget_parameters function."""
    from muse.readers.toml import check_foresight

    check_foresight(settings)


def test_check_time_slices(settings: dict):
    """Tests the check_budget_parameters function."""
    from muse.readers.toml import check_time_slices

    check_time_slices(settings)


def test_check_global_data_files(settings: dict, user_data_files):
    """Tests the check_global_data_files function."""
    from muse.readers.toml import check_global_data_files

    check_global_data_files(settings)

    new_file = Path(settings["global_input_files"]["projections"])
    new_file.rename(new_file.parent / "my_file")
    with raises(AssertionError):
        check_global_data_files(settings)


def test_check_global_data_dir(settings: dict, user_data_files):
    """Tests the check_global_data_files function."""
    from muse.readers.toml import check_global_data_files

    check_global_data_files(settings)

    path = Path(settings["global_input_files"]["path"])
    path.rename(path.parent / "my_directory")
    with raises(AssertionError):
        check_global_data_files(settings)


def test_check_plugins(settings: dict, plugins: Path):
    from muse.readers.toml import IncorrectSettings, check_plugins

    # Now we run check_plugins, which should succeed in finding the files
    check_plugins(settings)

    # Now we change the name of the module and check if there's an exception
    settings["plugins"] = plugins.parent / f"{plugins.stem}_2{plugins.suffix}"
    with raises(IncorrectSettings):
        check_plugins(settings)


def test_check_iteration_control(settings: dict):
    """Tests the whole loading settings function."""
    from muse.readers.toml import check_iteration_control

    settings["equilibrium"] = "off"
    settings["maximum_iterations"] = 7
    check_iteration_control(settings)

    assert not settings["equilibrium"]

    settings["equilibrium"] = True
    settings["maximum_iterations"] = -1
    with raises(AssertionError):
        check_iteration_control(settings)

    settings["maximum_iterations"] = 5
    settings["tolerance"] = -1
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
    from muse.readers.toml import read_split_toml
    from toml import dumps

    (tmpdir / "outer.toml").write(
        dumps(
            {
                "path": str(tmpdir),
                "some_section": {"include_path": "{path}/inner.toml"},
                "another_section": {"option_a": "a"},
            }
        )
    )

    (tmpdir / "inner.toml").write(dumps({"some_section": {"my_option": "found it!"}}))

    result = read_split_toml(tmpdir / "outer.toml")
    assert set(result) == {"path", "some_section", "another_section"}
    assert set(result["another_section"]) == {"option_a"}
    assert result["another_section"]["option_a"] == "a"
    assert set(result["some_section"]) == {"my_option"}
    assert result["some_section"]["my_option"] == "found it!"


def test_split_toml_nested(tmpdir):
    from muse.readers.toml import read_split_toml
    from toml import dumps

    (tmpdir / "outer.toml").write(
        dumps(
            {
                "path": str(tmpdir),
                "some_section": {"nested": {"include_path": "{path}/inner.toml"}},
                "another_section": {"option_a": "a"},
            }
        )
    )

    (tmpdir / "inner.toml").write(dumps({"nested": {"my_option": "found it!"}}))

    result = read_split_toml(tmpdir / "outer.toml")
    assert set(result) == {"path", "some_section", "another_section"}
    assert set(result["another_section"]) == {"option_a"}
    assert result["another_section"]["option_a"] == "a"
    assert set(result["some_section"]) == {"nested"}
    assert set(result["some_section"]["nested"]) == {"my_option"}
    assert result["some_section"]["nested"]["my_option"] == "found it!"


def test_split_toml_too_manyops_in_outer(tmpdir):
    from muse.readers.toml import IncorrectSettings, read_split_toml
    from pytest import raises
    from toml import dumps

    (tmpdir / "outer.toml").write(
        dumps(
            {
                "path": str(tmpdir),
                "some_section": {
                    "nested": {"include_path": "{path}/inner.toml", "extra": "error"}
                },
                "another_section": {"option_a": "a"},
            }
        )
    )

    (tmpdir / "inner.toml").write(dumps({"nested": {"my_option": "found it!"}}))

    with raises(IncorrectSettings):
        read_split_toml(tmpdir / "outer.toml")


def test_split_toml_too_manyops_in_inner(tmpdir):
    from muse.readers.toml import IncorrectSettings, read_split_toml
    from pytest import raises
    from toml import dumps

    (tmpdir / "outer.toml").write(
        dumps(
            {
                "path": str(tmpdir),
                "some_section": {"nested": {"include_path": "{path}/inner.toml"}},
                "another_section": {"option_a": "a"},
            }
        )
    )

    (tmpdir / "inner.toml").write(
        dumps({"extra": "error", "nested": {"my_option": "found it!"}})
    )

    with raises(IncorrectSettings):
        read_split_toml(tmpdir / "outer.toml")


def test_split_toml_incorrect_inner_name(tmpdir):
    from muse.readers.toml import MissingSettings, read_split_toml
    from pytest import raises
    from toml import dumps

    (tmpdir / "outer.toml").write(
        dumps(
            {
                "path": str(tmpdir),
                "some_section": {"nested": {"include_path": "{path}/inner.toml"}},
                "another_section": {"option_a": "a"},
            }
        )
    )

    (tmpdir / "inner.toml").write(dumps({"incorrect_name": {"my_option": "found it!"}}))

    with raises(MissingSettings):
        read_split_toml(tmpdir / "outer.toml")


def test_format_path():
    from muse.readers.toml import format_path

    path = "this_path"
    cwd = "current_path"
    muse_sectors = "sectors_path"

    assert format_path("{cwd}/{other_param}", cwd=cwd) == str(
        Path(cwd).absolute() / "{other_param}"
    )
    assert format_path("{path}/{other_param}", path=path) == str(
        Path(path).absolute() / "{other_param}"
    )
    assert format_path(
        "{muse_sectors}/{other_param}", muse_sectors=muse_sectors
    ) == str(Path(muse_sectors).absolute() / "{other_param}")


@mark.parametrize("suffix", (".xlsx", ".csv", ".toml", ".py", ".xls", ".nc"))
def test_suffix_path_formatting(suffix, tmpdir):
    from muse.readers.toml import read_split_toml

    settings = {"this": 0, "plugins": f"{{path}}/thisfile{suffix}"}
    input_file = tmpdir.join("settings.toml")
    with open(input_file, "w") as f:
        toml.dump(settings, f)

    result = read_split_toml(input_file, path=str(tmpdir))
    assert result["plugins"] == str(tmpdir / f"thisfile{suffix}")

    settings["plugins"] = [f"{{cwd}}/other/thisfile{suffix}"]
    input_file = tmpdir.join("settings.toml")
    with open(input_file, "w") as f:
        toml.dump(settings, f)

    result = read_split_toml(input_file, path="hello")
    assert result["plugins"][0] == str(
        (Path() / "other" / f"thisfile{suffix}").absolute()
    )


def test_read_existing_trade(tmp_path):
    from muse.examples import copy_model
    from muse.readers.csv import read_trade

    copy_model("trade", tmp_path)
    path = tmp_path / "model" / "technodata" / "gas" / "ExistingTrade.csv"
    data = read_trade(path, skiprows=[1])

    assert isinstance(data, xr.DataArray)
    assert set(data.dims) == {"year", "technology", "dst_region", "region"}


def test_read_trade_technodata(tmp_path):
    from muse.examples import copy_model
    from muse.readers.csv import read_trade

    copy_model("trade", tmp_path)
    path = tmp_path / "model" / "technodata" / "gas" / "TradeTechnodata.csv"
    data = read_trade(path, drop="Unit")

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"technology", "dst_region", "region"}
    assert set(data.data_vars) == {
        "cap_par",
        "cap_exp",
        "fix_par",
        "fix_exp",
        "max_capacity_addition",
        "max_capacity_growth",
        "total_capacity_limit",
    }
