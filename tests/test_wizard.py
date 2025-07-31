from pathlib import Path

import pandas as pd
import pytest
from tomlkit import dumps, parse

from muse import examples
from muse.wizard import (
    add_agent,
    add_new_commodity,
    add_new_process,
    add_region,
    add_technodata_for_new_year,
    add_timeslice,
    get_sectors,
    modify_toml,
)


@pytest.fixture
def model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default", path=tmp_path)
    return tmp_path / "model"


@pytest.fixture
def model_path_retro(tmp_path):
    """Creates temporary folder containing the default_retro model."""
    examples.copy_model(name="default_retro", path=tmp_path)
    return tmp_path / "model"


def assert_values_in_csv(file_path: Path, column: str, expected_values: list):
    """Helper function to check if values exist in a CSV column."""
    df = pd.read_csv(file_path)
    for value in expected_values:
        assert value in df[column].values


def assert_columns_exist(file_path: Path, columns: list):
    """Helper function to check if columns exist in a CSV file."""
    df = pd.read_csv(file_path)
    for column in columns:
        assert column in df.columns


def test_modify_toml(tmp_path):
    """Test the modify_toml function."""
    toml_path = tmp_path / "temp.toml"
    initial_data = {"name": "Tomm", "age": 299}
    toml_path.write_text(dumps(initial_data))

    def modify_function(data):
        data.update({"name": "Tom", "age": 29})

    modify_toml(toml_path, modify_function)
    modified_data = parse(toml_path.read_text())

    assert modified_data == {"name": "Tom", "age": 29}


def test_get_sectors(tmp_path):
    """Test the get_sectors function."""
    model_path = tmp_path / "model"
    model_path.mkdir()

    # Create test sector folders
    for sector in ["sector1", "sector2", "sector3"]:
        sector_path = model_path / sector
        sector_path.mkdir(parents=True)
        if sector != "sector3":
            (sector_path / "Technodata.csv").touch()

    assert set(get_sectors(model_path)) == {"sector1", "sector2"}


def test_add_new_commodity(model_path):
    """Test the add_new_commodity function on the default model."""
    add_new_commodity(model_path, "new_commodity", "power", "wind")

    # Check global commodities
    assert_values_in_csv(
        model_path / "GlobalCommodities.csv", "commodity", ["new_commodity"]
    )


def test_add_new_process(model_path):
    """Test the add_new_process function on the default model."""
    add_new_process(model_path, "new_process", "power", "windturbine")

    files_to_check = [
        "CommIn.csv",
        "CommOut.csv",
        "ExistingCapacity.csv",
        "Technodata.csv",
    ]
    for file in files_to_check:
        assert_values_in_csv(model_path / "power" / file, "technology", ["new_process"])


def test_technodata_for_new_year(model_path):
    """Test the add_price_data_for_new_year function on the default model."""
    add_technodata_for_new_year(model_path, 2030, "power", 2020)

    assert_values_in_csv(model_path / "power" / "Technodata.csv", "year", [2030])


def test_add_agent(model_path_retro):
    """Test the add_agent function on the default_retro model."""
    add_agent(model_path_retro, "A2", "A1", "Agent3", "Agent4")

    # Check Agents.csv
    assert_values_in_csv(model_path_retro / "Agents.csv", "name", ["A2"])
    for share in ["Agent3", "Agent4"]:
        assert_values_in_csv(model_path_retro / "Agents.csv", "agent_share", [share])

    # Check Technodata.csv files
    for sector in ["power", "gas"]:
        assert_columns_exist(model_path_retro / sector / "Technodata.csv", ["Agent4"])


def test_add_region(model_path):
    """Test the add_region function on the default model."""
    add_region(model_path, "R2", "R1")

    # Check settings.toml
    with open(model_path / "settings.toml") as f:
        settings = parse(f.read())
        assert "R2" in settings["regions"]

    # Check sector files
    files_to_check = [
        "Technodata.csv",
        "CommIn.csv",
        "CommOut.csv",
        "ExistingCapacity.csv",
    ]
    for sector in get_sectors(model_path):
        for file in files_to_check:
            assert_values_in_csv(model_path / sector / file, "region", ["R2"])


def test_add_timeslice(model_path):
    """Test the add_timeslice function on the default model."""
    add_timeslice(model_path, "midnight", "evening")

    # Check settings.toml
    with open(model_path / "settings.toml") as f:
        settings = parse(f.read())
        timeslices = settings["timeslices"]["all-year"]["all-week"]
        assert "midnight" in timeslices
        n_timeslices = len(timeslices)

    # Check preset files
    for preset in ["Residential2020Consumption.csv", "Residential2050Consumption.csv"]:
        df = pd.read_csv(model_path / "residential_presets" / preset)
        assert len(df["timeslice"].unique()) == n_timeslices
