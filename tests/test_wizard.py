import pandas as pd
import pytest
from muse import examples
from muse.wizard import (
    add_agent,
    add_new_commodity,
    add_new_process,
    add_price_data_for_new_year,
    add_region,
    add_timeslice,
    get_sectors,
    modify_toml,
)
from tomlkit import dumps, parse


@pytest.fixture
def model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default", path=tmp_path)
    return tmp_path / "model"


def test_modify_toml(tmp_path):
    """Test the modify_toml function."""
    # Create a temporary toml file
    toml_path = tmp_path / "temp.toml"

    # Create initial toml data
    initial_data = {"name": "Tomm", "age": 299}

    # Write initial toml data to the temporary file
    toml_path.write_text(dumps(initial_data))

    # Define the function to modify the toml data
    def modify_function(data):
        data["name"] = "Tom"
        data["age"] = 29

    # Call the modify_toml function
    modify_toml(toml_path, modify_function)

    # Read the modified toml data
    modified_data = parse(toml_path.read_text())

    # Assert that the modifications were applied correctly
    assert modified_data["name"] == "Tom"
    assert modified_data["age"] == 29


def test_get_sectors(tmp_path):
    """Test the get_sectors function."""
    # Create a temporary model folder
    model_path = tmp_path / "model"
    model_path.mkdir()

    # Create some sector folders with Technodata.csv files
    sector1 = model_path / "technodata" / "sector1"
    sector1.mkdir(parents=True)
    (sector1 / "Technodata.csv").touch()

    sector2 = model_path / "technodata" / "sector2"
    sector2.mkdir(parents=True)
    (sector2 / "Technodata.csv").touch()

    sector3 = model_path / "technodata" / "sector3"
    sector3.mkdir(parents=True)

    # Call the get_sectors function
    sectors = get_sectors(model_path)

    # Check the returned sectors
    assert set(sectors) == {"sector1", "sector2"}


def test_add_new_commodity(model_path):
    """Test the add_new_commodity function on the default model."""
    add_new_commodity(model_path, "new_commodity", "power", "wind")

    # Check if the new commodity is added to the global commodities file
    global_commodities_file = model_path / "input/GlobalCommodities.csv"
    df = pd.read_csv(global_commodities_file)
    assert "new_commodity" in df["CommodityName"].values

    # Check if the new column is added to additional files
    files_to_check = [
        model_path / file
        for file in [
            "technodata/power/CommIn.csv",
            "technodata/power/CommOut.csv",
            "input/BaseYearImport.csv",
            "input/BaseYearExport.csv",
            "input/Projections.csv",
        ]
    ] + list((model_path / "technodata/preset").glob("*"))
    for file in files_to_check:
        df = pd.read_csv(model_path / file)
        assert "new_commodity" in df.columns


def test_add_new_process(model_path):
    """Test the add_new_process function on the default model."""
    add_new_process(model_path, "new_process", "power", "windturbine")

    # Check if the new process is added to the files
    files_to_check = [
        "technodata/power/CommIn.csv",
        "technodata/power/CommOut.csv",
        "technodata/power/ExistingCapacity.csv",
        "technodata/power/Technodata.csv",
    ]
    for file in files_to_check:
        df = pd.read_csv(model_path / file)
        assert "new_process" in df["ProcessName"].values


def test_add_price_data_for_new_year(model_path):
    """Test the add_price_data_for_new_year function on the default model."""
    add_price_data_for_new_year(model_path, "2030", "power", "2020")

    # Check if the new price data is added to the files
    files_to_check = [
        "technodata/power/Technodata.csv",
        "technodata/power/CommIn.csv",
        "technodata/power/CommOut.csv",
    ]
    for file in files_to_check:
        df = pd.read_csv(model_path / file)
        assert "2030" in df["Time"].values


def test_add_agent(model_path):
    """Test the add_agent function on the default model."""
    add_agent(model_path, "A2", "A1", "Agent3", "Agent4")

    # Check if the new agent is added to the Agents.csv file
    df = pd.read_csv(model_path / "technodata/Agents.csv")
    assert "A2" in df["Name"].values
    assert "Agent3" in df["AgentShare"].values
    assert "Agent4" in df["AgentShare"].values

    # Check if the retrofit agent is added to the Technodata.csv files
    sector1_file = model_path / "technodata/power/Technodata.csv"
    sector2_file = model_path / "technodata/gas/Technodata.csv"
    df_sector1 = pd.read_csv(sector1_file)
    df_sector2 = pd.read_csv(sector2_file)
    assert "Agent4" in df_sector1.columns
    assert "Agent4" in df_sector2.columns


def test_add_region(model_path):
    """Test the add_region function on the default model."""
    add_region(model_path, "R2", "R1")

    # Check if the new region is added to the settings.toml file
    with open(model_path / "settings.toml", "r") as f:
        modified_settings_data = parse(f.read())
    assert "R2" in modified_settings_data["regions"]

    # Check if the new region is added to the technodata files
    sector_files = [
        model_path / "technodata" / sector / file
        for sector in get_sectors(model_path)
        for file in [
            "Technodata.csv",
            "CommIn.csv",
            "CommOut.csv",
            "ExistingCapacity.csv",
        ]
    ]
    for file in sector_files:
        df = pd.read_csv(file)
        assert "R2" in df["RegionName"].values


def test_add_timeslice(model_path):
    """Test the add_timeslice function on the default model."""
    add_timeslice(model_path, "midnight", "evening")

    # Check if the new timeslice is added to the settings.toml file
    with open(model_path / "settings.toml", "r") as f:
        modified_settings_data = parse(f.read())
    assert "midnight" in modified_settings_data["timeslices"]["all-year"]["all-week"]
    n_timeslices = len(modified_settings_data["timeslices"]["all-year"]["all-week"])

    # Check if the new timeslice is added to the preset files
    df_preset1 = pd.read_csv(
        model_path / "technodata/preset/Residential2020Consumption.csv"
    )
    df_preset2 = pd.read_csv(
        model_path / "technodata/preset/Residential2050Consumption.csv"
    )
    assert len(df_preset1["Timeslice"].unique()) == n_timeslices
    assert len(df_preset2["Timeslice"].unique()) == n_timeslices
