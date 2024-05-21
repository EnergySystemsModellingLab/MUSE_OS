import os
from pathlib import Path
from typing import Callable, List

import pandas as pd
from tomlkit import dumps, parse


def modify_toml(path_to_toml: Path, function: Callable):
    """Apply the specified function to modify a toml file.

    Args:
        path_to_toml: Path to the toml file
        function: Function to apply to the toml data. Must take a dictionary as a single
            input and modify it in place.

    """
    data = parse(path_to_toml.read_text())
    function(data)
    path_to_toml.write_text(dumps(data))


def get_sectors(model_path: Path) -> List[str]:
    """Get a list of sector names for a model.

    Args:
        model_path: Path to the model folder.

    Returns:
        List of sector names
    """
    return [
        s.name
        for s in (model_path / "technodata").iterdir()
        if (s / "Technodata.csv").is_file()
    ]


def add_new_commodity(
    model_path: Path, commodity_name: str, sector: str, copy_from: str
) -> None:
    """Add a new commodity to a sector by copying an existing one.

    Args:
        model_path: Path to the model folder.
        commodity_name: Name of the new commodity.
        sector: Sector to add the commodity to.
        copy_from: Name of the commodity to copy from.
    """
    files_to_update = [
        model_path / file
        for file in [
            f"technodata/{sector}/CommIn.csv",
            f"technodata/{sector}/CommOut.csv",
            "input/BaseYearImport.csv",
            "input/BaseYearExport.csv",
            "input/Projections.csv",
        ]
    ]

    for file in files_to_update:
        df = pd.read_csv(file)
        df[commodity_name] = df[copy_from]
        df.to_csv(file, index=False)

    global_commodities_file = model_path / "input/GlobalCommodities.csv"
    df = pd.read_csv(global_commodities_file)
    new_rows = df[df["Commodity"] == copy_from.capitalize()].assign(
        Commodity=commodity_name.capitalize(), CommodityName=commodity_name
    )
    df = pd.concat([df, new_rows])
    df.to_csv(global_commodities_file, index=False)

    # Add to projections
    for file in (model_path / "technodata/preset").glob("*"):
        df = pd.read_csv(file)
        df[commodity_name] = df[copy_from]
        df.to_csv(file, index=False)


def add_new_process(
    model_path: Path, process_name: str, sector: str, copy_from: str
) -> None:
    """Add a new process to a sector by copying an existing one.

    Args:
        model_path: Path to the model folder.
        process_name: Name of the new process.
        sector: Sector to add the process to.
        copy_from: Name of the process to copy from.
    """
    files_to_update = [
        model_path / file
        for file in [
            f"technodata/{sector}/CommIn.csv",
            f"technodata/{sector}/CommOut.csv",
            f"technodata/{sector}/ExistingCapacity.csv",
            f"technodata/{sector}/Technodata.csv",
        ]
    ]

    for file in files_to_update:
        df = pd.read_csv(file)
        new_rows = df[df["ProcessName"] == copy_from].copy()
        new_rows["ProcessName"] = process_name
        df = pd.concat([df, new_rows])
        df.to_csv(file, index=False)


def add_price_data_for_new_year(
    model_path: Path, year: str, sector: str, copy_from: str
) -> None:
    """Add price data for a new year by copying from an existing year.

    Args:
        model_path: Path to the model folder.
        year: Year to add the price data to.
        sector: Sector to add the price data to.
        copy_from: Year to copy the price data from.
    """
    files_to_update = [
        model_path / f"technodata/{sector}/{file}"
        for file in ["Technodata.csv", "CommIn.csv", "CommOut.csv"]
    ]

    for file in files_to_update:
        df = pd.read_csv(file)
        df["index"] = df.index
        new_rows = df[df["Time"] == copy_from].copy()
        new_rows["Time"] = year
        df = pd.concat([df, new_rows], ignore_index=True)
        df.sort_values(by=["index", "Time"], inplace=True)
        df.drop(columns=["index"], inplace=True)
        df.to_csv(file, index=False)


def add_agent(
    model_path: Path,
    agent_name: str,
    copy_from: str,
    agentshare_new: str,
    agentshare_retrofit: str,
) -> None:
    """Add a new agent to the model by copying an existing one.

    Args:
        model_path: Path to the model folder.
        agent_name: Name of the new agent.
        copy_from: Name of the agent to copy from.
        agentshare_new: Name of the new agent share for new agent.
        agentshare_retrofit: Name of the retrofit agent share for new agent.
    """
    agents_file = model_path / "technodata/Agents.csv"
    df = pd.read_csv(agents_file)
    copy_from_new = df.loc[
        (df["Name"] == copy_from) & (df["Type"] == "New"), "AgentShare"
    ].values[0]
    copy_from_retrofit = df.loc[
        (df["Name"] == copy_from) & (df["Type"] == "Retrofit"), "AgentShare"
    ].values[0]

    new_rows = df[df["Name"] == copy_from].copy()
    new_rows["Name"] = agent_name
    new_rows.loc[new_rows["Type"] == "New", "AgentShare"] = agentshare_new
    new_rows.loc[new_rows["Type"] == "Retrofit", "AgentShare"] = agentshare_retrofit
    df = pd.concat([df, new_rows])
    df.to_csv(agents_file, index=False)

    for sector in get_sectors(model_path):
        technodata_file = model_path / f"technodata/{sector}/Technodata.csv"
        df = pd.read_csv(technodata_file)
        if copy_from_retrofit in df.columns:
            df[agentshare_retrofit] = df[copy_from_retrofit]
        if copy_from_new in df.columns:
            df[agentshare_new] = df[copy_from_new]
        df.to_csv(technodata_file, index=False)


def add_region(model_path: Path, region_name: str, copy_from: str) -> None:
    """Add a new region to the MUSE model.

    Args:
        model_path: The path to the MUSE model directory.
        region_name: The name of the new region to be added.
        copy_from: The name of the region to copy data from.
    """
    # Append region to settings.toml
    settings_file = model_path / "settings.toml"
    modify_toml(settings_file, lambda x: x["regions"].append(region_name))

    # Modify csv files
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
    preset_files = [
        model_path / "technodata" / "preset" / file
        for file in os.listdir(model_path / "technodata" / "preset")
    ]
    global_files = [
        model_path / file
        for file in [
            "technodata/Agents.csv",
            "input/BaseYearImport.csv",
            "input/BaseYearExport.csv",
            "input/Projections.csv",
        ]
    ]
    for file_path in sector_files + preset_files + global_files:
        df = pd.read_csv(file_path)
        new_rows = df[df["RegionName"] == copy_from].copy()
        new_rows["RegionName"] = region_name
        df = pd.concat([df, new_rows])
        df.to_csv(file_path, index=False)


def add_timeslice(model_path: Path, timeslice_name: str, copy_from: str) -> None:
    """Add a new timeslice to the model.

    Args:
        model_path: The path to the model directory.
        timeslice_name: The name of the new timeslice.
        copy_from: The name of the timeslice to copy from.
    """
    # Append timeslice to timeslices in settings.toml
    settings_file = model_path / "settings.toml"
    settings = parse(settings_file.read_text())
    timeslices = settings["timeslices"]["all-year"]["all-week"]
    copy_from_number = list(timeslices).index(copy_from) + 1
    timeslices[timeslice_name] = timeslices[copy_from]
    settings_file.write_text(dumps(settings))

    # Loop through all preset files
    preset_dir = model_path / "technodata" / "preset"
    for file_name in os.listdir(preset_dir):
        file_path = preset_dir / file_name
        df = pd.read_csv(file_path)
        new_rows = df[df["Timeslice"] == copy_from_number].copy()
        new_rows["Timeslice"] = len(timeslices)
        df = pd.concat([df, new_rows])
        df = df.sort_values(by=["RegionName", "Timeslice"]).reset_index(drop=True)
        df.to_csv(file_path, index=False)
