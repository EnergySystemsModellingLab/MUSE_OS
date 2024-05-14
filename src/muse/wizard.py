import os

import pandas as pd
import toml


def modify_toml(path_to_toml, function):
    """Apply the specified function to modify a toml file

    Args:
        path_to_toml: Path to the toml file
        function: Function to apply to the toml data. Must take a dictionary as a single
            input.

    """
    with open(path_to_toml, "r") as file:
        data = toml.load(file)
    function(data)
    with open(path_to_toml, "w") as file:
        toml.dump(data, file)


def get_sectors(model_path: str) -> list[str]:
    """Get a list of sector names for a model.

    Args:
        model_path: Path to the model folder.

    Returns:
        List of sector names
    """
    return [
        s
        for s in os.listdir(os.path.join(model_path, "technodata"))
        if os.path.isfile(os.path.join(model_path, "technodata", s, "technodata.csv"))
    ]


def add_new_commodity(
    model_path: str, commodity_name: str, sector: str, copy_from: str
) -> None:
    """Add a new commodity to a sector by copying an existing one.

    Args:
        model_path: Path to the model folder.
        commodity_name: Name of the new commodity.
        sector: Sector to add the commodity to.
        copy_from: Name of the commodity to copy from.
    """
    files_to_update = [
        os.path.join(model_path, file)
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

    global_commodities_file = os.path.join(model_path, "input/GlobalCommodities.csv")
    df = pd.read_csv(global_commodities_file)
    new_rows = df[df["Commodity"] == copy_from.capitalize()].copy()
    new_rows["Commodity"] = commodity_name.capitalize()
    new_rows["CommodityName"] = commodity_name
    df = pd.concat([df, new_rows])
    df.to_csv(global_commodities_file, index=False)


def add_new_process(
    model_path: str, process_name: str, sector: str, copy_from: str
) -> None:
    """Add a new process to a sector by copying an existing one.

    Args:
        model_path: Path to the model folder.
        process_name: Name of the new process.
        sector: Sector to add the process to.
        copy_from: Name of the process to copy from.
    """
    files_to_update = [
        os.path.join(model_path, file)
        for file in [
            f"technodata/{sector}/CommIn.csv",
            f"technodata/{sector}/CommOut.csv",
            f"technodata/{sector}/ExistingCapacity.csv",
            f"technodata/{sector}/technodata.csv",
        ]
    ]

    for file in files_to_update:
        df = pd.read_csv(file)
        new_rows = df[df["ProcessName"] == copy_from].copy()
        new_rows["ProcessName"] = process_name
        df = pd.concat([df, new_rows])
        df.to_csv(file, index=False)


def add_price_data_for_new_year(
    model_path: str, year: str, sector: str, copy_from: str
) -> None:
    """Add price data for a new year by copying from an existing year.

    Args:
        model_path: Path to the model folder.
        year: Year to add the price data to.
        sector: Sector to add the price data to.
        copy_from: Year to copy the price data from.
    """
    files_to_update = [
        os.path.join(model_path, f"technodata/{sector}/{file}")
        for file in ["technodata.csv", "CommIn.csv", "CommOut.csv"]
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
    model_path: str,
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
    agents_file = os.path.join(model_path, "technodata/Agents.csv")
    df = pd.read_csv(agents_file)
    new_rows = df[df["Name"] == copy_from].copy()
    new_rows["Name"] = agent_name
    new_rows.loc[new_rows["Type"] == "New", "AgentShare"] = agentshare_new
    new_rows.loc[new_rows["Type"] == "Retrofit", "AgentShare"] = agentshare_retrofit
    df = pd.concat([df, new_rows])
    df.to_csv(agents_file, index=False)

    copy_from_new = df.loc[
        (df["Name"] == copy_from) & (df["Type"] == "New"), "AgentShare"
    ].values[0]
    copy_from_retrofit = df.loc[
        (df["Name"] == copy_from) & (df["Type"] == "Retrofit"), "AgentShare"
    ].values[0]

    for sector in get_sectors(model_path):
        technodata_file = os.path.join(
            model_path, f"technodata/{sector}/technodata.csv"
        )
        df = pd.read_csv(technodata_file)
        if copy_from_retrofit in df.columns:
            df[agentshare_retrofit] = df[copy_from_retrofit]
        if copy_from_new in df.columns:
            df[agentshare_new] = df[copy_from_new]
        df.to_csv(technodata_file, index=False)
