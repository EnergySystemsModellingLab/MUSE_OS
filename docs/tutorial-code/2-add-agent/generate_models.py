import shutil
from pathlib import Path

import pandas as pd
from muse.wizard import add_agent, get_sectors

parent_path = Path(__file__).parent


"""
Model 2 - Single objective

Note: Generating the models in reverse order as this makes more sense programmatically.
Consider changing the tutorial

"""


def generate_model_2():
    model_name = "2-single-objective"

    # Starting point: copy model from previous tutorial
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(parent_path / "../1-add-new-technology/2-scenario", model_path)

    # Modify MaxCapacityGrowth (Undocumented)
    technodata_file = model_path / "technodata/residential/technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityGrowth"] = 0.04
    df.to_csv(technodata_file, index=False)

    # Copy agent A1 -> A2
    add_agent(
        model_path,
        agent_name="A2",
        copy_from="A1",
        agentshare_new="Agent3",
        agentshare_retrofit="Agent4",
    )

    # Split capacity equally between the two agents
    for sector in get_sectors(model_path):
        technodata_file = model_path / f"technodata/{sector}/Technodata.csv"
        df = pd.read_csv(technodata_file)
        df.loc[1:, "Agent2"] = 0.5
        df.loc[1:, "Agent4"] = 0.5
        df.to_csv(technodata_file, index=False)


"""
Model 1 - Multiple objective

"""


def generate_model_1():
    model_name = "1-multiple-objective"

    # Starting point: copy model from previous tutorial
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(parent_path / "2-single-objective", model_path)

    # Special handling for Agents.csv
    agents_file = model_path / "technodata/Agents.csv"
    df = pd.read_csv(agents_file)
    df.loc[df["Name"] == "A2", "Objective2"] = "EAC"
    df.loc[df["Name"] == "A2", "DecisionMethod"] = "weighted_sum"
    df.loc[df["Name"] == "A2", ["ObjData1", "ObjData2"]] = 0.5
    df.to_csv(agents_file, index=False)


if __name__ == "__main__":
    generate_model_2()
    generate_model_1()
