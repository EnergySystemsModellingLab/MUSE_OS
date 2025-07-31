import os
import shutil
from pathlib import Path

import pandas as pd

from muse import examples
from muse.wizard import add_agent, get_sectors

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates the first model for tutorial 1.

    Adds a new agent to the model with a single objective.

    """
    model_name = "1-single-objective"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Copy agent A1 -> A2
    add_agent(
        model_path,
        agent_name="A2",
        copy_from="A1",
        agentshare_new="Agent2",
    )

    # Change objective for agent A2
    agents_file = model_path / "Agents.csv"
    df = pd.read_csv(agents_file)
    df.loc[df["name"] == "A2", "objective1"] = "fuel_consumption_cost"
    df.to_csv(agents_file, index=False)

    # Split population between the two agents
    agents_file = model_path / "Agents.csv"
    df = pd.read_csv(agents_file)
    df.loc[:, "quantity"] = 0.5
    df.to_csv(agents_file, index=False)

    # Split capacity equally between the two agents
    for sector in get_sectors(model_path):
        technodata_file = model_path / f"{sector}/Technodata.csv"
        df = pd.read_csv(technodata_file)
        df.loc[:, "Agent1"] = 0.5
        df.loc[:, "Agent2"] = 0.5
        df.to_csv(technodata_file, index=False)


def generate_model_2():
    """Generates the second model for tutorial 2.

    Adds a second objective for agent A2.

    """
    model_name = "2-multiple-objective"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy model from previous tutorial
    shutil.copytree(parent_path / "1-single-objective", model_path)
    if (model_path / "Results").exists():
        shutil.rmtree(model_path / "Results")

    # Add second objective for agent A2
    agents_file = model_path / "Agents.csv"
    df = pd.read_csv(agents_file)
    df.loc[df["name"] == "A2", "objective2"] = "LCOE"
    df.loc[df["name"] == "A2", "decision_method"] = "weighted_sum"
    df.loc[df["name"] == "A2", ["obj_data1", "obj_data2"]] = 0.5
    df.loc[df["name"] == "A2", "obj_sort2"] = True
    df.to_csv(agents_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
