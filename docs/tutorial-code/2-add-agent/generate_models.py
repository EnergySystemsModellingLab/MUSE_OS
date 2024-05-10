import os
import shutil
import sys

import pandas as pd
from muse.wizard import add_agent

parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))

# WORK IN PROGRESS

"""
Model 1 - Multiple objective

"""


def generate_model_1():
    model_name = "1-multiple-objective"

    # Starting point: copy model from previous tutorial
    model_path = os.path.join(parent_path, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    shutil.copytree(
        os.path.join(parent_path, "../1-add-new-technology/2-scenario"), model_path
    )

    # Copy agent A1 -> A2
    add_agent(
        model_path,
        agent_name="A2",
        copy_from="A1",
        agentshare_new="Agent3",
        agentshare_retrofit="Agent4",
    )

    # Special handling for Agents.csv
    agents_file = os.path.join(model_path, "technodata/Agents.csv")
    df = pd.read_csv(agents_file)
    df.loc[df["Name"] == "A2", "Objective2"] = "EAC"
    df.loc[df["Name"] == "A2", "DecisionMethod"] = "weighted_sum"
    df.loc[df["Name"] == "A2", ["ObjData1", "ObjData2"]] = 0.5
    df.to_csv(agents_file, index=False)


if __name__ == "__main__":
    generate_model_1()
