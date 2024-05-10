import os
import shutil
import sys

import pandas as pd
from muse import examples
from muse.wizard import (
    add_agent_allocation,
    add_new_commodity,
    add_new_process,
    add_price_data_for_new_year,
    get_sectors,
)

# WORK IN PROGRESS

"""
Model 1 - Introduction

"""


def generate_model_1():
    model_name = "1-introduction"
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    model_path = os.path.join(parent_path, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(os.path.join(parent_path, "model"), model_path)

    # Global change: modify MaxCapacityGrowth (Undocumented in notebook)
    technodata_file = os.path.join(model_path, "technodata/power/technodata.csv")
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityGrowth"] = 0.2
    df.to_csv(technodata_file, index=False)

    # Global change: Add Agent 1 to technodata files (Undocumented in notebook)
    add_agent_allocation(model_path, "Agent1", "Agent2")
    for sector in get_sectors(model_path):
        technodata_file = os.path.join(
            model_path, f"technodata/{sector}/technodata.csv"
        )
        df = pd.read_csv(technodata_file)
        df.loc[0, "Agent1"] = "New"
        df.loc[1:, "Agent1"] = 0
        df.to_csv(technodata_file, index=False)

    ### Tutorial

    # Copy wind commodity in power sector -> solar
    add_new_commodity(model_path, "solar", "power", "wind")

    # Copy windturbine process in power sector -> solarPV
    add_new_process(model_path, "solarPV", "power", "windturbine")

    # Special handling for CommIn.csv
    commin_file = os.path.join(model_path, "technodata/power/CommIn.csv")
    df = pd.read_csv(commin_file)
    df.loc[(df["ProcessName"] == "solarPV"), "solar"] = 1
    df.loc[(df["ProcessName"] == "solarPV"), "wind"] = 0
    df.loc[(df["ProcessName"] == "windturbine"), "solar"] = 0
    df.to_csv(commin_file, index=False)

    # Special handling for technodata.csv
    technodata_file = os.path.join(model_path, "technodata/power/technodata.csv")
    df = pd.read_csv(technodata_file)
    df.loc[df["ProcessName"] == "solarPV", "cap_par"] = 30
    df.loc[df["ProcessName"] == "solarPV", "Fuel"] = "solar"
    df.to_csv(technodata_file, index=False)


def generate_model_2():
    model_name = "2-scenario"
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    model_path = os.path.join(parent_path, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Starting point: copy model from 1-introduction
    shutil.copytree(os.path.join(parent_path, "1-introduction"), model_path)

    # Copy price data for 2020 -> 2040
    add_price_data_for_new_year(model_path, "2040", "power", "2020")

    # Special handling for technodata.csv
    technodata_file = os.path.join(model_path, "technodata/power/technodata.csv")
    df = pd.read_csv(technodata_file)
    df.loc[(df["ProcessName"] == "solarPV") & (df["Time"] == "2040"), "cap_par"] = 30
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
