import os
import shutil
import sys

import pandas as pd
from muse import examples
from muse.wizard import (
    add_new_commodity,
    add_new_process,
    add_price_data_for_new_year,
    modify_toml,
)

"""
Model 1 - Introduction

"""


def generate_model_1() -> None:
    """
    Code to generate model 1-introduction.

    """
    model_name = "1-introduction"
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    model_path = os.path.join(parent_path, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(os.path.join(parent_path, "model"), model_path)

    # Modify MaxCapacityGrowth (Undocumented)
    technodata_file = os.path.join(model_path, "technodata/power/technodata.csv")
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityGrowth"] = 0.2
    df.to_csv(technodata_file, index=False)

    # Change maximum_iterations to 100 (Undocumented)
    settings_file = os.path.join(model_path, "settings.toml")
    modify_toml(
        settings_file, lambda settings: settings.update({"maximum_iterations": 100})
    )

    # Change carbon budget method (Undocumented)
    modify_toml(
        settings_file,
        lambda settings: settings["carbon_budget_control"].update(
            {"method": "fitting"}
        ),
    )

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

    # Add solar to excluded commodities (Undocumented)
    settings_file = os.path.join(model_path, "settings.toml")
    modify_toml(
        settings_file, lambda settings: settings["excluded_commodities"].append("solar")
    )


def generate_model_2() -> None:
    """
    Code to generate model 2-scenario.

    """
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
    df.loc[(df["ProcessName"] == "solarPV") & (df["Time"] == "2020"), "cap_par"] = 40
    df.loc[(df["ProcessName"] == "solarPV") & (df["Time"] == "2040"), "cap_par"] = 30
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
