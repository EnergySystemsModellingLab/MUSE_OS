import shutil
from pathlib import Path

import pandas as pd
from muse.wizard import add_new_commodity, add_new_process

parent_path = Path(__file__).parent


"""
Model 1 - New service demand

"""


def generate_model_1():
    model_name = "1-exogenous-demand"

    # Starting point: copy model from tutorial 4
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(
        parent_path / "../4-modify-timing-data/1-modify-timeslices",
        model_path,
    )

    # Copy gas commodity in power sector -> cook
    add_new_commodity(model_path, "cook", "residential", "heat")

    # Modify cook projections
    projections_file = model_path / "input/Projections.csv"
    df = pd.read_csv(projections_file)
    df.loc[1:, "cook"] = 100
    df.to_csv(projections_file, index=False)

    # Copy processes in power sector -> cook
    add_new_process(model_path, "electric_stove", "residential", "heatpump")
    add_new_process(model_path, "gas_stove", "residential", "gasboiler")

    # Modify output commodities
    commout_file = model_path / "technodata/residential/CommOut.csv"
    df = pd.read_csv(commout_file)
    df.loc[1:, "cook"] = 0
    df.loc[df["ProcessName"] == "gas_stove", df.columns[-6:]] = [0, 0, 0, 50, 0, 1]
    df.loc[df["ProcessName"] == "electric_stove", df.columns[-6:]] = [0, 0, 0, 0, 0, 1]
    df.to_csv(commout_file, index=False)

    # Modify input commodities
    commin_file = model_path / "technodata/residential/CommIn.csv"
    df = pd.read_csv(commin_file)
    df.loc[1:, "cook"] = 0
    df.to_csv(commin_file, index=False)

    # Change cap_par, Fuel and EndUse
    technodata_file = model_path / "technodata/residential/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[df["ProcessName"] == "gas_stove", "Fuel"] = "gas"
    df.loc[df["ProcessName"] == "electric_stove", "Fuel"] = "electricity"
    df.loc[df["ProcessName"] == "gas_stove", "EndUse"] = "cook"
    df.loc[df["ProcessName"] == "electric_stove", "EndUse"] = "cook"
    df.to_csv(technodata_file, index=False)

    # Increase capacity limits in power sector
    technodata_file = model_path / "technodata/power/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityAddition"] = pd.to_numeric(df.loc[1:, "MaxCapacityAddition"])
    df.loc[1:, "MaxCapacityAddition"] *= 2
    df.loc[1:, "MaxCapacityGrowth"] = pd.to_numeric(df.loc[1:, "MaxCapacityGrowth"])
    df.loc[1:, "MaxCapacityGrowth"] *= 2
    df.loc[1:, "TotalCapacityLimit"] = pd.to_numeric(df.loc[1:, "TotalCapacityLimit"])
    df.loc[1:, "TotalCapacityLimit"] *= 2
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
