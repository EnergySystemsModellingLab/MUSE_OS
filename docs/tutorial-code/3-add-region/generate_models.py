import os
import shutil
import sys

import pandas as pd
from muse.wizard import add_region

parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))


"""
Model 1 - New region

"""


def generate_model_1():
    model_name = "1-new-region"

    # Starting point: copy model from tutorial 1
    model_path = os.path.join(parent_path, model_name)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    shutil.copytree(
        os.path.join(parent_path, "../1-add-new-technology/2-scenario"), model_path
    )

    # Add region R2
    add_region(model_path, region_name="R2", copy_from="R1")

    # Special handling for power/Technodata.csv
    technodata_file = os.path.join(model_path, "technodata/power/Technodata.csv")
    df = pd.read_csv(technodata_file)
    mask = (df["RegionName"] == "R2") & (df["ProcessName"] == "windturbine")
    df.loc[mask, "MaxCapacityAddition"] = 5
    df.loc[mask, "MaxCapacityGrowth"] = 0.5  # incorrect in notebook
    df.loc[mask, "TotalCapacityLimit"] = 100
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
