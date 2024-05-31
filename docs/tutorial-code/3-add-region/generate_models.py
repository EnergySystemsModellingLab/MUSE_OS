import shutil
from pathlib import Path

import pandas as pd
from muse.wizard import add_region

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates the first model for tutorial 3.

    Adds a new region to the model.

    """
    model_name = "1-new-region"

    # Starting point: copy model from tutorial 1
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(parent_path / "../1-add-new-technology/2-scenario", model_path)

    # Add region R2
    add_region(model_path, region_name="R2", copy_from="R1")

    # Change growth/capacity limits for windturbine in R2
    technodata_file = model_path / "technodata/power/Technodata.csv"
    df = pd.read_csv(technodata_file)
    mask = (df["RegionName"] == "R2") & (df["ProcessName"] == "windturbine")
    df.loc[mask, "MaxCapacityAddition"] = 5
    df.loc[mask, "MaxCapacityGrowth"] = 0.5  # incorrect in notebook
    df.loc[mask, "TotalCapacityLimit"] = 100
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
