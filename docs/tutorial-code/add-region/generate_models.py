import os
import shutil
from pathlib import Path

import pandas as pd

from muse import examples
from muse.wizard import add_region

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates the first model for tutorial 3.

    Adds a new region to the model.

    """
    model_name = "1-new-region"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Add region R2
    add_region(model_path, region_name="R2", copy_from="R1")

    # Reduce capacity limit for heatpump in R2
    technodata_file = model_path / "residential/Technodata.csv"
    df = pd.read_csv(technodata_file)
    mask = (df["region"] == "R2") & (df["technology"] == "heatpump")
    df.loc[mask, "total_capacity_limit"] = 20
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
