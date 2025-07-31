import os
import shutil
from pathlib import Path

import pandas as pd

from muse import examples
from muse.wizard import add_new_commodity, add_new_process

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates the first model for tutorial 5.

    Adds a new service demand for cooking.

    """
    model_name = "1-exogenous-demand"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Copy gas heat in residential sector -> cook
    add_new_commodity(model_path, "cook", "residential", "heat")

    # Copy processes in residential sector -> electric_stove and gas_stove
    add_new_process(model_path, "electric_stove", "residential", "heatpump")
    add_new_process(model_path, "gas_stove", "residential", "gasboiler")

    # Add preset demand for cook
    residential_presets_path = model_path / "residential_presets"
    for file in residential_presets_path.glob("*.csv"):
        df = pd.read_csv(file)
        df.loc[:, "cook"] = df.loc[:, "heat"]
        df.to_csv(file, index=False)

    # Modify output commodities
    commout_file = model_path / "residential/CommOut.csv"
    df = pd.read_csv(commout_file)
    df.loc[:, "cook"] = 0
    df.loc[df["technology"] == "gas_stove", df.columns[-3:]] = [0, 50, 1]
    df.loc[df["technology"] == "electric_stove", df.columns[-3:]] = [0, 0, 1]
    df.to_csv(commout_file, index=False)

    # Modify input commodities
    commin_file = model_path / "residential/CommIn.csv"
    df = pd.read_csv(commin_file)
    df.loc[:, "cook"] = 0
    df.to_csv(commin_file, index=False)

    # Change cap_par
    technodata_file = model_path / "residential/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
