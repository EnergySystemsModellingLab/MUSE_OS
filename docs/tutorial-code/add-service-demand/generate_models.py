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

    # Copy gas commodity in power sector -> cook
    add_new_commodity(model_path, "cook", "residential", "heat")

    # Modify cook projections
    projections_file = model_path / "Projections.csv"
    df = pd.read_csv(projections_file)
    df.loc[1:, "cook"] = 100
    df.to_csv(projections_file, index=False)

    # Copy processes in power sector -> cook
    add_new_process(model_path, "electric_stove", "residential", "heatpump")
    add_new_process(model_path, "gas_stove", "residential", "gasboiler")

    # Modify output commodities
    commout_file = model_path / "residential/CommOut.csv"
    df = pd.read_csv(commout_file)
    df.loc[1:, "cook"] = 0
    df.loc[df["ProcessName"] == "gas_stove", df.columns[-6:]] = [0, 0, 0, 50, 0, 1]
    df.loc[df["ProcessName"] == "electric_stove", df.columns[-6:]] = [0, 0, 0, 0, 0, 1]
    df.to_csv(commout_file, index=False)

    # Modify input commodities
    commin_file = model_path / "residential/CommIn.csv"
    df = pd.read_csv(commin_file)
    df.loc[1:, "cook"] = 0
    df.to_csv(commin_file, index=False)

    # Change cap_par
    technodata_file = model_path / "residential/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
