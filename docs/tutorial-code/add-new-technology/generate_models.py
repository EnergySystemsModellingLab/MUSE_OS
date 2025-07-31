import os
import shutil
from pathlib import Path

import pandas as pd

from muse import examples
from muse.wizard import (
    add_new_commodity,
    add_new_process,
    add_technodata_for_new_year,
    modify_toml,
)

parent_path = Path(__file__).parent


def generate_model_1() -> None:
    """Generates the first model for tutorial 1.

    Adds solarPV to the default model.

    """
    model_name = "1-introduction"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Copy wind commodity in power sector -> solar
    add_new_commodity(model_path, "solar", "power", "wind")

    # Copy windturbine process in power sector -> solarPV
    add_new_process(model_path, "solarPV", "power", "windturbine")

    # Modify input commodities
    commin_file = model_path / "power/CommIn.csv"
    df = pd.read_csv(commin_file)
    df.loc[(df["technology"] == "solarPV"), "solar"] = 1
    df.loc[(df["technology"] == "solarPV"), "wind"] = 0
    df.loc[(df["technology"] == "windturbine"), "solar"] = 0
    df.fillna(0, inplace=True)
    df.to_csv(commin_file, index=False)

    # Modify technodata for solarPV
    technodata_file = model_path / "power/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[df["technology"] == "solarPV", "cap_par"] = 30
    df.to_csv(technodata_file, index=False)

    # Add solar to excluded commodities
    settings_file = model_path / "settings.toml"
    modify_toml(
        settings_file, lambda settings: settings["excluded_commodities"].append("solar")
    )


def generate_model_2() -> None:
    """Generates the second model for tutorial 1.

    Changes the price of solar between 2020 and 2040.

    """
    model_name = "2-scenario"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy model from 1-introduction
    shutil.copytree(parent_path / "1-introduction", model_path)
    if (model_path / "Results").exists():
        shutil.rmtree(model_path / "Results")

    # Copy technodata for 2020 -> 2040
    add_technodata_for_new_year(model_path, 2040, "power", 2020)

    # Modify cap_par for solarPV
    technodata_file = model_path / "power/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[(df["technology"] == "solarPV") & (df["year"] == 2020), "cap_par"] = 40
    df.loc[(df["technology"] == "solarPV") & (df["year"] == 2040), "cap_par"] = 30
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
