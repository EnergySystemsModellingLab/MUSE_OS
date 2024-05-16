import os
import shutil
from pathlib import Path

import pandas as pd
from muse import examples

parent_path = Path(__file__).parent


"""
Model 1 - Min constraint

"""


def generate_model_1() -> None:
    model_name = "1-min-constraint"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default_timeslice model
    examples.copy_model(name="default_timeslice", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Modify MinimumServiceFactor for gasCCGT
    timeslices_file = model_path / "technodata/power/TechnodataTimeslices.csv"
    df = pd.read_csv(timeslices_file)
    df.loc[df["ProcessName"] == "gasCCGT", "MinimumServiceFactor"] = [
        0.2,
        0.4,
        0.6,
        0.4,
        0.8,
        1,
    ]
    df.to_csv(timeslices_file, index=False)


def generate_model_2() -> None:
    model_name = "2-max-constraint"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy model from 1-introduction
    shutil.copytree(parent_path / "1-min-constraint", model_path)

    # Modify UtilizationFactor and MinimumServiceFactor for windturbine
    timeslices_file = model_path / "technodata/power/TechnodataTimeslices.csv"
    df = pd.read_csv(timeslices_file)
    df.loc[df["ProcessName"] == "windturbine", "MinimumServiceFactor"] = 0
    df.loc[
        (df["ProcessName"] == "windturbine")
        & (df["hour"].isin(["morning", "afternoon"])),
        "UtilizationFactor",
    ] = 0.5
    df.to_csv(timeslices_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
