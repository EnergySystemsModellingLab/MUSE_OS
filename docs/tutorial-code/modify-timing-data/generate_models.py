import os
import shutil
from pathlib import Path

import pandas as pd
from tomlkit import dumps, parse

from muse import examples
from muse.wizard import add_timeslice, modify_toml

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates first model for tutorial 4.

    Adds two new timeslices to the model.

    """
    model_name = "1-modify-timeslices"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Add timeslices
    add_timeslice(model_path, timeslice_name="early-morning", copy_from="evening")
    add_timeslice(model_path, timeslice_name="late-afternoon", copy_from="evening")

    # Modify timeslices
    settings_file = model_path / "settings.toml"
    settings = parse(settings_file.read_text())
    timeslices = settings["timeslices"]["all-year"]["all-week"]
    for key in timeslices:
        timeslices[key] = 1095
    settings["timeslices"]["all-year"]["all-week"] = {
        key if key != "afternoon" else "mid-afternoon": value
        for key, value in timeslices.items()
    }  # hacky way to preserve the order
    settings_file.write_text(dumps(settings))

    # Change consumption profile
    consumption_values = [
        0.7,
        1.0,
        0.7,
        1.0,
        2.1,
        1.4,
        1.4,
        1.4,
    ]
    for year, multiplier in zip([2020, 2050], [1, 3]):
        file = model_path / f"residential_presets/Residential{year}Consumption.csv"
        df = pd.read_csv(file)
        df["heat"] = [round(i * multiplier, 1) for i in consumption_values]
        df.to_csv(file, index=False)


def generate_model_2():
    """Generates the second model for tutorial 4.

    Modifies the time framework of the model to every two years.

    """
    model_name = "2-modify-time-framework"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy previous model
    shutil.copytree(parent_path / "1-modify-timeslices", model_path)
    if (model_path / "Results").exists():
        shutil.rmtree(model_path / "Results")

    # Modify time framework
    settings_file = model_path / "settings.toml"
    time_framework = [2020, 2022, 2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040]
    modify_toml(settings_file, lambda x: x.update({"time_framework": time_framework}))


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
