import shutil
from pathlib import Path

import pandas as pd
from muse.wizard import add_timeslice, get_sectors, modify_toml
from tomlkit import dumps, parse

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates first model for tutorial 4.

    Adds two new timeslices to the model.

    """
    model_name = "1-modify-timeslices"

    # Starting point: copy model from tutorial 3
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(parent_path / "../3-add-region/1-new-region", model_path)

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
        file = model_path / f"technodata/preset/Residential{year}Consumption.csv"
        df = pd.read_csv(file)
        df["heat"] = [round(i * multiplier, 1) for i in consumption_values]
        df.to_csv(file, index=False)


def generate_model_2():
    """Generates the second model for tutorial 4.

    Modifies the time framework of the model to every two years.

    """
    model_name = "2-modify-time-framework"

    # Starting point: copy previous model
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.copytree(parent_path / "1-modify-timeslices", model_path)

    # Modify time framework
    settings_file = model_path / "settings.toml"
    time_framework = [2020, 2022, 2024, 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040]
    modify_toml(settings_file, lambda x: x.update({"time_framework": time_framework}))
    modify_toml(settings_file, lambda x: x.update({"foresight": 2}))
    for sector in get_sectors(model_path):
        modify_toml(
            settings_file,
            lambda x: x["sectors"][sector]["subsectors"]["all"].update({"forecast": 2}),
        )

    # Increase capacity limits in power sector
    technodata_file = model_path / "technodata/power/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityAddition"] = pd.to_numeric(df.loc[1:, "MaxCapacityAddition"])
    df.loc[1:, "MaxCapacityAddition"] *= 3
    df.loc[1:, "MaxCapacityGrowth"] = pd.to_numeric(df.loc[1:, "MaxCapacityGrowth"])
    df.loc[1:, "MaxCapacityGrowth"] *= 3
    df.loc[1:, "TotalCapacityLimit"] = pd.to_numeric(df.loc[1:, "TotalCapacityLimit"])
    df.loc[1:, "TotalCapacityLimit"] *= 3
    df.to_csv(technodata_file, index=False)

    # Increase capacity limits in residential sector
    technodata_file = model_path / "technodata/residential/Technodata.csv"
    df = pd.read_csv(technodata_file)
    df.loc[1:, "MaxCapacityAddition"] = pd.to_numeric(df.loc[1:, "MaxCapacityAddition"])
    df.loc[1:, "MaxCapacityAddition"] *= 2
    df.loc[1:, "MaxCapacityGrowth"] = pd.to_numeric(df.loc[1:, "MaxCapacityGrowth"])
    df.loc[1:, "MaxCapacityGrowth"] *= 3
    df.loc[1:, "TotalCapacityLimit"] = pd.to_numeric(df.loc[1:, "TotalCapacityLimit"])
    df.loc[1:, "TotalCapacityLimit"] *= 2
    df.to_csv(technodata_file, index=False)


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
