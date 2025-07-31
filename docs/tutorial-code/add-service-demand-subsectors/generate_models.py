import os
import shutil
from pathlib import Path

import pandas as pd

from muse.wizard import modify_toml

parent_path = Path(__file__).parent


def generate_model_1():
    """Generates the first model.

    Takes the model from the cook tutorial and splits residential sector into
    subsectors.

    """
    model_name = "1-residential-subsectors"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy model from previous tutorial
    shutil.copytree(
        parent_path.parent / "add-service-demand/1-exogenous-demand", model_path
    )
    if (model_path / "Results").exists():
        shutil.rmtree(model_path / "Results")

    # Split existing capacity
    df = pd.read_csv(model_path / "residential" / "ExistingCapacity.csv")
    df_heat = df[df["technology"].isin(["gasboiler", "heatpump"])]
    df_cook = df[df["technology"].isin(["electric_stove", "gas_stove"])]
    df_heat.to_csv(model_path / "residential" / "ExistingCapacityHeat.csv", index=False)
    df_cook.to_csv(model_path / "residential" / "ExistingCapacityCook.csv", index=False)
    os.remove(model_path / "residential" / "ExistingCapacity.csv")

    # Update settings file to create multiple subsectors
    settings_file = model_path / "settings.toml"
    modify_toml(
        settings_file,
        lambda settings: settings["sectors"]["residential"].update(
            {
                "subsectors": {
                    "heat": {
                        "agents": "{path}/Agents.csv",
                        "existing_capacity": "{path}/residential/ExistingCapacityHeat.csv",  # noqa: E501
                        "commodities": ["heat"],
                    },
                    "cook": {
                        "agents": "{path}/Agents.csv",
                        "existing_capacity": "{path}/residential/ExistingCapacityCook.csv",  # noqa: E501
                        "commodities": ["cook"],
                    },
                }
            }
        ),
    )


def generate_model_2():
    """Generates the second model.

    Changes the agent objective in the cook subsector.

    """
    model_name = "2-agents"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy model from previous tutorial
    shutil.copytree(parent_path / "1-residential-subsectors", model_path)
    if (model_path / "Results").exists():
        shutil.rmtree(model_path / "Results")

    # Duplicate agents file
    shutil.copy(
        model_path / "Agents.csv", model_path / "residential" / "AgentsCook.csv"
    )
    shutil.copy(
        model_path / "Agents.csv", model_path / "residential" / "AgentsHeat.csv"
    )

    # Modify agent objective in the cook subsector
    df = pd.read_csv(model_path / "residential" / "AgentsCook.csv")
    df["objective1"] = "fuel_consumption_cost"
    df.to_csv(model_path / "residential" / "AgentsCook.csv", index=False)

    # Update settings file to link to new agents files
    settings_file = model_path / "settings.toml"
    modify_toml(
        settings_file,
        lambda settings: settings["sectors"]["residential"]["subsectors"][
            "cook"
        ].update({"agents": "{path}/residential/AgentsCook.csv"}),
    )
    modify_toml(
        settings_file,
        lambda settings: settings["sectors"]["residential"]["subsectors"][
            "heat"
        ].update({"agents": "{path}/residential/AgentsHeat.csv"}),
    )


if __name__ == "__main__":
    generate_model_1()
    generate_model_2()
