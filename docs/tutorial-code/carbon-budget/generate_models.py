import os
import shutil
from pathlib import Path

from muse import examples
from muse.wizard import (
    modify_toml,
)

parent_path = Path(__file__).parent


def generate_model_1() -> None:
    """Generates the first model for tutorial 1.

    Default model with a carbon budget.

    """
    model_name = "1-carbon-budget"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Modify time framework
    settings_file = model_path / "settings.toml"
    time_framework = [2020, 2025, 2030, 2035]
    modify_toml(settings_file, lambda x: x.update({"time_framework": time_framework}))

    # Add supply output
    modify_toml(
        settings_file,
        lambda x: x["outputs"].append(
            {
                "quantity": "supply",
                "sink": "aggregate",
                "filename": "{cwd}/{default_output_dir}/MCA{Quantity}.csv",
            }
        ),
    )

    # Add carbon budget
    carbon_budget_settings = {
        "budget": [300, 300, 300, 300],
        "commodities": ["CO2f"],
        "method": "bisection",
        "control_undershoot": False,
        "control_overshoot": False,
        "method_options": {
            "max_iterations": 5,
            "tolerance": 0.2,
        },
    }
    modify_toml(
        settings_file,
        lambda x: x.update({"carbon_budget_control": carbon_budget_settings}),
    )


if __name__ == "__main__":
    generate_model_1()
