import os
import shutil
from pathlib import Path

from muse import examples
from muse.wizard import (
    modify_toml,
)

parent_path = Path(__file__).parent


def generate_model_1() -> None:
    """Generates the first model for tutorial 6.

    Adds correlation demand for residential sector.

    """
    model_name = "1-correlation"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Starting point: copy default model
    examples.copy_model(name="default", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)

    # Delete presets
    shutil.rmtree(model_path / "residential_presets")

    # Copy regression files
    os.mkdir(model_path / "residential_presets")
    for file in [
        "Macrodrivers.csv",
        "regressionparameters.csv",
        "TimesliceSharepreset.csv",
    ]:
        shutil.copy(
            parent_path / file,
            model_path / "residential_presets" / file,
        )

    # Modify toml file to point to new presets
    settings_file = model_path / "settings.toml"
    path_prefix = "{path}/residential_presets/"
    modify_toml(
        settings_file,
        lambda settings: (
            settings["sectors"]["residential_presets"].update(
                {
                    "timeslice_shares_path": path_prefix + "TimesliceSharepreset.csv",
                    "macrodrivers_path": path_prefix + "Macrodrivers.csv",
                    "regression_path": path_prefix + "regressionparameters.csv",
                }
            ),
            settings["sectors"]["residential_presets"].pop("consumption_path"),
        ),
    )


if __name__ == "__main__":
    generate_model_1()
