import os
import shutil
from pathlib import Path

from muse import examples

parent_path = Path(__file__).parent


def generate_model_1() -> None:
    """Generates the first model for tutorial 6.

    Adds correlation demand for residential sector.

    """
    model_name = "1-correlation"
    model_path = parent_path / model_name
    if model_path.exists():
        shutil.rmtree(model_path)

    # Copy default_correlation model
    examples.copy_model(name="default_correlation", path=parent_path, overwrite=True)
    os.rename(parent_path / "model", model_path)


if __name__ == "__main__":
    generate_model_1()
