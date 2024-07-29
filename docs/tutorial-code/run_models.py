import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union

from muse.mca import MCA
from muse.readers.toml import read_settings

parent_path = Path(__file__).parent


def run_model(settings_file: Path) -> Union[str, None]:
    """Run a model with the specified settings file.

    Args:
        settings_file: Path to the settings file to run the model with.

    Returns:
        Error message for failed runs, otherwise None.
    """
    try:
        os.chdir(settings_file.parent)
        settings = read_settings(settings_file)
        model = MCA.factory(settings)
        model.run()
        return None
    except Exception as e:
        return f"Failed to run model with settings file {settings_file}. Error: {e}"


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_model, settings_file)
            for settings_file in parent_path.rglob("*/*settings.toml")
        }
    for future in futures:
        error_message = future.result()
        if error_message:
            print(error_message)
