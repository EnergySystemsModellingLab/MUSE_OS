import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union

from muse.examples import AVAILABLE_EXAMPLES, model

parent_path = Path(__file__).parent


def run_model(name: str) -> Union[str, None]:
    """Run a model with the specified settings file.

    Args:
        name: Name of the model to run

    Returns:
        Error message for failed runs, otherwise None.
    """
    prev_cwd = Path.cwd()
    run_dir = parent_path / name
    try:
        # Prepare results folder
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir()

        # Run model
        os.chdir(run_dir)
        m = model(name)
        m.run()
        return None
    except Exception as e:
        return f"Failed to run model {name}. Error: {e}"
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_model, model) for model in AVAILABLE_EXAMPLES}
    for future in futures:
        error_message = future.result()
        if error_message:
            print(error_message)
