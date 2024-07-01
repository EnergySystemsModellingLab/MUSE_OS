from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pytest import mark


def available_notebooks() -> list[Path]:
    """Locate the available notebooks in the docs."""
    base_path = Path(__file__).parent.parent / "docs"
    return [p for p in base_path.rglob("*.ipynb") if "build" not in str(p)]


@mark.notebook
@mark.parametrize("notebook", available_notebooks())
def test_notebook(notebook):
    with notebook.open() as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name="python3")
    proc.preprocess(nb, {"metadata": {"path": str(notebook.parent)}})
