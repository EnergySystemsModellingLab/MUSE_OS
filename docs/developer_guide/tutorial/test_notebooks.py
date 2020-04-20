"""Runs notebooks to check they still work."""
from pytest import mark


@mark.parametrize(
    "filename", ["GeneralizedSectorOverview", "ExtendingTheGeneralizedSector"]
)
def test_notebooks(tmpdir, filename):
    """Runs a given notebook in a tmpdir."""
    from pathlib import Path
    from shutil import copytree, rmtree
    from nbformat import read
    from nbconvert.preprocessors import ExecutePreprocessor
    from sys import version_info

    directory = copytree(Path(__file__).parent, tmpdir / "notebooks")
    if (tmpdir / "notebooks" / "Results").exists():
        rmtree(tmpdir / "notebooks" / "Results")
    with (directory / (filename + ".ipynb")).open("r") as notebook_file:
        notebook = read(notebook_file, as_version=4)
    preprocessor = ExecutePreprocessor(
        timeout=200, kernel_name="python%i" % version_info.major
    )
    preprocessor.preprocess(notebook, {"metadata": {"path": str(directory)}})
