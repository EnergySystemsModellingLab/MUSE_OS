from pathlib import Path

from conftest import chdir
from pytest import mark, xfail

from muse.examples import AVAILABLE_EXAMPLES


@mark.regression
@mark.example
@mark.parametrize("model", AVAILABLE_EXAMPLES)
def test_fullsim_regression(model, tmp_path, compare_dirs):
    from warnings import simplefilter

    from pandas.errors import DtypeWarning

    from muse.examples import copy_model
    from muse.mca import MCA

    # fail the test if this warning crops up
    simplefilter("error", DtypeWarning)

    # Copy the data to tmp_path
    model_path = copy_model(name=model, path=tmp_path)

    # main() will output to cwd
    with chdir(tmp_path):
        MCA.factory(model_path / "settings.toml").run()

    compare_dirs(
        tmp_path / "Results",
        Path(__file__).parent / "example_outputs" / model.replace("-", "_") / "Results",
        rtol=1e-4,
        atol=1e-7,
    )


def available_tutorials():
    base_path = Path(__file__).parent.parent / "docs" / "tutorial-code"
    return [p.parent for p in base_path.rglob("settings.toml")]


@mark.regression
@mark.tutorial
@mark.parametrize("tutorial_path", available_tutorials())
def test_tutorial_regression(tutorial_path, tmp_path, compare_dirs):
    import shutil
    from warnings import simplefilter

    from pandas.errors import DtypeWarning

    from muse.mca import MCA

    # Mark as xfail for a specific tutorial
    if "modify-time-framework" in str(tutorial_path):
        xfail(reason="Known issue with this tutorial (#371)")

    # fail the test if this warning crops up
    simplefilter("error", DtypeWarning)

    # Copy the data to tmp_path
    shutil.copytree(tutorial_path, tmp_path, dirs_exist_ok=True)

    # Get the toml file to run. There should be only one settings file
    settings = list(tmp_path.glob("*.toml"))
    assert len(settings) == 1

    # Rename existing results directory for comparison
    expected = tmp_path / "Expected"
    (tmp_path / "Results").rename(expected)

    # main() will output to cwd
    # Change working directory to tmp_path
    with chdir(tmp_path):
        MCA.factory(settings[0]).run()

    compare_dirs(
        tmp_path / "Results",
        expected,
        rtol=1e-4,
        atol=1e-7,
    )
