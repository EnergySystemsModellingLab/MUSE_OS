from pathlib import Path

from pytest import mark

from muse.examples import available_examples


@mark.regression
@mark.example
@mark.parametrize("model", available_examples())
def test_fullsim_regression(model, tmpdir, compare_dirs):
    from warnings import simplefilter

    from pandas.errors import DtypeWarning

    from muse.examples import copy_model
    from muse.mca import MCA

    # fail the test if this warning crops up
    simplefilter("error", DtypeWarning)

    # Copy the data to tmpdir
    model_path = copy_model(name=model, path=tmpdir)

    # main() will output to cwd
    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    compare_dirs(
        tmpdir / "Results",
        Path(__file__).parent / "example_outputs" / model.replace("-", "_") / "Results",
        rtol=1e-4,
        atol=1e-7,
    )


def available_tutorials():
    base_path = Path(__file__).parent.parent / "docs" / "tutorial-code"
    return [d.parent for d in base_path.rglob("*/input") if d.is_dir()]


@mark.regression
@mark.tutorial
@mark.parametrize("tutorial_path", available_tutorials())
def test_tutorial_regression(tutorial_path, tmpdir, compare_dirs):
    import shutil
    from warnings import simplefilter

    from pandas.errors import DtypeWarning

    from muse.mca import MCA

    # fail the test if this warning crops up
    simplefilter("error", DtypeWarning)

    # Copy the data to tmpdir
    shutil.copytree(tutorial_path, tmpdir, dirs_exist_ok=True)

    # Get the toml file to run. There should be only one settings file
    settings = list(Path(tmpdir).glob("*.toml"))
    assert len(settings) == 1

    # Rename existing results directory for comparison
    expected = Path(tmpdir) / "Expected"
    (Path(tmpdir) / "Results").rename(expected)

    # main() will output to cwd
    with tmpdir.as_cwd():
        MCA.factory(settings[0]).run()

    compare_dirs(
        tmpdir / "Results",
        expected,
        rtol=1e-4,
        atol=1e-7,
    )
