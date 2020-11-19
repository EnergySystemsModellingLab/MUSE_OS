from pytest import mark


@mark.usefixtures("save_timeslice_globals")
@mark.regression
<<<<<<< HEAD
@mark.parametrize("model", ["default", "minimum-service"])
def test_fullsim_regression(model, tmpdir, compare_dirs):
    from warnings import simplefilter
    from pandas.errors import DtypeWarning
    from muse.mca import MCA
    from muse.examples import copy_model, example_data_dir
=======
def test_fullsim_regression(fullsim_dir, tmpdir, compare_dirs):
    from warnings import simplefilter
    from pathlib import Path
    from pandas.errors import DtypeWarning
    from muse.mca import MCA
    from muse.examples import copy_model
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    # fail the test if this warning crops up
    simplefilter("error", DtypeWarning)

    # Copy the data to tmpdir
<<<<<<< HEAD
    model_path = copy_model(name=model, path=tmpdir)

    # main() will output to cwd
    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    compare_dirs(
        tmpdir / "Results",
        example_data_dir() / "outputs" / model.replace("-", "_"),
        rtol=1e-5,
        atol=1e-7,
    )
=======
    copy_model(path=tmpdir)

    # main() will output to cwd
    with tmpdir.as_cwd():
        MCA.factory(Path(tmpdir) / "model" / "settings.toml").run()

    compare_dirs(tmpdir / "Results", fullsim_dir / "output")
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
