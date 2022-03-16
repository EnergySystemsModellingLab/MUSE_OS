from pytest import mark


@mark.usefixtures("save_timeslice_globals")
@mark.regression
@mark.parametrize("model", ["default", "minimum-service", "trade"])
def test_fullsim_regression(model, tmpdir, compare_dirs):
    from warnings import simplefilter

    from pandas.errors import DtypeWarning

    from muse.examples import copy_model, example_data_dir
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
        example_data_dir() / "outputs" / model.replace("-", "_"),
        rtol=1e-5,
        atol=1e-7,
    )
