from pytest import approx, fixture, mark
from xarray import DataArray
from toml import load, dump


@mark.parametrize("utilization_factors", [""])
def test_fullsim_timeslices(utilization_factors, tmpdir, compare_dirs):
    from muse.examples import example_data_dir
    from muse import examples
    from muse.mca import MCA
    from pathlib import Path

    project_dir = Path(__file__).resolve().parents[1]

    model_path = examples.copy_model(overwrite=True)
    settings = load(model_path / "settings.toml")

    technodata_timeslices = {
        "technodata_timeslices": "{}/src/muse/data/example/default_timeslice/technodata/power/TechnodataTimeslices.csv".format(
            project_dir
        )
    }
    print(type(settings))
    settings["sectors"]["power"] += technodata_timeslices

    dump(settings, (tmpdir / "modified_settings.toml").open("w"))

    MCA.factory(tmpdir / "modified_settings.toml").run()
    # with tmpdir.as_cwd():

    assert 1 == 0


# TODO: Unit test of one sector

# TODO: Check that there is zero supply in the timeslice that the technology has zero utilization factor

# TODO: Check that there is no consumption in the timeslice that the technology has zero utilization factor

# TODO: Observe that there are differences in investments for technologies with low utilization factor

# TODO: Check that LCOE is infinite for technology with 0 utilization factor in all timeslices
