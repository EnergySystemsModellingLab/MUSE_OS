from pytest import approx, fixture, mark
from xarray import DataArray
from toml import load


@mark.parametrize("utilization_factors", [""])
def test_fullsim_timeslices(utilization_factors, tmpdir, compare_dirs):
    from muse.examples import example_data_dir
    from muse import examples
    from muse.mca import MCA

    model_path = examples.copy_model(overwrite=True)
    settings = load(model_path / "settings.toml")

    print("printing {path}")
    # technodata_timeslices = {
    #     "technodata_timeslices": "{path}/technodata/power/TechnodataTimeslices.csv"
    # }

    # settings["sectors"]["power"].append(technodata_timeslices)

    # with tmpdir.as_cwd():
    # MCA.factory(model_path / "settings.toml").run()


# TODO: Unit test of one sector

# TODO: Check that there is zero supply in the timeslice that the technology has zero utilization factor

# TODO: Check that there is no consumption in the timeslice that the technology has zero utilization factor

# TODO: Observe that there are differences in investments for technologies with low utilization factor

# TODO: Check that LCOE is infinite for technology with 0 utilization factor in all timeslices
