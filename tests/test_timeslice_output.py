from pytest import approx, fixture, mark
from xarray import DataArray
from toml import load, dump


@mark.parametrize("utilization_factors", [""])
def test_fullsim_timeslices(utilization_factors, tmpdir, compare_dirs):
    from muse.examples import example_data_dir
    from muse import examples
    from muse.mca import MCA
    from pathlib import Path

    model_path = examples.copy_model(name="default_timeslice", overwrite=True)

    print(model_path / "settings.toml")
    MCA.factory(model_path / "settings.toml").run()

    assert 1 == 1


# TODO: Unit test of one sector

# TODO: Check that there is zero supply in the timeslice that the technology has zero utilization factor

# TODO: Check that there is no consumption in the timeslice that the technology has zero utilization factor

# TODO: Observe that there are differences in investments for technologies with low utilization factor

# TODO: Check that LCOE is infinite for technology with 0 utilization factor in all timeslices
