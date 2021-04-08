from pytest import mark


def modify_technodata_timeslices(model_path, sector, process_name, utilization_factor):
    import pandas as pd

    technodata_timeslices = pd.read_csv(
        model_path / "technodata" / sector / "TechnodataTimeslices.csv"
    )

    technodata_timeslices.loc[
        technodata_timeslices["ProcessName"] == process_name[0], "UtilizationFactor"
    ] = utilization_factor[0]

    technodata_timeslices.loc[
        technodata_timeslices["ProcessName"] == process_name[1], "UtilizationFactor"
    ] = utilization_factor[1]

    return technodata_timeslices


@mark.parametrize("utilization_factor", [(0.1, 1), (1, 0.1)])
@mark.parametrize("process_name", [("gasCCGT", "windturbine")])
def test_fullsim_timeslices(tmpdir, utilization_factor, process_name):
    from muse import examples
    from muse.mca import MCA
    import pandas as pd
    from operator import le, ge

    sector = "power"

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )

    technodata_timeslices = modify_technodata_timeslices(
        model_path=model_path,
        sector=sector,
        process_name=process_name,
        utilization_factor=utilization_factor,
    )

    technodata_timeslices.to_csv(
        model_path / "technodata" / sector / "TechnodataTimeslices.csv"
    )

    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    MCACapacity = pd.read_csv(tmpdir / "Results/MCACapacity.csv")

    if utilization_factor[0] > utilization_factor[1]:
        operator = ge
    else:
        operator = le

    assert operator(
        len(
            MCACapacity[
                (MCACapacity.sector == sector)
                & (MCACapacity.technology == process_name[0])
            ]
        ),
        len(
            MCACapacity[
                (MCACapacity.sector == sector)
                & (MCACapacity.technology == process_name[1])
            ]
        ),
    )


# TODO: Unit test of one sector

# TODO: Check that there is zero supply in the timeslice that the technology has
# zero utilization factor

# TODO: Check that there is no consumption in the timeslice that the technology has
# zero utilization factor

# TODO: Observe that there are differences in investments for technologies with low
# utilization factor

# TODO: Check that LCOE is infinite for technology with 0 utilization factor in all
# timeslices
