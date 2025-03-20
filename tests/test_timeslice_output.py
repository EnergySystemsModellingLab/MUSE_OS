from pytest import mark, raises


def modify_technodata_timeslices(model_path, sector, process_name, utilization_factors):
    import pandas as pd

    technodata_timeslices = pd.read_csv(
        model_path / sector / "TechnodataTimeslices.csv"
    )

    technodata_timeslices.loc[
        technodata_timeslices["ProcessName"] == process_name[0], "UtilizationFactor"
    ] = utilization_factors[0]

    technodata_timeslices.loc[
        technodata_timeslices["ProcessName"] == process_name[1], "UtilizationFactor"
    ] = utilization_factors[1]
    technodata_timeslices["MinimumServiceFactor"] = 0
    return technodata_timeslices


@mark.parametrize("utilization_factors", [([0.1], [1]), ([1], [0.1])])
@mark.parametrize("process_name", [("gasCCGT", "windturbine")])
def test_fullsim_timeslices(tmpdir, utilization_factors, process_name):
    from operator import ge, le

    import pandas as pd

    from muse import examples
    from muse.mca import MCA

    sector = "power"

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )
    technodata_timeslices = modify_technodata_timeslices(
        model_path=model_path,
        sector=sector,
        process_name=process_name,
        utilization_factors=utilization_factors,
    )

    technodata_timeslices.to_csv(
        model_path / sector / "TechnodataTimeslices.csv", index=False
    )

    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    MCACapacity = pd.read_csv(tmpdir / "Results/MCACapacity.csv")

    if utilization_factors[0] > utilization_factors[1]:
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


@mark.parametrize(
    "utilization_factors",
    [
        ([0.0001, 0.0001, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        ([1, 1, 1, 1, 1, 1], [1, 1, 0.0001, 0.0001, 1, 1]),
    ],
)
@mark.parametrize("process_name", [("gasCCGT", "windturbine")])
def test_zero_utilization_factor_supply_timeslice(
    tmpdir, utilization_factors, process_name
):
    import pandas as pd

    from muse import examples
    from muse.mca import MCA

    sector = "power"

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )

    technodata_timeslices = modify_technodata_timeslices(
        model_path=model_path,
        sector=sector,
        process_name=process_name,
        utilization_factors=utilization_factors,
    )

    technodata_timeslices.to_csv(
        model_path / sector / "TechnodataTimeslices.csv", index=False
    )

    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    path = str(tmpdir / "Results" / "Power_Supply.csv")

    output = pd.read_csv(path)

    output = output.reset_index()
    zero_utilization_factors = [i for i, e in enumerate(utilization_factors) if e == 0]

    assert (
        len(
            output[
                (
                    output.timeslice.isin(zero_utilization_factors)
                    & (output.technology == process_name)
                )
            ]
        )
        == 0
    )


@mark.parametrize("utilization_factors", [([0], [1]), ([1], [0])])
@mark.parametrize("process_name", [("gasCCGT", "windturbine")])
def test_all_zero_fatal_error(tmpdir, utilization_factors, process_name):
    from muse import examples
    from muse.mca import MCA

    sector = "power"

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )
    technodata_timeslices = modify_technodata_timeslices(
        model_path=model_path,
        sector=sector,
        process_name=process_name,
        utilization_factors=utilization_factors,
    )

    technodata_timeslices.to_csv(
        model_path / sector / "TechnodataTimeslices.csv", index=False
    )

    with tmpdir.as_cwd(), raises(ValueError):
        MCA.factory(model_path / "settings.toml").run()
