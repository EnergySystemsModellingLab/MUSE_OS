from itertools import chain, permutations

import numpy as np
from pytest import mark, raises


def modify_minimum_service_factors(
    model_path, sector, processes, minimum_service_factors
):
    import pandas as pd

    technodata_timeslices = pd.read_csv(
        model_path / "technodata" / sector / "TechnodataTimeslices.csv"
    )

    for process, minimum in zip(processes, minimum_service_factors):
        technodata_timeslices.loc[
            technodata_timeslices["ProcessName"] == process, "MinimumServiceFactor"
        ] = minimum

    return technodata_timeslices


@mark.parametrize(
    "minimum_service_factors",
    permutations((np.linspace(0, 1, 6), [0] * 6)),
)
def test_minimum_service_factor(tmpdir, minimum_service_factors):
    import pandas as pd
    from muse import examples
    from muse.mca import MCA

    sector = "power"
    processes = ("gasCCGT", "windturbine")

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )

    technodata_timeslices = modify_minimum_service_factors(
        model_path=model_path,
        sector=sector,
        processes=processes,
        minimum_service_factors=minimum_service_factors,
    )

    technodata_timeslices.to_csv(
        model_path / "technodata" / sector / "TechnodataTimeslices.csv", index=False
    )

    with tmpdir.as_cwd():
        MCA.factory(model_path / "settings.toml").run()

    supply_timeslice = pd.read_csv(tmpdir / "Results/MCAMetric_Supply.csv")

    for process, service_factor in zip(processes, minimum_service_factors):
        for i, factor in enumerate(service_factor):
            assert (
                supply_timeslice[
                    (supply_timeslice.technology == process)
                    & (supply_timeslice.commodity == "electricity")
                    & (supply_timeslice.timeslice == i)
                ].supply
                >= factor
            ).all()


@mark.parametrize(
    "minimum_service_factors",
    chain.from_iterable(map(permutations, ((-1, 0), (2, 0), (float("nan"), 0)))),
)
def test_minimum_service_factor_invalid_input(tmpdir, minimum_service_factors):
    from muse import examples
    from muse.mca import MCA

    sector = "power"
    processes = ("gasCCGT", "windturbine")

    # Copy the model inputs to tmpdir
    model_path = examples.copy_model(
        name="default_timeslice", path=tmpdir, overwrite=True
    )

    technodata_timeslices = modify_minimum_service_factors(
        model_path=model_path,
        sector=sector,
        processes=processes,
        minimum_service_factors=minimum_service_factors,
    )

    technodata_timeslices.to_csv(
        model_path / "technodata" / sector / "TechnodataTimeslices.csv", index=False
    )

    with raises(ValueError):
        with tmpdir.as_cwd():
            MCA.factory(model_path / "settings.toml").run()
