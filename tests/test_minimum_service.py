from itertools import permutations
from unittest.mock import patch

import numpy as np
from pytest import mark


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
@patch("muse.readers.csv.check_utilization_and_minimum_service_factors")
def test_minimum_service_factor(check_mock, tmpdir, minimum_service_factors):
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
    check_mock.assert_called_once()

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
