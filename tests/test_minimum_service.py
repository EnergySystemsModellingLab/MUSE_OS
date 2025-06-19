from itertools import permutations
from unittest.mock import patch

import numpy as np
from conftest import chdir
from pytest import mark


def modify_minimum_service_factors(
    model_path, sector, processes, minimum_service_factors
):
    import pandas as pd

    technodata_timeslices = pd.read_csv(
        model_path / sector / "TechnodataTimeslices.csv"
    )

    for process, minimum in zip(processes, minimum_service_factors):
        technodata_timeslices.loc[
            technodata_timeslices["ProcessName"] == process, "MinimumServiceFactor"
        ] = minimum

    return technodata_timeslices


@mark.xfail
@mark.parametrize(
    "minimum_service_factors",
    permutations((np.linspace(0, 1, 6), [0] * 6)),
)
@patch("muse.readers.csv.check_utilization_and_minimum_service_factors")
def test_minimum_service_factor(check_mock, tmp_path, minimum_service_factors):
    """NOTE: Failing due to incorrect supply output (#335)."""
    import pandas as pd

    from muse import examples
    from muse.mca import MCA

    sector = "power"
    processes = ("gasCCGT", "windturbine")

    # Copy the model inputs to tmp_path
    model_path = examples.copy_model(
        name="default_timeslice", path=tmp_path, overwrite=True
    )

    technodata_timeslices = modify_minimum_service_factors(
        model_path=model_path,
        sector=sector,
        processes=processes,
        minimum_service_factors=minimum_service_factors,
    )

    technodata_timeslices.to_csv(
        model_path / sector / "TechnodataTimeslices.csv", index=False
    )

    with chdir(tmp_path):
        MCA.factory(model_path / "settings.toml").run()
    check_mock.assert_called()

    techno_out = pd.read_csv(model_path / sector / "CommOut.csv")
    capacity = pd.read_csv(tmp_path / "Results/MCACapacity.csv")
    capacity_summed = capacity.groupby(["year", "technology"]).sum().reset_index()
    supply = pd.read_csv(tmp_path / "Results/MCAMetric_Supply.csv")
    supply = supply[supply.commodity == "electricity"].merge(
        capacity_summed[["year", "technology", "capacity"]],
        on=["year", "technology"],
        how="left",
    )
    for process, service_factor in zip(processes, minimum_service_factors):
        supply_process = supply[supply.technology == process]
        supply_process.loc[:, "min_supply"] = supply_process.apply(
            lambda x: x.capacity
            * service_factor[x.timeslice]
            * float(
                techno_out[techno_out.ProcessName == process]["electricity"].values[0]
            ),
            axis=1,
        )
        assert (supply_process["supply"] >= supply_process["min_supply"]).all()
