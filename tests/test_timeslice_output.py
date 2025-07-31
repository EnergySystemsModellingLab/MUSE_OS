from operator import ge, le
from pathlib import Path

import pandas as pd
from conftest import chdir
from pytest import mark, raises

from muse import examples
from muse.mca import MCA


def setup_test_environment(
    tmp_path: Path,
    sector: str,
    process_names: tuple[str, str],
    utilization_factors: list[float],
) -> Path:
    """Set up test environment with modified technodata timeslices."""
    model_path = examples.copy_model(
        name="default_timeslice", path=tmp_path, overwrite=True
    )

    # Read and modify technodata timeslices
    technodata = pd.read_csv(model_path / sector / "TechnodataTimeslices.csv")
    technodata["utilization_factor"] = technodata["utilization_factor"].astype(float)
    for process, factor in zip(process_names, utilization_factors):
        technodata.loc[technodata["technology"] == process, "utilization_factor"] = (
            factor
        )
    technodata["minimum_service_factor"] = 0

    # Save modified data
    output_path = model_path / sector / "TechnodataTimeslices.csv"
    technodata.to_csv(output_path, index=False)
    return model_path


PROCESS_PAIR = [("gasCCGT", "windturbine")]


@mark.parametrize("utilization_factors", [(0.1, 1), (1, 0.1)])
@mark.parametrize("process_names", PROCESS_PAIR)
def test_fullsim_timeslices(tmp_path, utilization_factors, process_names):
    sector = "power"
    model_path = setup_test_environment(
        tmp_path, sector, process_names, utilization_factors
    )

    with chdir(tmp_path):
        MCA.factory(model_path / "settings.toml").run()

    mca_capacity = pd.read_csv(tmp_path / "Results/MCACapacity.csv")
    operator = ge if utilization_factors[0] > utilization_factors[1] else le

    tech1_count = len(
        mca_capacity[
            (mca_capacity.sector == sector)
            & (mca_capacity.technology == process_names[0])
        ]
    )
    tech2_count = len(
        mca_capacity[
            (mca_capacity.sector == sector)
            & (mca_capacity.technology == process_names[1])
        ]
    )
    assert operator(tech1_count, tech2_count)


@mark.parametrize(
    "utilization_factors",
    [
        ([0.0001, 0.0001, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        ([1, 1, 1, 1, 1, 1], [1, 1, 0.0001, 0.0001, 1, 1]),
    ],
)
@mark.parametrize("process_names", PROCESS_PAIR)
def test_zero_utilization_factor_supply_timeslice(
    tmp_path, utilization_factors, process_names
):
    sector = "power"
    model_path = setup_test_environment(
        tmp_path, sector, process_names, utilization_factors
    )

    with chdir(tmp_path):
        MCA.factory(model_path / "settings.toml").run()

    power_supply = pd.read_csv(tmp_path / "Results/Power_Supply.csv").reset_index()
    zero_utilization_indices = [
        i for i, factor in enumerate(utilization_factors) if factor == 0
    ]

    zero_output = power_supply[
        power_supply.timeslice.isin(zero_utilization_indices)
        & (power_supply.technology == process_names)
    ]
    assert len(zero_output) == 0


@mark.parametrize("utilization_factors", [(0, 1), (1, 0)])
@mark.parametrize("process_names", PROCESS_PAIR)
def test_all_zero_fatal_error(tmp_path, utilization_factors, process_names):
    sector = "power"
    model_path = setup_test_environment(
        tmp_path, sector, process_names, utilization_factors
    )

    with chdir(tmp_path), raises(ValueError):
        MCA.factory(model_path / "settings.toml").run()
