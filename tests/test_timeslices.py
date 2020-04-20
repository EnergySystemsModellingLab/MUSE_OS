"""Test timeslice utilities."""
from pytest import approx, fixture, mark
from xarray import DataArray

from muse.timeslices import QuantityType, convert_timeslice


@fixture
def toml():
    return """
        ["timeslices"]
        winter.weekday.day = 10
        winter.weekday.night = 5
        winter.weekend.day = 2
        winter.weekend.night = 2
        winter.weekend.dusk = 1
        summer.weekday.day = 5
        summer.weekday.night = 5
        summer.weekend.day = 2
        summer.weekend.night = 2
        summer.weekend.dusk = 1
        level_names = ["semester", "week", "day"]
        [timeslices.aggregates]
        allday = ["day", "night"]
    """


@fixture
def reference(toml):
    from muse.timeslices import reference_timeslice

    return reference_timeslice(toml)


@fixture
def transforms(toml, reference):
    from muse.timeslices import aggregate_transforms

    return aggregate_transforms(toml, reference)


@fixture
def rough(reference):
    from pandas import MultiIndex

    return DataArray(
        [1, 2, 3],
        coords={
            "timeslice": MultiIndex.from_tuples(
                [
                    ("winter", "weekday", "allday"),
                    ("winter", "weekend", "dusk"),
                    ("summer", "weekend", "night"),
                ],
                names=reference.get_index("timeslice").names,
            )
        },
        dims="timeslice",
    )


def test_convert_extensive_timeslice(reference, rough, transforms):
    z = convert_timeslice(rough, reference, finest=reference, transforms=transforms)
    assert z.shape == reference.shape
    assert z.values == approx(
        [
            float(rough[0] * reference[0] / (reference[0] + reference[1])),
            float(rough[0] * reference[1] / (reference[0] + reference[1])),
            0,
            0,
            float(rough[1]),
            0,
            0,
            0,
            float(rough[2]),
            0,
        ]
    )


def test_convert_intensive_timeslice(reference, rough, transforms):
    z = convert_timeslice(
        rough,
        reference,
        finest=reference,
        transforms=transforms,
        quantity=QuantityType.INTENSIVE,
    )

    assert z.values == approx(
        [
            float(rough[0]),
            float(rough[0]),
            0,
            0,
            float(rough[1]),
            0,
            0,
            0,
            float(rough[2]),
            0,
        ]
    )


@mark.legacy
def test_legacy_timeslices(residential_dir, settings):
    """Test the creation of the legacy sectors."""
    import pandas as pd
    from muse.readers.toml import check_time_slices, read_timeslices
    from muse.timeslices import new_to_old_timeslice

    check_time_slices(settings)
    ts = read_timeslices(
        """
        [timeslice_levels]
        hour = ["all-day"]
        day = ["all-week"]
        """
    ).timeslice

    old_settings = residential_dir / "input" / "MUSEGlobalSettings.csv"
    colname = ["AgLevel", "SN", "Month", "Day", "Hour", "RepresentHours"]
    old_ts = pd.read_csv(old_settings)[colname].dropna()
    old_ts["SN"] = old_ts["SN"].astype(int, copy=False)
    old_ts = old_ts.to_dict("list")

    converted_ts = new_to_old_timeslice(ts)
    assert set(old_ts) == set(converted_ts)
    assert set(old_ts["AgLevel"]) == set(converted_ts["AgLevel"])
    assert set(old_ts["SN"]) == set(converted_ts["SN"])
    assert set(old_ts["Month"]) == set(converted_ts["Month"])
    assert set(old_ts["Day"]) == set(converted_ts["Day"])
    assert set(old_ts["Hour"]) == set(converted_ts["Hour"])


def test_reference_timeslice():
    from toml import loads
    from muse.timeslices import reference_timeslice

    inputs = loads(
        """
        [timeslices]
        winter.weekday.night = 396
        winter.weekday.morning = 396
        winter.weekday.afternoon = 264
        winter.weekend.night = 156
        winter.weekend.morning = 156
        winter.weekend.afternoon = 156
        springautumn.weekday.night = 792
        springautumn.weekday.morning = 792
        springautumn.weekday.afternoon = 528
        springautumn.weekend.night = 300
        springautumn.weekend.morning = 300
        springautumn.weekend.afternoon = 300
        summer.weekday.night = 396
        summer.weekday.morning  = 396
        summer.weekday.afternoon = 264
        summer.weekend.night = 150
        summer.weekend.morning = 150
        summer.weekend.afternoon = 150
        """
    )

    ts = reference_timeslice(inputs)
    assert isinstance(ts, DataArray)
    assert "timeslice" in ts.coords


def test_no_overlap():
    from pytest import raises
    from muse.timeslices import reference_timeslice

    with raises(ValueError):
        reference_timeslice(
            """
            [timeslices]
            winter.weekday.night = 396
            winter.weekday.morning = 396
            winter.weekday.weekend = 156
            winter.weekend.night = 156
            winter.weekend.morning = 156
            winter.weekend.weekend = 156
            """
        )


def test_aggregate_transforms_no_aggregates():
    from typing import Dict
    from itertools import product
    from numpy import ndarray, zeros
    from muse.timeslices import reference_timeslice, aggregate_transforms

    reference = reference_timeslice(
        """
        [timeslices]
        spring.weekday = 396
        spring.weekend = 396
        autumn.weekday = 396
        autumn.weekend = 156
        """
    )

    vectors = aggregate_transforms(timeslice=reference)
    assert isinstance(vectors, Dict)
    assert set(vectors) == set(product(["spring", "autumn"], ["weekday", "weekend"]))
    for i in range(reference.shape[0]):
        index = reference.timeslice[i].values.tolist()
        vector = vectors[index]
        assert isinstance(vector, ndarray)
        expected = zeros(reference.shape, dtype=int)
        expected[i] = 1
        assert vector == approx(expected)


def test_aggregate_transforms_with_aggregates():
    from typing import Dict
    from itertools import product
    from toml import loads
    from muse.timeslices import reference_timeslice, aggregate_transforms

    toml = loads(
        """
        [timeslices]
        spring.weekday.day = 396
        spring.weekday.night = 396
        spring.weekend.day = 156
        spring.weekend.night = 156
        summer.weekday.day = 396
        summer.weekday.night = 396
        summer.weekend.day = 156
        summer.weekend.night = 156
        autumn.weekday.day = 396
        autumn.weekday.night = 396
        autumn.weekend.day = 156
        autumn.weekend.night = 156
        winter.weekday.day = 396
        winter.weekday.night = 396
        winter.weekend.day = 156
        winter.weekend.night = 156

        [timeslices.aggregates]
        springautumn = ["spring", "autumn"]
        allday = ["day", "night"]
        week = ["weekday", "weekend"]
        """
    )
    reference = reference_timeslice(toml)

    vectors = aggregate_transforms(toml, reference)
    assert isinstance(vectors, Dict)
    assert set(vectors) == set(
        product(
            ["winter", "spring", "summer", "autumn", "springautumn"],
            ["weekend", "weekday", "week"],
            ["day", "night", "allday"],
        )
    )

    def to_bitstring(x):
        return "".join(x.astype(str))

    assert to_bitstring(vectors[("spring", "weekday", "night")]) == "0100000000000000"
    assert to_bitstring(vectors[("autumn", "weekday", "night")]) == "0000000001000000"
    assert to_bitstring(vectors[("spring", "weekend", "night")]) == "0001000000000000"
    assert to_bitstring(vectors[("autumn", "weekend", "night")]) == "0000000000010000"
    assert (
        to_bitstring(vectors[("springautumn", "weekday", "night")])
        == "0100000001000000"
    )
    assert to_bitstring(vectors[("spring", "week", "night")]) == "0101000000000000"
    assert (
        to_bitstring(vectors[("springautumn", "week", "night")]) == "0101000001010000"
    )
