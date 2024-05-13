"""Test timeslice utilities."""

from muse.timeslices import QuantityType, convert_timeslice
from pytest import approx, fixture
from xarray import DataArray


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


def test_reference_timeslice():
    from muse.timeslices import reference_timeslice
    from toml import loads

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
    from muse.timeslices import reference_timeslice
    from pytest import raises

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
    from itertools import product
    from typing import Dict

    from muse.timeslices import aggregate_transforms, reference_timeslice
    from numpy import ndarray, zeros

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
    from itertools import product
    from typing import Dict

    from muse.timeslices import aggregate_transforms, reference_timeslice
    from toml import loads

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
