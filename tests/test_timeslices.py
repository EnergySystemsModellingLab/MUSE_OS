"""Test timeslice utilities."""

from pytest import fixture
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
    from muse.timeslices import read_timeslices

    return read_timeslices(toml)


@fixture
def timeslice_dataarray(reference):
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


def test_reference_timeslice():
    from toml import loads

    from muse.timeslices import read_timeslices

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

    ts = read_timeslices(inputs)
    assert isinstance(ts, DataArray)
    assert "timeslice" in ts.coords


def test_no_overlap():
    from pytest import raises

    from muse.timeslices import read_timeslices

    with raises(ValueError):
        read_timeslices(
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


def test_drop_timeslice(timeslice_dataarray):
    from muse.timeslices import drop_timeslice

    dropped = drop_timeslice(timeslice_dataarray)
    coords_to_check = {"timeslice", "semester", "week", "day"}
    assert coords_to_check.issubset(timeslice_dataarray.coords)
    assert not coords_to_check.intersection(dropped.coords)

    # Test on arrays without timeslice data
    data_without_timeslice = DataArray([1, 2, 3], dims=["x"])
    assert drop_timeslice(data_without_timeslice).equals(data_without_timeslice)
    assert drop_timeslice(dropped).equals(dropped)
