"""Test timeslice utilities."""

import numpy as np
from pytest import approx, fixture, raises
from xarray import DataArray


@fixture
def non_timesliced_dataarray():
    return DataArray([1, 2, 3], dims=["x"])


@fixture
def timeslice():
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
    return ts


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


def test_drop_timeslice(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, drop_timeslice

    # Test on array with timeslice data
    timesliced_dataarray = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice)
    dropped = drop_timeslice(timesliced_dataarray)
    coords_to_check = {"timeslice", "month", "day", "hour"}
    assert coords_to_check.issubset(timesliced_dataarray.coords)
    assert not coords_to_check.intersection(dropped.coords)

    # Test on arrays without timeslice data
    assert drop_timeslice(non_timesliced_dataarray).equals(non_timesliced_dataarray)
    assert drop_timeslice(dropped).equals(dropped)


def test_broadcast_timeslice(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, compress_timeslice

    # Broadcast array to different levels of granularity
    for level in ["month", "day", "hour"]:
        out = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice, level=level)
        target_timeslices = compress_timeslice(
            timeslice, ts=timeslice, level=level, operation="sum"
        )

        # Check that timeslicing in output matches the global scheme
        assert out.timeslice.equals(target_timeslices.timeslice)

        # Check that all timeslices in the output are equal to each other
        assert (out.diff(dim="timeslice") == 0).all()

        # Check that all values in the output are equal to the input
        assert all(
            (out.isel(timeslice=i) == non_timesliced_dataarray).all()
            for i in range(out.sizes["timeslice"])
        )

    # Calling on a fully timesliced array: the input should be returned unchanged
    out2 = broadcast_timeslice(out, ts=timeslice)
    assert out2.equals(out)

    # Calling on an array with inappropriate timeslicing: ValueError should be raised
    with raises(ValueError):
        broadcast_timeslice(
            compress_timeslice(out, ts=timeslice, level="day"), ts=timeslice
        )


def test_distribute_timeslice(non_timesliced_dataarray, timeslice):
    from muse.timeslices import (
        broadcast_timeslice,
        compress_timeslice,
        distribute_timeslice,
    )

    # Distribute array to different levels of granularity
    for level in ["month", "day", "hour"]:
        out = distribute_timeslice(non_timesliced_dataarray, ts=timeslice, level=level)
        target_timeslices = compress_timeslice(
            timeslice, ts=timeslice, level=level, operation="sum"
        )

        # Check that timeslicing in output matches the global scheme
        assert out.timeslice.equals(target_timeslices.timeslice)

        # Check that all values are proportional to timeslice lengths
        out_proportions = out / broadcast_timeslice(
            out.sum("timeslice"), ts=timeslice, level=level
        )
        ts_proportions = target_timeslices / broadcast_timeslice(
            target_timeslices.sum("timeslice"), ts=timeslice, level=level
        )
        assert abs(out_proportions - ts_proportions).max() < 1e-6

        # Check that the sum across timeslices is equal to the input
        assert (out.sum("timeslice") == approx(non_timesliced_dataarray)).all()

    # Calling on a fully timesliced array: the input should be returned unchanged
    out2 = distribute_timeslice(out, ts=timeslice)
    assert out2.equals(out)

    # Calling on an array with inappropraite timeslicing: ValueError should be raised
    with raises(ValueError):
        distribute_timeslice(
            compress_timeslice(out, ts=timeslice, level="day"), ts=timeslice
        )


def test_compress_timeslice(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, compress_timeslice, get_level

    # Create timesliced dataarray for compressing
    timesliced_dataarray = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice)

    # Compress array to different levels of granularity
    for level in ["month", "day", "hour"]:
        # Sum operation
        out = compress_timeslice(
            timesliced_dataarray, ts=timeslice, operation="sum", level=level
        )
        assert get_level(out) == level
        assert (
            out.sum("timeslice") == approx(timesliced_dataarray.sum("timeslice"))
        ).all()

        # Mean operation
        out = compress_timeslice(
            timesliced_dataarray, ts=timeslice, operation="mean", level=level
        )
        assert get_level(out) == level
        assert (
            out.mean("timeslice") == approx(timesliced_dataarray.mean("timeslice"))
        ).all()  # NB in general this should be a weighted mean, but this works here
        # because the data is equal in every timeslice

    # Calling without specifying a level: the input should be returned unchanged
    out = compress_timeslice(timesliced_dataarray, ts=timeslice)
    assert out.equals(timesliced_dataarray)

    # Calling with an invalid level: ValueError should be raised
    with raises(ValueError):
        compress_timeslice(timesliced_dataarray, ts=timeslice, level="invalid")

    # Calling with an invalid operation: ValueError should be raised
    with raises(ValueError):
        compress_timeslice(
            timesliced_dataarray, ts=timeslice, level="day", operation="invalid"
        )


def test_expand_timeslice(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, expand_timeslice

    # Different starting points for expansion
    for level in ["month", "day", "hour"]:
        timesliced_dataarray = broadcast_timeslice(
            non_timesliced_dataarray, ts=timeslice, level=level
        )

        # Broadcast operation
        out = expand_timeslice(
            timesliced_dataarray, ts=timeslice, operation="broadcast"
        )
        assert out.timeslice.equals(timeslice.timeslice)
        assert (
            out.mean("timeslice") == approx(timesliced_dataarray.mean("timeslice"))
        ).all()

        # Distribute operation
        out = expand_timeslice(
            timesliced_dataarray, ts=timeslice, operation="distribute"
        )
        assert out.timeslice.equals(timeslice.timeslice)
        assert (
            out.sum("timeslice") == approx(timesliced_dataarray.sum("timeslice"))
        ).all()

    # Calling on an already expanded array: the input should be returned unchanged
    out2 = expand_timeslice(out, ts=timeslice)
    assert out.equals(out2)

    # Calling with an invalid operation: ValueError should be raised
    with raises(ValueError):
        timesliced_dataarray = broadcast_timeslice(
            non_timesliced_dataarray, ts=timeslice, level="month"
        )
        expand_timeslice(timesliced_dataarray, ts=timeslice, operation="invalid")


def test_get_level(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, get_level

    for level in ["month", "day", "hour"]:
        timesliced_dataarray = broadcast_timeslice(
            non_timesliced_dataarray, ts=timeslice, level=level
        )
        assert get_level(timesliced_dataarray) == level

    # Should raise error with non-timesliced array
    with raises(ValueError):
        get_level(non_timesliced_dataarray)


def test_sort_timeslices(non_timesliced_dataarray, timeslice):
    from muse.timeslices import broadcast_timeslice, sort_timeslices

    # Finest timeslice level -> should match ordering of `timeslice`
    timesliced_dataarray = broadcast_timeslice(
        non_timesliced_dataarray, ts=timeslice, level="hour"
    )
    sorted = sort_timeslices(timesliced_dataarray, timeslice)
    assert sorted.timeslice.equals(timeslice.timeslice)
    assert not sorted.timeslice.equals(
        timesliced_dataarray.sortby("timeslice").timeslice
    )  # but could be true if the timeslices in `timeslice` are in alphabetical order

    # Coarser timeslice level -> should match xarray sortby
    timesliced_dataarray = broadcast_timeslice(
        non_timesliced_dataarray, ts=timeslice, level="month"
    )
    sorted = sort_timeslices(timesliced_dataarray, timeslice)
    assert sorted.timeslice.equals(timesliced_dataarray.sortby("timeslice").timeslice)


def test_timeslice_max(non_timesliced_dataarray):
    from muse.timeslices import broadcast_timeslice, read_timeslices, timeslice_max

    # With two equal timeslice lengths, this should be equivalent to max * 2
    ts = read_timeslices(
        """
            [timeslices]
            winter.weekday.night = 396
            winter.weekday.morning = 396
            """
    )
    timesliced_dataarray = broadcast_timeslice(non_timesliced_dataarray, ts=ts)
    timesliced_dataarray = timesliced_dataarray + np.random.rand(
        *timesliced_dataarray.shape
    )
    timeslice_max_dataarray = timeslice_max(timesliced_dataarray, ts=ts)
    assert timeslice_max_dataarray.equals(timesliced_dataarray.max("timeslice") * 2)
