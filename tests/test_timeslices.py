"""Test timeslice utilities."""

import numpy as np
from pytest import approx, fixture, raises
from xarray import DataArray

from muse.timeslices import (
    broadcast_timeslice,
    compress_timeslice,
    distribute_timeslice,
    drop_timeslice,
    expand_timeslice,
    get_level,
    read_timeslices,
    sort_timeslices,
    timeslice_max,
)

# Constants
TIMESLICE_LEVELS = ["month", "day", "hour"]
TIMESLICE_COORDS = {"timeslice", "month", "day", "hour"}


@fixture
def non_timesliced_dataarray():
    """Create a simple non-timesliced DataArray for testing."""
    return DataArray([1, 2, 3], dims=["x"])


@fixture
def timeslice_toml():
    """TOML configuration for timeslice testing."""
    return """
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


@fixture(params=[False, True])
def timeslice(request, timeslice_toml):
    """Generate a timeslice DataArray.

    Creates two versions:
    - A one-dimensional DataArray with a single "timeslice" dimension
    - A two-dimensional DataArray with an additional "x" dimension and randomized
     weights
    """
    from toml import loads

    ts = read_timeslices(loads(timeslice_toml))

    if request.param:
        ts = ts.expand_dims({"x": 3})
        ts = ts + np.random.randint(0, 10, ts.shape)

    return ts


def test_no_overlap():
    """Test that overlapping timeslice definitions raise ValueError."""
    invalid_toml = """
    [timeslices]
    winter.weekday.night = 396
    winter.weekday.morning = 396
    winter.weekday.weekend = 156
    winter.weekend.night = 156
    winter.weekend.morning = 156
    winter.weekend.weekend = 156
    """
    with raises(ValueError):
        read_timeslices(invalid_toml)


def test_drop_timeslice(non_timesliced_dataarray, timeslice):
    """Test dropping timeslice coordinates."""
    timesliced = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice)
    dropped = drop_timeslice(timesliced)

    assert TIMESLICE_COORDS.issubset(timesliced.coords)
    assert not TIMESLICE_COORDS.intersection(dropped.coords)
    assert drop_timeslice(non_timesliced_dataarray).equals(non_timesliced_dataarray)
    assert drop_timeslice(dropped).equals(dropped)


def test_broadcast_timeslice(non_timesliced_dataarray, timeslice):
    """Test broadcasting arrays to different timeslice granularities."""
    for level in TIMESLICE_LEVELS:
        out = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice, level=level)
        target = compress_timeslice(
            timeslice, ts=timeslice, level=level, operation="sum"
        )

        assert out.timeslice.equals(target.timeslice)
        assert (out.diff(dim="timeslice") == 0).all()
        assert all(
            (out.isel(timeslice=i) == non_timesliced_dataarray).all()
            for i in range(out.sizes["timeslice"])
        )

    # Test broadcasting an already timesliced array (should return unchanged)
    out2 = broadcast_timeslice(out, ts=timeslice)
    assert out2.equals(out)

    # Test broadcasting with incompatible timeslice levels
    with raises(ValueError):
        broadcast_timeslice(
            compress_timeslice(out, ts=timeslice, level="day"), ts=timeslice
        )


def test_distribute_timeslice(non_timesliced_dataarray, timeslice):
    """Test distributing arrays across timeslices."""
    for level in TIMESLICE_LEVELS:
        out = distribute_timeslice(non_timesliced_dataarray, ts=timeslice, level=level)
        target = compress_timeslice(
            timeslice, ts=timeslice, level=level, operation="sum"
        )

        assert out.timeslice.equals(target.timeslice)

        # Check proportionality
        out_prop = out / broadcast_timeslice(
            out.sum("timeslice"), ts=timeslice, level=level
        )
        ts_prop = target / broadcast_timeslice(
            target.sum("timeslice"), ts=timeslice, level=level
        )
        assert abs(out_prop - ts_prop).max() < 1e-6

        assert (out.sum("timeslice") == approx(non_timesliced_dataarray)).all()

    # Test distributing an already timesliced array (should return unchanged)
    assert distribute_timeslice(out, ts=timeslice).equals(out)

    # Test distributing with incompatible timeslice levels
    with raises(ValueError):
        distribute_timeslice(
            compress_timeslice(out, ts=timeslice, level="day"), ts=timeslice
        )


def test_compress_timeslice(non_timesliced_dataarray, timeslice):
    """Test compressing timesliced arrays."""
    timesliced = broadcast_timeslice(non_timesliced_dataarray, ts=timeslice)

    for level in TIMESLICE_LEVELS:
        for operation in ["sum", "mean"]:
            out = compress_timeslice(
                timesliced, ts=timeslice, operation=operation, level=level
            )
            assert get_level(out) == level

            if operation == "sum":
                assert (
                    out.sum("timeslice") == approx(timesliced.sum("timeslice"))
                ).all()
            else:  # mean
                assert (
                    out.mean("timeslice") == approx(timesliced.mean("timeslice"))
                ).all()

    # Test compressing without specifying a level (should return unchanged)
    assert compress_timeslice(timesliced, ts=timeslice).equals(timesliced)

    # Test compressing with invalid level name
    with raises(ValueError):
        compress_timeslice(timesliced, ts=timeslice, level="invalid")

    # Test compressing with invalid operation type
    with raises(ValueError):
        compress_timeslice(timesliced, ts=timeslice, level="day", operation="invalid")


def test_expand_timeslice(non_timesliced_dataarray, timeslice):
    """Test expanding timesliced arrays."""
    for level in TIMESLICE_LEVELS:
        timesliced = broadcast_timeslice(
            non_timesliced_dataarray, ts=timeslice, level=level
        )

        for operation in ["broadcast", "distribute"]:
            out = expand_timeslice(timesliced, ts=timeslice, operation=operation)
            assert out.timeslice.equals(timeslice.timeslice)

            if operation == "broadcast":
                assert (
                    out.mean("timeslice") == approx(timesliced.mean("timeslice"))
                ).all()
            else:  # distribute
                assert (
                    out.sum("timeslice") == approx(timesliced.sum("timeslice"))
                ).all()

    # Test expanding an already expanded array (should return unchanged)
    assert expand_timeslice(out, ts=timeslice).equals(out)

    # Test expanding with invalid operation type
    timesliced = broadcast_timeslice(
        non_timesliced_dataarray, ts=timeslice, level="month"
    )
    with raises(ValueError):
        expand_timeslice(timesliced, ts=timeslice, operation="invalid")


def test_get_level(non_timesliced_dataarray, timeslice):
    """Test getting timeslice level."""
    for level in TIMESLICE_LEVELS:
        timesliced = broadcast_timeslice(
            non_timesliced_dataarray, ts=timeslice, level=level
        )
        assert get_level(timesliced) == level

    with raises(ValueError):
        get_level(non_timesliced_dataarray)


def test_sort_timeslices(non_timesliced_dataarray, timeslice):
    """Test sorting timeslices."""
    # Test hour level
    timesliced = broadcast_timeslice(
        non_timesliced_dataarray, ts=timeslice, level="hour"
    )
    sorted_data = sort_timeslices(timesliced, timeslice)
    assert sorted_data.timeslice.equals(timeslice.timeslice)

    # Test month level
    timesliced = broadcast_timeslice(
        non_timesliced_dataarray, ts=timeslice, level="month"
    )
    sorted_data = sort_timeslices(timesliced, timeslice)
    assert sorted_data.timeslice.equals(timesliced.sortby("timeslice").timeslice)


def test_timeslice_max(non_timesliced_dataarray):
    """Test timeslice maximum calculation."""
    ts = read_timeslices("""
        [timeslices]
        winter.weekday.night = 396
        winter.weekday.morning = 396
    """)

    timesliced = broadcast_timeslice(non_timesliced_dataarray, ts=ts)
    timesliced = timesliced + np.random.rand(*timesliced.shape)
    max_val = timeslice_max(timesliced, ts=ts)
    assert max_val.equals(timesliced.max("timeslice") * 2)
