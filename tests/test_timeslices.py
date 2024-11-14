"""Test timeslice utilities."""

from pytest import approx, fixture, raises
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


def test_read_timeslices():
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


@fixture
def non_timesliced_dataarray():
    return DataArray([1, 2, 3], dims=["x"])


def test_broadcast_timeslice(non_timesliced_dataarray, timeslice, timeslice_dataarray):
    from muse.timeslices import broadcast_timeslice

    out = broadcast_timeslice(non_timesliced_dataarray)

    # Check that timeslicing in output matches the global scheme
    assert out.timeslice.equals(timeslice.timeslice)

    # Check that all timeslices in the output are equal to each other
    assert (out.diff(dim="timeslice") == 0).all()

    # Check that all values in the output are equal to the input
    assert all(
        (out.isel(timeslice=i) == non_timesliced_dataarray).all()
        for i in range(out.sizes["timeslice"])
    )

    # Calling on an already timesliced array: the input should be returned unchanged
    out2 = broadcast_timeslice(out)
    assert out2.equals(out)

    # Calling with an incompatible timeslicing scheme: ValueError should be raised
    with raises(ValueError):
        broadcast_timeslice(out, ts=timeslice_dataarray)


def test_distribute_timeslice(non_timesliced_dataarray, timeslice, timeslice_dataarray):
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    out = distribute_timeslice(non_timesliced_dataarray)

    # Check that timeslicing in output matches the global scheme
    assert out.timeslice.equals(timeslice.timeslice)

    # Check that all values are proportional to timeslice lengths
    out_proportions = out / broadcast_timeslice(out.sum("timeslice"))
    ts_proportions = timeslice / broadcast_timeslice(timeslice.sum("timeslice"))
    assert abs(out_proportions - ts_proportions).max() < 1e-6

    # Check that the sum across timeslices is equal to the input
    assert (out.sum("timeslice") == approx(non_timesliced_dataarray)).all()

    # Calling on an already timesliced array: the input should be returned unchanged
    out2 = distribute_timeslice(out)
    assert out2.equals(out)

    # Calling with an incompatible timeslicing scheme: ValueError should be raised
    with raises(ValueError):
        distribute_timeslice(out, ts=timeslice_dataarray)


def test_compress_timeslice(non_timesliced_dataarray, timeslice, timeslice_dataarray):
    # Test 1: without specifying level
    # Assert output matches input

    # Test 2: invalid operation
    # Assert ValueError is raised

    # Test 3: sum operation
    # Assert timeslicing is the correct level
    # Assert sum of output equals sum of input

    # Test 4: mean operation
    # Assert timeslicing is the correct level
    # Assert weighted mean of output equals weighted mean of input

    pass


def test_expand_timeslice(timeslice_dataarray):
    # Test 1: calling on an already expanded array
    # Assert the input is returned unchanged

    # Test 2: invalid operation
    # Assert ValueError is raised

    # Test 3: broadcast operation
    # Assert timeslicing matches the global scheme
    # Assert all values are equal to each other
    # Assert all values in the output are equal to the input

    # Test 4: distribute operation
    # Assert timeslicing matches the global scheme
    # Assert all values are in proportion to timeslice length
    # Assert sum of output across timeslices is equal to the input

    pass
