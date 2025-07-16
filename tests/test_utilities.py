import numpy as np
import xarray as xr
from pytest import approx, mark, raises


def make_array(array):
    """Create a random DataArray with the same dimensions and coordinates as input."""
    data = np.random.randint(1, 5, len(array)) * (
        np.random.randint(0, 10, len(array)) > 8
    )
    return xr.DataArray(data, dims=array.dims, coords=array.coords)


def assert_shape_and_dims(actual, expected_shape, expected_dims=None):
    """Helper to verify array shape and dimensions.

    Works with both numpy arrays and xarray objects.
    For numpy arrays, only shape is checked.
    For xarray objects, both shape and dimensions are checked.
    """
    assert actual.ndim == len(expected_shape)
    assert actual.shape == expected_shape
    if expected_dims is not None and hasattr(actual, "dims"):
        assert set(actual.dims) == set(expected_dims)


@mark.parametrize(
    "coordinates",
    [("technology", "installed", "region"), ("technology", "installed"), ("region",)],
)
def test_reduce_assets(coordinates: tuple, capacity: xr.DataArray):
    """Test reducing assets along specified coordinates."""
    from muse.utilities import reduce_assets

    actual = reduce_assets(capacity, coords=coordinates)

    uniques = set(zip(*(getattr(capacity, d).values for d in coordinates)))
    assert len(uniques) == len(actual.asset)
    assert uniques == set(zip(*(getattr(actual, d).values for d in coordinates)))

    for index in actual.asset:
        condition = True
        for coord in coordinates:
            condition = condition & (getattr(capacity, coord) == getattr(index, coord))
        expected = capacity.isel(asset=condition).sum("asset")
        assert actual.isel(asset=index).values == approx(expected.values)


def test_reduce_assets_with_zero_size(capacity: xr.DataArray):
    """Test reducing assets with empty selection."""
    from muse.utilities import reduce_assets

    x = capacity.sel(asset=[])
    actual = reduce_assets(x)
    assert (actual == x).all()


def test_broadcast_over_assets(technologies, capacity):
    """Test broadcasting over assets with different year settings."""
    from muse.utilities import broadcast_over_assets

    result1 = broadcast_over_assets(technologies, capacity, installed_as_year=True)
    assert set(result1.dims) == {"asset", "commodity"}
    assert (result1.asset == capacity.asset).all()

    result2 = broadcast_over_assets(technologies, capacity, installed_as_year=False)
    assert set(result2.dims) == {"asset", "commodity", "year"}
    assert (result2.asset == capacity.asset).all()

    # Template without "asset" dimensions (TODO: need to make the function stricter)
    # with raises(AssertionError):
    #     broadcast_over_assets(technologies, technologies)


def create_test_array(shape, scale=20, offset=-10):
    """Create a random integer array for testing."""
    return (np.random.rand(*shape) * scale + offset).astype(int)


def test_tupled_dimension_no_tupling():
    """Test tupled dimension conversion without actual tupling."""
    from muse.utilities import tupled_dimension

    array = create_test_array((10, 1))
    actual = tupled_dimension(array, 1)
    assert_shape_and_dims(actual, array.shape[:-1])
    assert actual == approx(array.reshape(array.shape[0]))

    array = create_test_array((10, 1, 5))
    actual = tupled_dimension(array, 1)
    assert_shape_and_dims(actual, (array.shape[0], array.shape[2]))
    assert actual == approx(array.reshape(array.shape[0], array.shape[2]))


def verify_tupled_output(actual, array, axis, indices):
    """Helper to verify tupled dimension output."""
    for idx in indices:
        if len(idx) == 1:
            i = idx[0]
            assert len(actual[i]) == array.shape[axis]
            assert isinstance(actual[i], tuple)
            assert tuple(array[i, :] if axis == 1 else array[:, i]) == actual[i]
        else:
            i, j = idx
            assert len(actual[i, j]) == array.shape[axis]
            assert isinstance(actual[i, j], tuple)
            if axis == 0:
                assert tuple(array[:, i, j]) == actual[i, j]
            elif axis == 1:
                assert tuple(array[i, :, j]) == actual[i, j]
            else:
                assert tuple(array[i, j, :]) == actual[i, j]


def test_tupled_dimension_2d():
    """Test tupled dimension conversion for 2D arrays."""
    from muse.utilities import tupled_dimension

    array = create_test_array((10, 3))

    for axis in (0, 1):
        actual = tupled_dimension(array, axis)
        expected_shape = (array.shape[1],) if axis == 0 else (array.shape[0],)
        assert_shape_and_dims(actual, expected_shape)
        verify_tupled_output(
            actual, array, axis, [(i,) for i in range(expected_shape[0])]
        )


def test_tupled_dimension_3d():
    """Test tupled dimension conversion for 3D arrays."""
    from muse.utilities import tupled_dimension

    array = create_test_array((10, 3, 5))

    for axis in (0, 1, 2):
        actual = tupled_dimension(array, axis)
        if axis == 0:
            expected_shape = (array.shape[1], array.shape[2])
        elif axis == 1:
            expected_shape = (array.shape[0], array.shape[2])
        else:
            expected_shape = (array.shape[0], array.shape[1])
        assert_shape_and_dims(actual, expected_shape)
        indices = [
            (i, j) for i in range(expected_shape[0]) for j in range(expected_shape[1])
        ]
        verify_tupled_output(actual, array, axis, indices)


def create_test_objectives(shape=(5, 10)):
    """Create test objectives dataset."""
    objectives = xr.Dataset()
    for var in ["a", "b", "c"]:
        objectives[var] = ("asset", "replacement"), np.random.rand(*shape) * 10 - 5
    objectives["asset"] = np.random.choice(
        objectives.replacement, len(objectives.asset), replace=False
    )
    return objectives


def create_test_binsizes():
    """Create test binsizes dataset."""
    return xr.Dataset(
        {
            "a": np.random.rand() * 0.1,
            "b": -np.random.rand() * 0.1,
            "c": np.random.rand(),
        }
    )


@mark.parametrize("order", [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])
def test_lexical_comparison(order):
    """Test lexical comparison with and without binning."""
    from muse.utilities import lexical_comparison

    objectives = create_test_objectives()
    binsizes = create_test_binsizes()

    def create_expected(bin_last=True):
        expected = np.zeros(shape=objectives.a.shape, dtype=object)
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                values = []
                for k in range(3):
                    val = objectives[order[k]][i, j] / binsizes[order[k]]
                    values.append(int(np.floor(val)) if bin_last or k < 2 else val)
                expected[i, j] = tuple(values)
        return expected

    # Test with binning
    actual = lexical_comparison(objectives, binsizes[order])
    expected = create_expected(bin_last=True)
    assert actual.shape == expected.shape
    assert (actual.values == expected).all()

    # Test without binning last value
    actual = lexical_comparison(objectives, binsizes[order], bin_last=False)
    expected = create_expected(bin_last=False)
    assert actual.shape == expected.shape
    assert (actual.values == expected).all()


def test_merge_assets():
    """Test merging assets with different coordinate orders."""
    from numpy import arange

    from muse.utilities import interpolate_capacity, merge_assets

    def fake(year, order=("installed", "technology")):
        result = xr.Dataset()
        result["year"] = "year", year
        result["installed"] = "asset", np.random.choice(result.year.values, 10)
        result["technology"] = "asset", np.random.choice(list("abc"), 10)
        result["capacity"] = (
            ("year", "asset"),
            np.random.rand(len(result.year), len(result.asset)),
        )
        return result[["capacity", *order]].set_coords(order).capacity

    order = ["installed", "technology"]
    capa_a = fake(np.arange(2010, 2020, 3, dtype="int64"), order)
    np.random.shuffle(order)
    capa_b = fake(arange(2014, 2024, 2, dtype="int64"), order)
    actual = merge_assets(capa_a, capa_b)

    # Verify coordinate preservation
    for coord in ["installed", "technology", "year"]:
        assert getattr(capa_a, coord).isin(getattr(actual, coord)).all()
        assert getattr(capa_b, coord).isin(getattr(actual, coord)).all()

    # Verify asset uniqueness
    assets = [(i, t) for i, t in zip(actual.installed.values, actual.technology.values)]
    assert len(actual.asset) == len(set(assets))
    all_assets = [
        (i, t) for i, t in zip(capa_a.installed.values, capa_a.technology.values)
    ]
    all_assets.extend(
        (i, t) for i, t in zip(capa_b.installed.values, capa_b.technology.values)
    )
    assert len(actual.asset) == len(set(all_assets))

    # Verify capacity values
    for inst, tech in zip(actual.installed.values, actual.technology.values):
        ab_side = actual.sel(
            asset=((actual.installed == inst) & (actual.technology == tech))
        ).squeeze("asset")
        a_side = interpolate_capacity(
            capa_a.sel(
                asset=((capa_a.installed == inst) & (capa_a.technology == tech))
            ).sum("asset"),
            year=ab_side.year,
        )
        b_side = interpolate_capacity(
            capa_b.sel(
                asset=((capa_b.installed == inst) & (capa_b.technology == tech))
            ).sum("asset"),
            year=ab_side.year,
        )
        assert (ab_side.capacity == approx((a_side + b_side).values)).all()


def test_avoid_repetitions():
    """Test avoiding repetitions in time series data."""
    from muse.utilities import avoid_repetitions

    start, end = 2010, 2010 + 3 * 5
    assets = xr.Dataset()
    assets["year"] = "year", list(range(start, end))
    assets["installed"] = "asset", np.random.choice(assets.year.values, 10)
    assets["technology"] = "asset", np.random.choice(list("abc"), 10)
    assets["capacity"] = (
        ("year", "asset"),
        np.random.randint(0, 10, (len(assets.year), len(assets.asset))),
    )

    # Create repetitions in the data
    for offset in (1, 2):
        assets.capacity.loc[{"year": list(range(start + offset, end, 3))}] = (
            assets.capacity.sel(year=list(range(start, end, 3))).values
        )

    result = assets.sel(year=avoid_repetitions(assets.capacity))
    assert 3 * len(result.year) == 2 * len(assets.year)
    original = result.interp(year=assets.year, method="linear")
    assert (original == assets).all()


def test_check_dimensions():
    """Test dimension checking functionality."""
    from muse.utilities import check_dimensions

    data = xr.DataArray(
        np.random.rand(4, 5),
        dims=["dim1", "dim2"],
        coords={"dim1": range(4), "dim2": range(5)},
    )

    # Test valid case
    check_dimensions(data, required=["dim1"], optional=["dim2"])

    # Test missing required dimension
    with raises(ValueError, match="Missing required dimensions"):
        check_dimensions(data, required=["dim1", "dim3"], optional=["dim2"])

    # Test extra dimension
    with raises(ValueError, match="Extra dimensions"):
        check_dimensions(data, required=["dim1"])


def test_interpolate_technodata():
    """Test interpolate_technodata function for all essential scenarios."""
    from muse.utilities import interpolate_technodata

    # Create test data
    data = xr.Dataset(
        {"efficiency": (["technology", "year"], [[0.8, 0.85, 0.9], [0.7, 0.75, 0.8]])},
        coords={"technology": ["tech1", "tech2"], "year": [2020, 2025, 2030]},
    )

    # Test 1: Basic interpolation
    time_framework = [2022, 2027]
    result = interpolate_technodata(data, time_framework)
    assert set(result.year.values) == {2020, 2022, 2025, 2027, 2030}
    assert "year" in result.efficiency.dims
    assert result.sel(technology="tech1", year=2022).efficiency == approx(0.82)

    # Test 2: Forward extrapolation (time_framework extends beyond data)
    result = interpolate_technodata(data, [2035, 2040])
    assert 2040 in result.year.values
    assert "year" in result.efficiency.dims
    assert result.sel(technology="tech1", year=2035).efficiency == 0.9

    # Test 3: Backward extrapolation (time_framework starts before data)
    result = interpolate_technodata(data, [2010, 2015])
    assert 2010 in result.year.values
    assert "year" in result.efficiency.dims
    assert result.sel(technology="tech1", year=2010).efficiency == 0.8

    # Test 4: Data with only one year
    single_year_data = xr.Dataset(
        {"efficiency": (["technology"], [0.8, 0.7])},
        coords={"technology": ["tech1", "tech2"], "year": [2025]},
    )
    result = interpolate_technodata(single_year_data, [2020, 2025, 2030])
    assert set(result.year.values) == {2020, 2025, 2030}
    assert "year" not in result.efficiency.dims  # underlying data is not duplicated
    for year in result.year.values:
        assert result.sel(technology="tech1", year=year).efficiency == 0.8
        assert result.sel(technology="tech2", year=year).efficiency == 0.7

    # Test 5: Data without year dimension
    no_year_data = xr.Dataset(
        {"efficiency": (["technology"], [0.8, 0.7])},
        coords={"technology": ["tech1", "tech2"]},
    )
    result = interpolate_technodata(no_year_data, [2020, 2025, 2030])
    assert set(result.year.values) == {2020, 2025, 2030}
    assert "year" not in result.efficiency.dims  # underlying data is not duplicated
    for year in result.year.values:
        assert result.sel(technology="tech1", year=year).efficiency == 0.8
        assert result.sel(technology="tech2", year=year).efficiency == 0.7
