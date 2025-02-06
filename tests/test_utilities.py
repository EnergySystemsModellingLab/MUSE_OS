import numpy as np
import xarray as xr
from pytest import approx, mark, raises


def make_array(array):
    data = np.random.randint(1, 5, len(array)) * (
        np.random.randint(0, 10, len(array)) > 8
    )
    return xr.DataArray(data, dims=array.dims, coords=array.coords)


@mark.parametrize(
    "coordinates",
    [("technology", "installed", "region"), ("technology", "installed"), ("region",)],
)
def test_reduce_assets(coordinates: tuple, capacity: xr.DataArray):
    from muse.utilities import reduce_assets

    actual = reduce_assets(capacity, coords=coordinates)

    uniques = set(zip(*(getattr(capacity, d).values for d in coordinates)))
    assert len(uniques) == len(actual.asset)
    actual_uniques = set(zip(*(getattr(actual, d).values for d in coordinates)))
    assert uniques == actual_uniques

    for index in actual.asset:
        condition = True
        for coord in coordinates:
            condition = condition & (getattr(capacity, coord) == getattr(index, coord))
        expected = capacity.isel(asset=condition).sum("asset")
        assert actual.isel(asset=index).values == approx(expected.values)


def test_reduce_assets_with_zero_size(capacity: xr.DataArray):
    from muse.utilities import reduce_assets

    x = capacity.sel(asset=[])
    actual = reduce_assets(x)
    assert (actual == x).all()


def test_broadcast_over_assets(technologies, capacity):
    from muse.utilities import broadcast_over_assets

    # Test with installed_as_year = True
    result1 = broadcast_over_assets(technologies, capacity, installed_as_year=True)
    assert set(result1.dims) == {"asset", "commodity"}
    assert (result1.asset == capacity.asset).all()

    # Test with installed_as_year = False
    result2 = broadcast_over_assets(technologies, capacity, installed_as_year=False)
    assert set(result2.dims) == {"asset", "commodity", "year"}
    assert (result2.asset == capacity.asset).all()

    # Template without "asset" dimensions (TODO: need to make the function stricter)
    # with raises(AssertionError):
    #     broadcast_over_assets(technologies, technologies)


def test_tupled_dimension_no_tupling():
    from muse.utilities import tupled_dimension

    array = (np.random.rand(10, 1) * 20 - 10).astype(int)
    actual = tupled_dimension(array, 1)
    assert actual.ndim == 1
    assert actual.shape == array.shape[:-1]
    assert actual == approx(array.reshape(array.shape[0]))

    array = (np.random.rand(10, 1, 5) * 20 - 10).astype(int)
    actual = tupled_dimension(array, 1)
    assert actual.ndim == 2
    assert actual.shape == (array.shape[0], array.shape[2])
    assert actual == approx(array.reshape(array.shape[0], array.shape[2]))


def test_tupled_dimension_2d():
    from muse.utilities import tupled_dimension

    array = (np.random.rand(10, 3) * 20 - 10).astype(int)

    actual = tupled_dimension(array, 1)
    assert actual.ndim == 1
    assert actual.shape == (array.shape[0],)
    for i in range(array.shape[0]):
        assert len(actual[i]) == array.shape[1]
        assert isinstance(actual[i], tuple)
        assert tuple(array[i, :]) == actual[i]

    actual = tupled_dimension(array, 0)
    assert actual.ndim == 1
    assert actual.shape == (array.shape[1],)
    for i in range(array.shape[1]):
        assert len(actual[i]) == array.shape[0]
        assert isinstance(actual[i], tuple)
        assert tuple(array[:, i]) == actual[i]


def test_tupled_dimension_3d():
    from muse.utilities import tupled_dimension

    array = (np.random.rand(10, 3, 5) * 20 - 10).astype(int)

    actual = tupled_dimension(array, 1)
    assert actual.ndim == 2
    assert actual.shape == (array.shape[0], array.shape[2])
    for i in range(array.shape[0]):
        for j in range(array.shape[2]):
            assert len(actual[i, j]) == array.shape[1]
            assert isinstance(actual[i, j], tuple)
            assert tuple(array[i, :, j]) == actual[i, j]

    actual = tupled_dimension(array, 0)
    assert actual.ndim == 2
    assert actual.shape == (array.shape[1], array.shape[2])
    for i in range(array.shape[1]):
        for j in range(array.shape[2]):
            assert len(actual[i, j]) == array.shape[0]
            assert isinstance(actual[i, j], tuple)
            assert tuple(array[:, i, j]) == actual[i, j]

    actual = tupled_dimension(array, 2)
    assert actual.ndim == 2
    assert actual.shape == (array.shape[0], array.shape[1])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            assert len(actual[i, j]) == array.shape[2]
            assert isinstance(actual[i, j], tuple)
            assert tuple(array[i, j, :]) == actual[i, j]


@mark.parametrize("order", [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])
def test_lexical_with_bin(order):
    """Test lexical comparison against hand-constructed tuples."""
    from muse.utilities import lexical_comparison

    objectives = xr.Dataset()
    objectives["a"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["b"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["c"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["asset"] = np.random.choice(
        objectives.replacement, len(objectives.asset), replace=False
    )

    binsizes = xr.Dataset(
        {
            "a": np.random.rand() * 0.1,
            "b": -np.random.rand() * 0.1,
            "c": np.random.rand(),
        }
    )
    expected = np.zeros(shape=objectives.a.shape, dtype=object)
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            expected[i, j] = (
                int(np.floor(objectives[order[0]][i, j] / binsizes[order[0]])),
                int(np.floor(objectives[order[1]][i, j] / binsizes[order[1]])),
                int(np.floor(objectives[order[2]][i, j] / binsizes[order[2]])),
            )

    actual = lexical_comparison(objectives, binsizes[order])
    assert actual.shape == expected.shape
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            assert actual.values[i, j] == expected[i, j]


@mark.parametrize("order", [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])
def test_lexical_nobin(order):
    """Test lexical comparison against hand-constructed tuples."""
    from muse.utilities import lexical_comparison

    objectives = xr.Dataset()
    objectives["a"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["b"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["c"] = ("asset", "replacement"), np.random.rand(5, 10) * 10 - 5
    objectives["asset"] = np.random.choice(
        objectives.replacement, len(objectives.asset), replace=False
    )

    binsizes = xr.Dataset(
        {
            "a": np.random.rand() * 0.1,
            "b": -np.random.rand() * 0.1,
            "c": np.random.rand(),
        }
    )
    expected = np.zeros(shape=objectives.a.shape, dtype=object)
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            expected[i, j] = (
                int(np.floor(objectives[order[0]][i, j] / binsizes[order[0]])),
                int(np.floor(objectives[order[1]][i, j] / binsizes[order[1]])),
                objectives[order[2]][i, j] / binsizes[order[2]],
            )

    actual = lexical_comparison(objectives, binsizes[order], bin_last=False)
    assert actual.shape == expected.shape
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            assert actual.values[i, j] == expected[i, j]


def test_merge_assets():
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
        result = result[["capacity", *order]].set_coords(order)
        return result.capacity

    # checks order of coords does not interfere with merging
    order = ["installed", "technology"]
    capa_a = fake(np.arange(2010, 2020, 3, dtype="int64"), order)
    np.random.shuffle(order)
    capa_b = fake(arange(2014, 2024, 2, dtype="int64"), order)
    actual = merge_assets(capa_a, capa_b)

    assert actual.installed.dtype == capa_a.installed.dtype
    assert capa_a.installed.isin(actual.installed).all()
    assert capa_b.installed.isin(actual.installed).all()
    assert capa_a.technology.isin(actual.technology).all()
    assert capa_b.technology.isin(actual.technology).all()
    assert capa_a.year.isin(actual.year).all()
    assert capa_b.year.isin(actual.year).all()

    assets = [(i, t) for i, t in zip(actual.installed.values, actual.technology.values)]
    assert len(actual.asset) == len(set(assets))
    assets = [
        (i, t) for i, t in zip(capa_a.installed.values, capa_a.technology.values)
    ] + [(i, t) for i, t in zip(capa_b.installed.values, capa_b.technology.values)]
    assert len(actual.asset) == len(set(assets))

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

    assets.capacity.loc[{"year": list(range(start + 1, end, 3))}] = assets.capacity.sel(
        year=list(range(start, end, 3))
    ).values
    assets.capacity.loc[{"year": list(range(start + 2, end, 3))}] = assets.capacity.sel(
        year=list(range(start, end, 3))
    ).values

    result = assets.sel(year=avoid_repetitions(assets.capacity))
    assert 3 * len(result.year) == 2 * len(assets.year)
    original = result.interp(year=assets.year, method="linear")
    assert (original == assets).all()


def test_check_dimensions():
    from muse.utilities import check_dimensions

    data = xr.DataArray(
        np.random.rand(4, 5),
        dims=["dim1", "dim2"],
        coords={"dim1": range(4), "dim2": range(5)},
    )

    # Valid
    check_dimensions(data, required=["dim1"], optional=["dim2"])

    # Missing required
    with raises(ValueError, match="Missing required dimensions"):
        check_dimensions(data, required=["dim1", "dim3"], optional=["dim2"])

    # Extra dimension
    with raises(ValueError, match="Extra dimensions"):
        check_dimensions(data, required=["dim1"])
