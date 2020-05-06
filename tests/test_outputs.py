"""Test saving outputs to file."""
from pathlib import Path

from pytest import importorskip, mark

from muse.outputs import register_output_quantity, save_output


@register_output_quantity
def streetcred(*args, **kwargs):
    from xarray import DataArray
    from numpy.random import randint

    return DataArray(
        randint(0, 5, (3, 2)),
        coords={
            "year": [2010, 2015],
            "technology": ("asset", ["a", "b", "c"]),
            "installed": ("asset", [2010, 2011, 2011]),
        },
        dims=("asset", "year"),
    )


def test_save_with_dir(tmpdir):
    from pandas import read_csv

    path = Path(tmpdir) / "results" / "stuff"
    config = {
        "filename": path / "{Sector}{year}{Quantity}.csv",
        "quantity": "streetcred",
        "sector": "yoyo",
        "year": 2010,
    }
    # can use None because we **know** none of the arguments are used here
    result = save_output(None, None, None, **config)
    assert result == path / "Yoyo2010Streetcred.csv"
    assert result.is_file()
    read_csv(result)


def test_overwrite(tmpdir):
    from pytest import raises

    path = Path(tmpdir) / "results" / "stuff"
    config = {
        "filename": path / "{Sector}{year}{Quantity}.csv",
        "quantity": "streetcred",
        "sector": "yoyo",
        "year": 2010,
    }
    # can use None because we **know** none of the arguments are used here
    result = save_output(None, None, None, **config)
    assert result == path / "Yoyo2010Streetcred.csv"
    assert result.is_file()

    with raises(IOError):
        save_output(None, None, None, **config)

    config["overwrite"] = False
    with raises(IOError):
        save_output(None, None, None, **config)

    config["overwrite"] = True
    save_output(None, None, None, **config)


def test_save_with_path_to_nc_with_suffix(tmpdir):
    from xarray import open_dataset

    path = Path(tmpdir) / "results" / "stuff"
    # can use None because we **know** none of the arguments are used here
    config = {
        "filename": path / "{Sector}{year}{Quantity}{suffix}",
        "quantity": "streetcred",
        "sector": "yoyo",
        "suffix": "nc",
        "year": 2010,
    }
    result = save_output(None, None, None, **config)
    assert result == path / "Yoyo2010Streetcred.nc"
    assert result.is_file()
    open_dataset(result)


def test_save_with_path_to_nc_with_sink(tmpdir):
    from xarray import open_dataset

    path = Path(tmpdir) / "results" / "stuff"
    # can use None because we **know** none of the arguments are used here
    config = {
        "filename": path / "{sector}{year}{quantity}.csv",
        "quantity": "streetcred",
        "sector": "yoyo",
        "sink": "nc",
        "year": 2010,
    }
    result = save_output(None, None, None, **config)
    assert result == path / "yoyo2010streetcred.csv"
    assert result.is_file()
    open_dataset(result)


def test_save_with_fullpath_to_excel_with_sink(tmpdir):
    from pandas import read_excel

    importorskip("openpyxl")

    path = Path(tmpdir) / "results" / "stuff" / "this.xlsx"
    # can use None because we **know** none of the arguments are used here
    config = {"filename": path, "quantity": "streetcred", "sink": "xlsx"}
    result = save_output(config, None, None, None)
    assert result == path
    assert result.is_file()
    read_excel(result)


@mark.sgidata
@mark.legacy
def test_from_sector_with_single_string(buildings, market, tmpdir):
    from muse.defaults import DEFAULT_OUTPUT_DIRECTORY
    from muse.outputs.sector import factory

    with tmpdir.as_cwd():
        output_func = factory("streetcred", sector_name="Residential")
        output_func(buildings.capacity, market, buildings.technologies)
        assert (
            Path(tmpdir) / DEFAULT_OUTPUT_DIRECTORY / "Residential2010Streetcred.csv"
        ).exists()
        assert (
            Path(tmpdir) / DEFAULT_OUTPUT_DIRECTORY / "Residential2010Streetcred.csv"
        ).is_file()


@mark.sgidata
@mark.legacy
def test_from_sector_with_directory(buildings, market, tmpdir):
    from muse.outputs.sector import factory

    output_func = factory(
        {"quantity": "streetcred", "filename": tmpdir / "abc.csv"},
        sector_name="Residential",
    )
    output_func(buildings.capacity, market, buildings.technologies)
    assert (Path(tmpdir) / "abc.csv").exists()
    assert (Path(tmpdir) / "abc.csv").is_file()
