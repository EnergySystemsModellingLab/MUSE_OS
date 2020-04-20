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
    result = save_output(config, None, None, None)
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
    result = save_output(config, None, None, None)
    assert result == path / "Yoyo2010Streetcred.csv"
    assert result.is_file()

    with raises(IOError):
        result = save_output(config, None, None, None)

    config["overwrite"] = False
    with raises(IOError):
        result = save_output(config, None, None, None)

    config["overwrite"] = True
    save_output(config, None, None, None)


def test_save_with_path_to_nc_with_suffix(tmpdir):
    from xarray import open_dataset

    path = Path(tmpdir) / "results" / "stuff"
    # can use None because we **know** none of the arguments are used here
    config = {
        "filename": path / "{Sector}{year}{Quantity}{suffix}",
        "quantity": "streetcred",
        "sector": "yoyo",
        "year": 2010,
        "suffix": "nc",
    }
    result = save_output(config, None, None, None)
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
        "year": 2010,
        "sink": "nc",
    }
    result = save_output(config, None, None, None)
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
def test_from_sector_with_single_string(buildings, market, tmpdir):
    from os import chdir
    from muse.defaults import DEFAULT_OUTPUT_DIRECTORY
    from muse.outputs import factory

    cwd = Path.cwd()
    try:
        chdir(tmpdir)
        output_func = factory("streetcred")
        output_func(
            buildings.capacity, market, buildings.technologies, sector="Residential"
        )
        assert (DEFAULT_OUTPUT_DIRECTORY / "Residential2010Streetcred.csv").exists()
        assert (DEFAULT_OUTPUT_DIRECTORY / "Residential2010Streetcred.csv").is_file()
    finally:
        chdir(cwd)


@mark.sgidata
def test_from_sector_with_directory(buildings, market, tmpdir):
    from muse.outputs import factory

    output_func = factory({"quantity": "streetcred", "filename": tmpdir / "abc.csv"})
    output_func(
        buildings.capacity, market, buildings.technologies, sector="Residential"
    )
    assert (Path(tmpdir) / "abc.csv").exists()
    assert (Path(tmpdir) / "abc.csv").is_file()
