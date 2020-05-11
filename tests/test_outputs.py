"""Test saving outputs to file."""
from pathlib import Path

import numpy as np
import xarray as xr
from pytest import importorskip, mark

from muse.outputs.sector import factory, register_output_quantity


@register_output_quantity
def streetcred(*args, **kwargs):

    return xr.DataArray(
        np.random.randint(0, 5, (3, 2)),
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
    }
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    result = factory(config, sector_name="Yoyo")(None, market, None)
    assert len(result) == 1
    assert result[0] == path / "Yoyo2010Streetcred.csv"
    assert result[0].exists()
    assert result[0].is_file()
    read_csv(result[0])


def test_overwrite(tmpdir):
    from pytest import raises

    path = Path(tmpdir) / "results" / "stuff"
    config = {
        "filename": path / "{Sector}{year}{Quantity}.csv",
        "quantity": "streetcred",
    }
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    outputter = factory(config, sector_name="Yoyo")
    result = outputter(None, market, None)
    assert result[0] == path / "Yoyo2010Streetcred.csv"
    assert result[0].is_file()

    # default is to never overwrite
    with raises(IOError):
        outputter(None, market, None)

    config["overwrite"] = True
    factory(config, sector_name="Yoyo")(None, market, None)


def test_save_with_path_to_nc_with_suffix(tmpdir):
    path = Path(tmpdir) / "results" / "stuff"
    config = {
        "filename": path / "{Sector}{year}{Quantity}{suffix}",
        "quantity": "streetcred",
        "suffix": "nc",
    }
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    result = factory(config, sector_name="Yoyo")(None, market, None)
    assert result[0] == path / "Yoyo2010Streetcred.nc"
    assert result[0].is_file()
    xr.open_dataset(result[0])


def test_save_with_path_to_nc_with_sink(tmpdir):
    path = Path(tmpdir) / "results" / "stuff"
    # can use None because we **know** none of the arguments are used here
    config = {
        "filename": path / "{sector}{year}{quantity}.csv",
        "quantity": "streetcred",
        "sink": "nc",
    }
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    result = factory(config, sector_name="Yoyo")(None, market, None)
    assert result[0] == path / "yoyo2010streetcred.csv"
    assert result[0].is_file()
    xr.open_dataset(result[0])


def test_save_with_fullpath_to_excel_with_sink(tmpdir):
    from pandas import read_excel
    from warnings import simplefilter

    importorskip("openpyxl")
    simplefilter("default", PendingDeprecationWarning)

    path = Path(tmpdir) / "results" / "stuff" / "this.xlsx"
    config = {"filename": path, "quantity": "streetcred", "sink": "xlsx"}
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    result = factory(config, sector_name="Yoyo")(None, market, None)
    assert result[0] == path
    assert result[0].is_file()
    read_excel(result[0])


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
