"""Test saving outputs to file."""
from pathlib import Path
from typing import Text

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, importorskip, mark

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
    result = factory(config, sector_name="Yoyo")(market, None, None)
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
    result = outputter(market, None, None)
    assert result[0] == path / "Yoyo2010Streetcred.csv"
    assert result[0].is_file()

    # default is to never overwrite
    with raises(IOError):
        outputter(market, None, None)

    config["overwrite"] = True
    factory(config, sector_name="Yoyo")(market, None, None)


def test_save_with_path_to_nc_with_suffix(tmpdir):
    path = Path(tmpdir) / "results" / "stuff"
    config = {
        "filename": path / "{Sector}{year}{Quantity}{suffix}",
        "quantity": "streetcred",
        "suffix": "nc",
    }
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    result = factory(config, sector_name="Yoyo")(market, None, None)
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
    result = factory(config, sector_name="Yoyo")(market, None, None)
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
    result = factory(config, sector_name="Yoyo")(market, None, None)
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


def test_no_sink_or_suffix(tmpdir):
    from muse.outputs.sector import factory

    config = dict(
        quantity="streetcred",
        filename=f"{tmpdir}/{{Sector}}{{Quantity}}{{year}}{{suffix}}",
    )
    outputs = factory(config)
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    result = outputs(market, None, None)
    assert len(result) == 1
    assert result[0].is_file()
    assert result[0].suffix == ".csv"


def test_can_register_class():
    from muse.outputs.sinks import register_output_sink, factory

    @register_output_sink
    class AClass:
        def __init__(self, sector, some_args=3):
            self.sector = sector
            self.some_args = some_args

        def __call__(self, x):
            pass

    settings = {"sink": {"name": "AClass"}}
    sink = factory(settings, sector_name="yoyo")
    assert isinstance(sink, AClass)
    assert sink.sector == "yoyo"
    assert sink.some_args == 3

    settings = {"sink": {"name": "AClass", "some_args": 5}}
    sink = factory(settings, sector_name="yoyo")
    assert isinstance(sink, AClass)
    assert sink.sector == "yoyo"
    assert sink.some_args == 5


def test_can_register_function():
    from muse.outputs.sinks import register_output_sink, factory

    @register_output_sink
    def a_function(*args):
        pass

    settings = {"sink": "a_function"}
    sink = factory(settings, sector_name="yoyo")
    assert sink.func is a_function


def test_yearly_aggregate():
    from muse.outputs.sinks import register_output_sink, factory

    received_data = None
    gyear = None
    gsector = None
    goverwrite = None

    class MySpecialReturn:
        pass

    @register_output_sink(name="dummy")
    def dummy(data, year: int, sector: Text, overwrite: bool) -> MySpecialReturn:
        nonlocal received_data, gyear, gsector, goverwrite
        received_data = data
        gyear = year
        gsector = sector
        goverwrite = overwrite
        return MySpecialReturn()

    sink = factory(
        dict(overwrite=True, sink=dict(aggregate="dummy")), sector_name="yoyo"
    )

    data = xr.DataArray([1, 0], coords=dict(a=[2, 4]), dims="a", name="nada")
    data["year"] = 2010

    assert isinstance(sink(data, 2010), MySpecialReturn)
    assert gyear == 2010
    assert gsector == "yoyo"
    assert goverwrite is True
    assert isinstance(received_data, pd.DataFrame)

    data = xr.DataArray([0, 1], coords=dict(a=[2, 4]), dims="a", name="nada")
    data["year"] = 2020
    assert isinstance(sink(data, 2020), MySpecialReturn)
    assert gyear == 2020
    assert gsector == "yoyo"
    assert received_data[received_data.year == 2010].nada.values == approx([1, 0])
    assert received_data[received_data.year == 2020].nada.values == approx([0, 1])


def test_yearly_aggregate_file(tmpdir):
    from muse.outputs.sinks import factory

    path = Path(tmpdir) / "file.csv"
    sink = factory(dict(filename=str(path), sink="aggregate"), sector_name="yoyo")

    data = xr.DataArray([1, 0], coords=dict(a=[2, 4]), dims="a", name="georges")
    data["year"] = 2010
    assert sink(data, 2010) == path
    dataframe = pd.read_csv(path)
    assert set(dataframe.columns) == {"a", "year", "georges"}
    assert dataframe.shape[0] == 2

    data = xr.DataArray([0, 1], coords=dict(a=[2, 4]), dims="a", name="georges")
    data["year"] = 2020
    assert sink(data, 2020) == path
    dataframe = pd.read_csv(path)
    assert set(dataframe.columns) == {"a", "year", "georges"}
    assert dataframe.shape[0] == 4


def test_yearly_aggregate_no_outputs(tmpdir):
    from muse.outputs.mca import factory

    outputs = factory()
    assert len(outputs(None, year=2010)) == 0


def test_mca_aggregate_outputs(tmpdir):
    from toml import load, dump
    from muse import examples
    from muse.mca import MCA

    examples.copy_model(path=str(tmpdir))
    settings = load(str(tmpdir / "model" / "settings.toml"))
    settings["outputs"] = [
        dict(filename="{path}/{Quantity}{suffix}", quantity="prices", sink="aggregate")
    ]
    settings["time_framework"] = settings["time_framework"][:2]
    dump(settings, (tmpdir / "model" / "settings.toml"))

    mca = MCA.factory(str(tmpdir / "model" / "settings.toml"))
    mca.run()

    assert (tmpdir / "model" / "Prices.csv").exists()
    data = pd.read_csv(tmpdir / "model" / "Prices.csv")
    assert set(data.year) == set(settings["time_framework"])


def test_path_formatting(tmpdir):
    from muse.mca import MCA
    from muse.examples import copy_model
    from muse.outputs.sinks import register_output_sink, sink_to_file
    from muse.outputs.mca import register_output_quantity
    from toml import load, dump

    # Copy the data to tmpdir
    copy_model(path=tmpdir)

    settings_file = tmpdir / "model" / "settings.toml"
    settings = load(settings_file)
    settings["outputs"] = [
        {
            "quantity": "dummy_quantity",
            "sink": "dummy",
            "filename": "{path}/{Quantity}{suffix}",
        }
    ]
    dump(settings, (settings_file))

    @register_output_sink(name="dummy")
    @sink_to_file(".dummy")
    def to_dummy(quantity, filename, **params) -> None:
        pass

    @register_output_quantity
    def dummy_quantity(market, **kwargs):
        return xr.DataArray()

    mca = MCA.factory(Path(settings_file))
    assert str(mca.outputs(mca.market)[0]) == settings["outputs"][0]["filename"].format(
        path=tmpdir / "model", Quantity="Dummy_Quantity", suffix=".dummy"
    )
