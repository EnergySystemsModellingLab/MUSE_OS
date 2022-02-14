"""Test saving outputs to file."""
from pathlib import Path
from typing import Text
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx, fixture, importorskip, mark, raises

from muse.outputs.sector import factory


@fixture
def streetcred(save_registries):
    from muse.outputs.sector import register_output_quantity

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


@fixture
def limits_path(tmp_path):
    from textwrap import dedent

    path = tmp_path / "limits.csv"
    path.write_text(
        dedent(
            """
            Year,Month,Day,Hour,Region,Gas
            2020,all-year,all-week,night,R1,5
            2020,all-year,all-week,morning,R1,5
            2020,all-year,all-week,afternoon,R1,5
            2020,all-year,all-week,early-peak,R1,5
            2020,all-year,all-week,late-peak,R1,5
            2020,all-year,all-week,evening,R1,5
            2050,all-year,all-week,night,R1,8
            2050,all-year,all-week,morning,R1,8
            2050,all-year,all-week,afternoon,R1,8
            2050,all-year,all-week,early-peak,R1,8
            2050,all-year,all-week,late-peak,R1,8
            2050,all-year,all-week,evening,R1,8
            """
        )
    )
    return path


@mark.usefixtures("streetcred")
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


@mark.usefixtures("streetcred")
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


@mark.usefixtures("streetcred")
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


@mark.usefixtures("streetcred")
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


@mark.usefixtures("streetcred")
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


@mark.usefixtures("streetcred")
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


@mark.usefixtures("save_registries")
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


@mark.usefixtures("save_registries")
def test_can_register_function():
    from muse.outputs.sinks import register_output_sink, factory

    @register_output_sink
    def a_function(*args):
        pass

    settings = {"sink": "a_function"}
    sink = factory(settings, sector_name="yoyo")
    assert sink.func is a_function


@mark.usefixtures("save_registries")
def test_yearly_aggregate():
    from muse.outputs.sinks import register_output_sink, factory

    received_data = None
    gyear = None
    gsector = None
    goverwrite = None

    class MySpecialReturn:
        pass

    @register_output_sink(overwrite=True)
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


@mark.usefixtures("save_registries")
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
        dict(quantity="dummy", sink="to_dummy", filename="{path}/{Quantity}{suffix}")
    ]
    dump(settings, (settings_file))

    @register_output_sink(name="dummy_sink")
    @sink_to_file(".dummy")
    def to_dummy(quantity, filename, **params) -> None:
        pass

    @register_output_quantity
    def dummy(market, **kwargs):
        return xr.DataArray()

    mca = MCA.factory(Path(settings_file))
    assert mca.outputs(mca.market)[0] == Path(
        settings["outputs"][0]["filename"].format(
            path=tmpdir / "model", Quantity="Dummy", suffix=".dummy"
        )
    )


def test_aggregate_resources(market):
    from muse.outputs.mca import AggregateResources

    commodity = str(market.commodity.isel(commodity=0).values)
    output = AggregateResources(commodity)
    a = output(market, []).copy()
    assert (
        a == market.consumption.sel(year=2010, commodity=commodity, drop=True)
    ).all()
    b = output(market, []).copy()
    assert (b == 2 * a).all()


def test_finite_resources_quantity(limits_path):
    from muse import examples
    from muse.outputs.mca import FiniteResources

    market = examples.mca_market()[["consumption"]]

    output = FiniteResources(limits_path=limits_path, commodities="gas")
    result = output(market, [])
    assert set(result.dims) == {"region", "timeslice", "commodity"}
    assert result.all()

    market.consumption.loc[dict(commodity="gas")] = 3.0
    result = output(market, [])
    assert result.all()

    result = output(market, [])
    assert not result.all()


def test_finite_resources_in_sim(tmp_path, limits_path):
    from muse import examples
    from muse.readers.toml import read_settings
    from muse.mca import MCA
    from muse.outputs.sinks import FiniteResourceException
    from toml import load, dump

    examples.copy_model("default", path=tmp_path)
    toml = load(tmp_path / "model" / "settings.toml")
    toml["outputs"].append(
        dict(
            quantity="finite_resources",
            limits_path=str(limits_path.resolve()),
            early_exit=True,
            commodities="gas",
        )
    )
    with open(tmp_path / "model" / "settings.toml", "w") as fileobj:
        dump(toml, fileobj)

    mca = MCA.factory(read_settings(tmp_path / "model" / "settings.toml"))
    with raises(FiniteResourceException):
        mca.run()


def test_register_output_quantity_cache():
    from muse.outputs.cache import register_output_quantity, OUTPUT_QUANTITIES

    @register_output_quantity
    def dummy_quantity(*args):
        pass

    assert OUTPUT_QUANTITIES[dummy_quantity.__name__] == dummy_quantity


class TestOutputCache:
    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_init(self, mock_factory, mock_subscribe):
        from muse.outputs.cache import OutputCache

        param = [dict(quantity="Height"), dict(quantity="Width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["Depth"] = lambda _: None
        topic = "BBC Muse"

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        assert mock_factory.call_count == len(param)
        mock_subscribe.assert_called_once_with(output_cache.cache, topic)

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_cache(self, mock_factory, mock_subscribe):
        from muse.outputs.cache import OutputCache
        import xarray as xr

        param = [dict(quantity="Height"), dict(quantity="Width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["Depth"] = lambda _: None
        topic = "BBC Muse"

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        valid = xr.DataArray([], name=param[0]["quantity"])
        not_valid = xr.DataArray([], name="Depth")
        output_cache.cache(valid)
        output_cache.cache(not_valid)

        assert len(output_cache.to_save.get(valid.name)) == 1
        assert len(output_cache.to_save.get(not_valid.name, [])) == 0

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_consolidate_cache(self, mock_factory, mock_subscribe):
        from muse.outputs.cache import OutputCache
        import xarray as xr

        param = [dict(quantity="Height"), dict(quantity="Width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["Depth"] = lambda _: None
        topic = "BBC Muse"
        year = 2042

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        valid = xr.DataArray([], name=param[0]["quantity"])
        output_cache.cache(valid)
        output_cache.consolidate_cache(year)

        output_cache.factory[valid.name].assert_called_once()
