"""Test saving outputs to file."""

from pathlib import Path
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
    from warnings import simplefilter

    from pandas import read_excel

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
    from muse.outputs.sinks import factory, register_output_sink

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
    from muse.outputs.sinks import factory, register_output_sink

    @register_output_sink
    def a_function(*args):
        pass

    settings = {"sink": "a_function"}
    sink = factory(settings, sector_name="yoyo")
    assert sink.func is a_function


@mark.usefixtures("save_registries")
def test_yearly_aggregate():
    from muse.outputs.sinks import factory, register_output_sink

    received_data = None
    gyear = None
    gsector = None
    goverwrite = None

    class MySpecialReturn:
        pass

    @register_output_sink(overwrite=True)
    def dummy(data, year: int, sector: str, overwrite: bool) -> MySpecialReturn:
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
    assert set(dataframe.columns) == {"year", "georges"}
    assert dataframe.shape[0] == 2

    data = xr.DataArray([0, 1], coords=dict(a=[2, 4]), dims="a", name="georges")
    data["year"] = 2020
    assert sink(data, 2020) == path
    dataframe = pd.read_csv(path)
    assert set(dataframe.columns) == {"year", "georges"}
    assert dataframe.shape[0] == 4


def test_yearly_aggregate_no_outputs(tmpdir):
    from muse.outputs.mca import factory

    outputs = factory()
    assert len(outputs(None, year=2010)) == 0


def test_mca_aggregate_outputs(tmpdir):
    from toml import dump, load

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
    from toml import dump, load

    from muse.examples import copy_model
    from muse.mca import MCA
    from muse.outputs.mca import register_output_quantity
    from muse.outputs.sinks import register_output_sink, sink_to_file

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
    from toml import dump, load

    from muse import examples
    from muse.mca import MCA
    from muse.outputs.sinks import FiniteResourceException
    from muse.readers.toml import read_settings

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
    from muse.outputs.cache import OUTPUT_QUANTITIES, register_cached_quantity

    @register_cached_quantity
    def dummy_quantity(*args):
        pass

    assert OUTPUT_QUANTITIES[dummy_quantity.__name__] == dummy_quantity


class TestOutputCache:
    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_init(self, mock_factory, mock_subscribe):
        from muse.outputs.cache import OutputCache

        param = [dict(quantity="height"), dict(quantity="width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["depth"] = lambda _: None
        topic = "BBC Muse"

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        assert mock_factory.call_count == len(param)
        mock_subscribe.assert_called_once_with(output_cache.cache, topic)

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_cache(self, mock_factory, mock_subscribe):
        import xarray as xr

        from muse.outputs.cache import OutputCache

        param = [dict(quantity="height"), dict(quantity="width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["depth"] = lambda _: None
        topic = "BBC Muse"

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        output_cache.cache(dict(height=xr.DataArray(), depth=xr.DataArray()))

        assert len(output_cache.to_save.get("height")) == 1
        assert len(output_cache.to_save.get("depth", [])) == 0

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_consolidate_cache(self, mock_factory, mock_subscribe):
        import xarray as xr

        from muse.outputs.cache import OutputCache

        param = [dict(quantity="height"), dict(quantity="width")]
        output_quantities = {q["quantity"]: lambda _: None for q in param}
        output_quantities["depth"] = lambda _: None
        topic = "BBC Muse"
        year = 2042

        output_cache = OutputCache(
            *param, output_quantities=output_quantities, topic=topic
        )

        output_cache.cache(dict(height=xr.DataArray()))
        output_cache.consolidate_cache(year)

        output_cache.factory["height"].assert_called_once()


@patch("pubsub.pub.sendMessage")
@patch("muse.outputs.cache.match_quantities")
def test_cache_quantity(mock_match, mock_send):
    from muse.outputs.cache import CACHE_TOPIC_CHANNEL, cache_quantity

    result = {"mass": 42}
    mock_match.return_value = result

    cache_quantity(**result)
    mock_send.assert_called_once_with(CACHE_TOPIC_CHANNEL, data=result)
    mock_send.reset_mock()

    with raises(ValueError):
        cache_quantity(function=lambda: None, mass=42)

    with raises(ValueError):

        @cache_quantity
        def fun():
            pass

    @cache_quantity(quantity="mass")
    def fun2():
        return 42

    fun2()
    mock_match.assert_called_once_with("mass", 42)
    mock_send.assert_called_once_with(CACHE_TOPIC_CHANNEL, data=result)


def test_match_quantities():
    import xarray as xr

    from muse.outputs.cache import match_quantities

    q = "mass"
    da = xr.DataArray(name=q)
    ds = xr.Dataset({q: da})

    def assert_equal(a: dict[str, xr.DataArray], b: dict[str, xr.DataArray]):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            xr.testing.assert_equal(a[k], b[k])

    actual = match_quantities(quantity=q, data=da)
    assert_equal(actual, {q: da})

    actual = match_quantities(quantity=q, data=ds)
    assert_equal(actual, {q: da})

    p = "height"
    ds = xr.Dataset({q: da, p: da, "rubish": da})
    actual = match_quantities(quantity=[q, p], data=ds)
    assert_equal(actual, {q: da, p: da})

    actual = match_quantities(quantity=[q, p], data=[da, da])
    assert_equal(actual, {q: da, p: da})

    with raises(ValueError):
        match_quantities(
            quantity=[q, p],
            data=[
                da,
            ],
        )

    with raises(TypeError):
        match_quantities(quantity=[q, p], data=42)


@patch("muse.outputs.cache.extract_agents_internal")
def test_extract_agents(mock_extract):
    from muse.outputs.cache import extract_agents

    sectors = set((1, 2, 3))
    mock_extract.side_effect = [{s: None} for s in sectors]

    actual = extract_agents(sectors)

    assert set(actual.keys()) == sectors
    assert set(actual.values()) == set((None,))


def test_extract_agents_internal(newcapa_agent, retro_agent):
    from types import SimpleNamespace

    from muse.outputs.cache import extract_agents_internal

    newcapa_agent.name = "A1"
    retro_agent.name = "A2"
    sector = SimpleNamespace(name="IT", agents=[newcapa_agent, retro_agent])

    actual = extract_agents_internal(sector)
    for agent in [newcapa_agent, retro_agent]:
        assert agent.uuid in actual
        assert tuple(actual[agent.uuid].keys()) == (
            "agent",
            "category",
            "sector",
            "year",
            "installed",
        )
        assert actual[agent.uuid]["agent"] == agent.name
        assert actual[agent.uuid]["category"] == agent.category
        assert actual[agent.uuid]["sector"] == "IT"


def test_aggregate_cache():
    import numpy as np
    import xarray as xr
    from pandas.testing import assert_frame_equal

    from muse.outputs.cache import _aggregate_cache

    quantity = "height"

    a = xr.DataArray(np.ones((3, 4, 5)), name=quantity)
    b = a.copy()
    b[0, 0, 0] = 0

    actual = _aggregate_cache(quantity, [a, b])
    assert_frame_equal(actual, b.to_dataframe().reset_index().astype(float))

    actual = _aggregate_cache(quantity, [b, a])
    assert_frame_equal(actual, a.to_dataframe().reset_index().astype(float))

    c = a.copy()
    c.assign_coords(dim_0=c.dim_0.data * 10)
    dc, da = (da.to_dataframe().reset_index() for da in [c, a])

    actual = _aggregate_cache(quantity, [c, a])
    expected = pd.DataFrame.merge(dc, da, how="outer").astype(float)
    assert_frame_equal(actual, expected)


def test_consolidate_quantity(newcapa_agent, retro_agent):
    from types import SimpleNamespace

    from muse.outputs.cache import consolidate_quantity, extract_agents_internal

    newcapa_agent.name = "A1"
    retro_agent.name = "A2"
    newcapa_agent.category = "newcapa"
    retro_agent.category = "retro"
    sector = SimpleNamespace(name="IT", agents=[newcapa_agent, retro_agent])
    agents = extract_agents_internal(sector)

    quantity = "height"
    a = xr.DataArray(
        np.ones((3, 4, 5)),
        dims=("agent", "replacement", "asset"),
        coords={
            "agent": [
                newcapa_agent.uuid,
            ]
            * 3
        },
        name=quantity,
    )
    b = a.copy()
    b[0, 0, 0] = 0
    b.assign_coords(
        agent=[
            retro_agent.uuid,
        ]
        * 3
    )

    actual = consolidate_quantity(quantity, [a, b], agents)

    cols = set(
        (*agents[retro_agent.uuid].keys(), "installed", "year", "technology", quantity)
    )
    assert set(actual.columns) == cols
    assert all(actual.year == newcapa_agent.forecast_year)
    assert all(actual.installed == newcapa_agent.year)
    assert all(
        name in (newcapa_agent.name, retro_agent.name) for name in actual.agent.unique()
    )


@patch("muse.outputs.cache.consolidate_quantity")
def test_output_capacity(mock_consolidate):
    import xarray as xr

    from muse.outputs.cache import capacity

    cached = [xr.DataArray() for _ in range(3)]
    agents = {}

    capacity(cached, agents)
    mock_consolidate.assert_called_once_with("capacity", cached, agents)


@patch("muse.outputs.cache.consolidate_quantity")
def test_output_production(mock_consolidate):
    import xarray as xr

    from muse.outputs.cache import production

    cached = [xr.DataArray() for _ in range(3)]
    agents = {}

    production(cached, agents)
    mock_consolidate.assert_called_once_with("production", cached, agents)


@patch("muse.outputs.cache.consolidate_quantity")
def test_output_lcoe(mock_consolidate):
    import xarray as xr

    from muse.outputs.cache import lcoe

    cached = [xr.DataArray() for _ in range(3)]
    agents = {}

    lcoe(cached, agents)
    mock_consolidate.assert_called_once_with("lcoe", cached, agents)
