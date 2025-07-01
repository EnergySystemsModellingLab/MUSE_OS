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
    """Create a test output quantity that returns random data."""
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
def market():
    """Common market fixture used in multiple tests."""
    return xr.DataArray([1], coords={"year": [2010]}, dims="year")


@fixture
def base_config(tmp_path):
    """Common config fixture used in multiple tests."""
    path = tmp_path / "results" / "stuff"
    return {
        "filename": path / "{Sector}{year}{Quantity}.csv",
        "quantity": "streetcred",
    }


def create_test_data_array(values, coords=None, name="test"):
    """Helper function to create test DataArrays."""
    if coords is None:
        coords = dict(a=[2, 4])
    return xr.DataArray(values, coords=coords, dims="a", name=name)


def assert_file_exists_and_readable(path, expected_columns=None):
    """Helper to verify file exists and can be read."""
    assert path.exists() and path.is_file()
    df = pd.read_csv(path)
    if expected_columns:
        assert set(df.columns) == set(expected_columns)
    return df


@mark.usefixtures("streetcred")
def test_save_with_dir(tmp_path, market, base_config):
    """Test saving output to directory with sector and year in filename."""
    result = factory(base_config, sector_name="Yoyo")(market, None, None)
    assert len(result) == 1
    expected_path = Path(base_config["filename"]).parent / "Yoyo2010Streetcred.csv"
    assert result[0] == expected_path
    assert_file_exists_and_readable(result[0])


@mark.usefixtures("streetcred")
def test_overwrite(tmp_path, market, base_config):
    """Test file overwrite behavior."""
    outputter = factory(base_config, sector_name="Yoyo")
    result = outputter(market, None, None)
    expected_path = Path(base_config["filename"]).parent / "Yoyo2010Streetcred.csv"
    assert result[0] == expected_path
    assert_file_exists_and_readable(result[0])

    # default is to never overwrite
    with raises(IOError):
        outputter(market, None, None)

    base_config["overwrite"] = True
    factory(base_config, sector_name="Yoyo")(market, None, None)


@mark.usefixtures("streetcred")
@mark.parametrize(
    "config_type,suffix",
    [
        ("suffix", "nc"),
        ("sink", "nc"),
    ],
)
def test_save_with_path_to_nc(tmp_path, market, base_config, config_type, suffix):
    """Test saving output to NC file with different config types."""
    path = tmp_path / "results" / "stuff"
    if config_type == "suffix":
        config = {
            "filename": path / "{Sector}{year}{Quantity}{suffix}",
            "quantity": "streetcred",
            "suffix": suffix,
        }
    else:
        config = {
            "filename": path / "{sector}{year}{quantity}.csv",
            "quantity": "streetcred",
            "sink": suffix,
        }
    result = factory(config, sector_name="Yoyo")(market, None, None)
    expected_path = path / (
        f"{'Yoyo' if config_type == 'suffix' else 'yoyo'}"
        f"2010"
        f"{'Streetcred' if config_type == 'suffix' else 'streetcred'}"
        f".{'nc' if config_type == 'suffix' else 'csv'}"
    )
    assert result[0] == expected_path
    assert result[0].is_file()
    xr.open_dataset(result[0])


@mark.usefixtures("streetcred")
def test_save_with_fullpath_to_excel(tmp_path):
    from warnings import simplefilter

    from pandas import read_excel

    importorskip("openpyxl")
    simplefilter("default", PendingDeprecationWarning)

    path = tmp_path / "results" / "stuff" / "this.xlsx"
    config = {"filename": path, "quantity": "streetcred", "sink": "xlsx"}
    market = xr.DataArray([1], coords={"year": [2010]}, dims="year")
    # can use None because we **know** none of the arguments are used here
    result = factory(config, sector_name="Yoyo")(market, None, None)
    assert result[0] == path
    assert result[0].is_file()
    read_excel(result[0])


@patch("muse.outputs.cache.consolidate_quantity")
def test_output_functions(mock_consolidate):
    """Test output functions (capacity, production, lcoe) with common setup."""
    from muse.outputs.cache import capacity, lcoe, production

    cached = [xr.DataArray() for _ in range(3)]
    agents = {}

    for func, quantity in [
        (capacity, "capacity"),
        (production, "production"),
        (lcoe, "lcoe"),
    ]:
        func(cached, agents)
        mock_consolidate.assert_called_once_with(quantity, cached, agents)
        mock_consolidate.reset_mock()


@mark.usefixtures("streetcred")
def test_no_sink_or_suffix(tmp_path, market):
    """Test default sink and suffix behavior."""
    config = dict(
        quantity="streetcred",
        filename=f"{tmp_path}/{{Sector}}{{Quantity}}{{year}}{{suffix}}",
    )
    result = factory(config)(market, None, None)
    assert len(result) == 1
    assert result[0].is_file()
    assert result[0].suffix == ".csv"


@mark.usefixtures("save_registries")
def test_can_register_class():
    """Test class registration functionality."""
    from muse.outputs.sinks import factory, register_output_sink

    @register_output_sink
    class AClass:
        def __init__(self, sector, some_args=3):
            self.sector = sector
            self.some_args = some_args

        def __call__(self, x):
            pass

    # Test default arguments
    settings = {"sink": {"name": "AClass"}}
    sink = factory(settings, sector_name="yoyo")
    assert isinstance(sink, AClass)
    assert sink.sector == "yoyo"
    assert sink.some_args == 3

    # Test custom arguments
    settings = {"sink": {"name": "AClass", "some_args": 5}}
    sink = factory(settings, sector_name="yoyo")
    assert isinstance(sink, AClass)
    assert sink.sector == "yoyo"
    assert sink.some_args == 5


@mark.usefixtures("save_registries")
def test_can_register_function():
    """Test function registration functionality."""
    from muse.outputs.sinks import factory, register_output_sink

    @register_output_sink
    def a_function(*args):
        pass

    settings = {"sink": "a_function"}
    sink = factory(settings, sector_name="yoyo")
    assert sink.func is a_function


@mark.usefixtures("save_registries")
def test_yearly_aggregate():
    """Test yearly aggregation with custom sink."""
    from muse.outputs.sinks import factory, register_output_sink

    # Setup tracking variables
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

    # Test first year
    data = create_test_data_array([1, 0], name="nada")
    data["year"] = 2010
    assert isinstance(sink(data, 2010), MySpecialReturn)
    assert gyear == 2010
    assert gsector == "yoyo"
    assert goverwrite is True
    assert isinstance(received_data, pd.DataFrame)

    # Test second year
    data = create_test_data_array([0, 1], name="nada")
    data["year"] = 2020
    assert isinstance(sink(data, 2020), MySpecialReturn)
    assert gyear == 2020
    assert gsector == "yoyo"
    assert received_data[received_data.year == 2010].nada.values == approx([1, 0])
    assert received_data[received_data.year == 2020].nada.values == approx([0, 1])


def test_yearly_aggregate_file(tmp_path):
    """Test yearly aggregation to file with multiple years of data."""
    from muse.outputs.sinks import factory

    path = tmp_path / "file.csv"
    sink = factory(dict(filename=str(path), sink="aggregate"), sector_name="yoyo")

    def verify_year_data(values, year, expected_rows):
        data = create_test_data_array(values, name="georges")
        data["year"] = year
        assert sink(data, year) == path
        df = assert_file_exists_and_readable(path, {"year", "georges"})
        assert df.shape[0] == expected_rows
        return df

    # Test first year
    verify_year_data([1, 0], 2010, 2)

    # Test second year (should append to existing file)
    df2 = verify_year_data([0, 1], 2020, 4)

    # Verify data from both years is present
    assert set(df2.year.unique()) == {2010, 2020}
    assert df2[df2.year == 2010].georges.tolist() == [1, 0]
    assert df2[df2.year == 2020].georges.tolist() == [0, 1]


def test_yearly_aggregate_no_outputs(tmp_path):
    """Test behavior with no outputs configured."""
    from muse.outputs.mca import factory

    outputs = factory()
    assert len(outputs(None, year=2010)) == 0


def setup_mca_test(tmp_path, outputs_config):
    """Helper function to set up MCA tests."""
    from toml import dumps, load

    from muse import examples
    from muse.mca import MCA

    examples.copy_model(path=tmp_path)
    settings = load(tmp_path / "model" / "settings.toml")
    settings["outputs"] = [outputs_config]
    settings["time_framework"] = settings["time_framework"][:2]
    file = tmp_path / "model" / "settings.toml"
    file.write_text(dumps(settings), encoding="utf-8")
    return MCA.factory(file)


def test_mca_aggregate_outputs(tmp_path):
    """Test MCA aggregate outputs."""
    mca = setup_mca_test(
        tmp_path,
        dict(filename="{path}/{Quantity}{suffix}", quantity="prices", sink="aggregate"),
    )
    mca.run()
    assert (tmp_path / "model" / "Prices.csv").exists()
    # TODO: should pass again after #612
    # data = pd.read_csv(tmp_path / "model" / "Prices.csv")
    # assert set(data.year) == set(settings["time_framework"])


@mark.usefixtures("save_registries")
def test_path_formatting(tmp_path):
    """Test path formatting with dummy sink and quantity."""
    from muse.outputs.mca import register_output_quantity
    from muse.outputs.sinks import register_output_sink, sink_to_file

    @register_output_sink(name="dummy_sink")
    @sink_to_file(".dummy")
    def to_dummy(quantity, filename, **params) -> None:
        pass

    @register_output_quantity
    def dummy(market, **kwargs):
        return xr.DataArray()

    mca = setup_mca_test(
        tmp_path,
        dict(quantity="dummy", sink="to_dummy", filename="{path}/{Quantity}{suffix}"),
    )
    assert mca.outputs(mca.market)[0] == tmp_path / "model" / "Dummy.dummy"


def test_register_output_quantity_cache():
    from muse.outputs.cache import OUTPUT_QUANTITIES, register_cached_quantity

    @register_cached_quantity
    def dummy_quantity(*args):
        pass

    assert OUTPUT_QUANTITIES[dummy_quantity.__name__] == dummy_quantity


class TestOutputCache:
    @fixture
    def output_params(self):
        return [dict(quantity="height"), dict(quantity="width")]

    @fixture
    def output_quantities(self, output_params):
        quantities = {q["quantity"]: lambda _: None for q in output_params}
        quantities["depth"] = lambda _: None
        return quantities

    @fixture
    def topic(self):
        return "BBC Muse"

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_init(
        self, mock_factory, mock_subscribe, output_params, output_quantities, topic
    ):
        from muse.outputs.cache import OutputCache

        output_cache = OutputCache(
            *output_params, output_quantities=output_quantities, topic=topic
        )
        assert mock_factory.call_count == len(output_params)
        mock_subscribe.assert_called_once_with(output_cache.cache, topic)

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_cache(
        self, mock_factory, mock_subscribe, output_params, output_quantities, topic
    ):
        from muse.outputs.cache import OutputCache

        output_cache = OutputCache(
            *output_params, output_quantities=output_quantities, topic=topic
        )
        output_cache.cache(dict(height=xr.DataArray(), depth=xr.DataArray()))
        assert len(output_cache.to_save.get("height")) == 1
        assert len(output_cache.to_save.get("depth", [])) == 0

    @patch("pubsub.pub.subscribe")
    @patch("muse.outputs.sector._factory")
    def test_consolidate_cache(
        self, mock_factory, mock_subscribe, output_params, output_quantities, topic
    ):
        from muse.outputs.cache import OutputCache

        output_cache = OutputCache(
            *output_params, output_quantities=output_quantities, topic=topic
        )
        output_cache.cache(dict(height=xr.DataArray()))
        output_cache.consolidate_cache(2042)
        output_cache.factory["height"].assert_called_once()


@patch("pubsub.pub.sendMessage")
@patch("muse.outputs.cache.match_quantities")
def test_cache_quantity(mock_match, mock_send):
    from muse.outputs.cache import CACHE_TOPIC_CHANNEL, cache_quantity

    result = {"mass": 42}
    mock_match.return_value = result

    def verify_message_sent():
        mock_send.assert_called_once_with(CACHE_TOPIC_CHANNEL, data=result)
        mock_send.reset_mock()

    cache_quantity(**result)
    verify_message_sent()

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
    verify_message_sent()


def test_match_quantities():
    import xarray as xr

    from muse.outputs.cache import match_quantities

    def assert_equal(a: dict[str, xr.DataArray], b: dict[str, xr.DataArray]):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            xr.testing.assert_equal(a[k], b[k])

    # Test single quantity with DataArray
    q = "mass"
    da = xr.DataArray(name=q)
    ds = xr.Dataset({q: da})
    assert_equal(match_quantities(quantity=q, data=da), {q: da})
    assert_equal(match_quantities(quantity=q, data=ds), {q: da})

    # Test multiple quantities with Dataset
    p = "height"
    ds = xr.Dataset({q: da, p: da, "rubish": da})
    assert_equal(match_quantities(quantity=[q, p], data=ds), {q: da, p: da})
    assert_equal(match_quantities(quantity=[q, p], data=[da, da]), {q: da, p: da})

    # Test error cases
    with raises(ValueError):
        match_quantities(quantity=[q, p], data=[da])
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
    """Test internal agent extraction."""
    from types import SimpleNamespace

    from muse.outputs.cache import extract_agents_internal

    def setup_agent(agent, name):
        agent.name = name
        return agent

    agents = [setup_agent(newcapa_agent, "A1"), setup_agent(retro_agent, "A2")]
    sector = SimpleNamespace(name="IT", agents=agents)

    actual = extract_agents_internal(sector)
    expected_keys = ("agent", "category", "sector", "dst_region")

    for agent in agents:
        assert agent.uuid in actual
        assert tuple(actual[agent.uuid].keys()) == expected_keys
        agent_data = actual[agent.uuid]
        assert agent_data["agent"] == agent.name
        assert agent_data["category"] == agent.category
        assert agent_data["sector"] == "IT"
        assert agent_data["dst_region"] == agent.region


def test_aggregate_cache():
    import numpy as np
    from pandas.testing import assert_frame_equal

    from muse.outputs.cache import _aggregate_cache

    quantity = "height"
    a = xr.DataArray(np.ones((3, 4, 5)), name=quantity)
    b = a.copy()
    b[0, 0, 0] = 0

    def to_df(arr):
        return arr.to_dataframe().reset_index().astype(float)

    actual = _aggregate_cache(quantity, [a, b])
    assert_frame_equal(actual, to_df(b))

    actual = _aggregate_cache(quantity, [b, a])
    assert_frame_equal(actual, to_df(a))

    c = a.copy()
    c.assign_coords(dim_0=c.dim_0.data * 10)
    dc, da = map(to_df, [c, a])

    actual = _aggregate_cache(quantity, [c, a])
    expected = pd.DataFrame.merge(dc, da, how="outer").astype(float)
    assert_frame_equal(actual, expected)


def test_consolidate_quantity(newcapa_agent, retro_agent):
    """Test consolidation of quantity data with agent information."""
    from types import SimpleNamespace

    from muse.outputs.cache import consolidate_quantity, extract_agents_internal

    def setup_agent(agent, name, category):
        agent.name = name
        agent.category = category
        return agent

    newcapa_agent = setup_agent(newcapa_agent, "A1", "newcapa")
    retro_agent = setup_agent(retro_agent, "A2", "retro")
    sector = SimpleNamespace(name="IT", agents=[newcapa_agent, retro_agent])
    agents = extract_agents_internal(sector)

    def create_agent_array(agent_uuid, modify_first=False):
        arr = xr.DataArray(
            np.ones((3, 4, 5)),
            dims=("agent", "replacement", "asset"),
            coords={"agent": [agent_uuid] * 3},
            name="height",
        )
        if modify_first:
            arr[0, 0, 0] = 0
        return arr

    a = create_agent_array(newcapa_agent.uuid)
    b = create_agent_array(retro_agent.uuid, modify_first=True)

    actual = consolidate_quantity("height", [a, b], agents)
    cols = set((*agents[retro_agent.uuid].keys(), "technology", "height"))
    assert set(actual.columns) == cols
    assert all(
        name in (newcapa_agent.name, retro_agent.name) for name in actual.agent.unique()
    )
