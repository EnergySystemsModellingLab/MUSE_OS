from pytest import fixture, mark

pytestmark = mark.usefixtures("default_timeslice_globals")


@fixture
def commercial_path():
    from pathlib import Path

    import muse_legacy

    return (
        Path(muse_legacy.__file__).parent / "data" / "test" / "presets" / "commercial"
    )


@mark.legacy
def test_presets_fails_on_missing_data(commercial_path):
    from pytest import raises

    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    nested = convert(dict(sectors=dict(preset=dict())))
    with raises(IOError):
        PresetSector.factory("preset", nested)


@mark.legacy
def test_presets_from_single(commercial_path):
    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(consumption_path=str(commercial_path / "*Consumption.csv"))
    nested = convert(dict(sectors=dict(preset=settings)))
    presets = PresetSector.factory("preset", nested)
    assert (presets.presets.supply.values == 0).all()
    assert (presets.presets.costs.values == 0).all()


@mark.legacy
def test_presets_from_outputs(commercial_path):
    from xarray import Dataset

    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(
        consumption_path=str(commercial_path / "*Consumption.csv"),
        supply_path=str(commercial_path / "*[0-9]Supply.csv"),
        lcoe_path=str(commercial_path / "*LCOE.csv"),
    )
    nested = convert(dict(sectors=dict(preset=settings)))
    presets = PresetSector.factory("preset", nested)
    market = Dataset(coords=presets.presets.coords).sel(year=[2010, 2015])
    assert presets.next(market) == presets.presets.sel(year=market.year)
    assert "comm_usage" in presets.next(market).coords
    assert "supply" in presets.next(market)
    assert "consumption" in presets.next(market)
    assert "costs" in presets.next(market)


@mark.sgidata
@mark.legacy
def test_presets_from_regression(sectors_dir, commercial_path):
    from xarray import Dataset

    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(
        macrodrivers_path=sectors_dir.join("Macrodrivers.csv"),
        regression_path=sectors_dir.join("Residential", "regressionparameters.csv"),
    )
    nested = convert(dict(sectors=dict(residential=settings)))
    presets = PresetSector.factory("residential", nested)
    market = Dataset(coords=presets.presets.coords).sel(year=[2010, 2015])
    assert "forecast" not in presets.presets.dims
    assert presets.next(market) == presets.presets.sel(year=market.year)
    assert (presets.presets.supply.values == 0).all()
    assert (presets.presets.costs.values == 0).all()
    assert "consumption" in presets.next(market)


@mark.sgidata
@mark.legacy
def test_presets_from_regression_with_forecast(sectors_dir, commercial_path):
    from xarray import Dataset

    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(
        macrodrivers_path=sectors_dir.join("Macrodrivers.csv"),
        regression_path=sectors_dir.join("Residential", "regressionparameters.csv"),
        forecast=[0, 5, 10],
    )
    nested = convert(dict(sectors=dict(residential=settings)))
    presets = PresetSector.factory("residential", nested)
    market = Dataset(coords=presets.presets.coords).sel(year=[2010, 2015])
    assert "forecast" in presets.presets.dims
    assert (presets.presets.supply.values == 0).all()
    assert (presets.presets.costs.values == 0).all()
    assert "consumption" in presets.next(market)
    assert "forecast" not in presets.next(market).dims
    assert "year" in presets.next(market).dims
    assert all(presets.next(market).year == market.year)


@mark.legacy
def test_presets_from_projection(commercial_path):
    from xarray import Dataset

    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(costs_path=str(commercial_path / "prices.csv"))
    nested = convert(dict(sectors=dict(residential=settings)))
    presets = PresetSector.factory("residential", nested)
    market = Dataset(coords=presets.presets.coords).sel(year=[2010, 2015])
    assert presets.next(market) == presets.presets.sel(year=market.year)
    assert (presets.presets.supply.values == 0).all()
    assert (presets.presets.consumption.values == 0).all()
    assert "costs" in presets.next(market)


@mark.sgidata
@mark.legacy
def test_presets_from_demand(sectors_dir, commercial_path):
    from muse.readers.toml import convert
    from muse.sectors import PresetSector

    settings = dict(
        commodities_path=str(commercial_path / "commodities.csv"),
        demand_path=str(sectors_dir.join("Demand.csv")),
    )
    nested = convert(dict(sectors=dict(residential=settings)))
    presets = PresetSector.factory("residential", nested)
    assert (presets.presets.supply.values == 0).all()
    assert (presets.presets.costs.values == 0).all()
    assert "consumption" in presets.presets
