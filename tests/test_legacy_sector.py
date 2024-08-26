from pathlib import Path
from typing import Optional

from pytest import approx, mark


def legacy_inputs():
    try:
        import muse_legacy
    except ImportError:
        return []

    from muse_legacy.sectors import SECTORS

    excluded = {
        "Bioenergy",
        "Commercial",
        "Industry",
        "NET",
        "Refinery",
        "Residential",
        "IndustryABM",
        "Sequestration",
        "TradeSupply",
        "TradeRefinery",
        "TradePower",
        "Transport",
        "Shipping",
        "Supply",
        "Power",
    }

    return [
        (
            sector,
            Path(muse_legacy.__file__).parent
            / "data"
            / "test"
            / "cases"
            / sector
            / f"settings_legacy_{sector.lower()}.toml",
        )
        for sector in set(SECTORS) - excluded
    ]


def legacy_input_file(sector: str) -> Optional[Path]:
    """Gets the legacy sector settings file."""
    input_file = (
        Path(__file__).parent
        / "data"
        / "cases"
        / sector
        / f"settings_legacy_{sector.lower()}.toml"
    )

    return input_file


def update_settings(settings, sec_dir, out_dir):
    """Updates a settings namedtuple with temporal sectors and output directories."""
    sectors = settings.sectors

    for s in sectors.list:
        path = Path(sec_dir) / s
        sector = getattr(sectors, s)._replace(
            userdata_path=path, technodata_path=path, output_path=out_dir
        )
        sectors = sectors._replace(**{s: sector})

    return settings._replace(sectors=sectors)


@mark.legacy
@mark.sgidata
@mark.parametrize("sector,filepath", legacy_inputs())
def test_legacy_sector_creation(sector, filepath):
    """Test the creation of the legacy sectors."""
    from muse.readers import read_settings
    from muse.sectors import SECTORS_REGISTERED

    settings = read_settings(filepath)

    SECTORS_REGISTERED["legacy"](name=sector, settings=settings)


def test_xarray_to_array(market):
    import numpy as np

    from muse.sectors.legacy_sector import xarray_to_ndarray
    from muse.timeslices import QuantityType

    dims = ("commodity", "region", "year", "timeslice")
    arr = xarray_to_ndarray(
        years=market.year,
        xdata=market.supply,
        ts=market.timeslice,
        qt=QuantityType.EXTENSIVE,
        global_commodities=market.commodity,
        dims=dims,
        regions=np.array(market.region),
    )

    assert arr == approx(market.supply.transpose(*dims).values)


def test_array_to_xarray(market):
    from numpy import array
    from xarray import broadcast

    from muse.sectors.legacy_sector import ndarray_to_xarray
    from muse.timeslices import QuantityType

    dims = ("commodity", "region", "year", "timeslice")
    arr = market.supply.transpose(*dims).values
    result = ndarray_to_xarray(
        years=market.year,
        data=arr,
        ts=market.timeslice,
        qt=QuantityType.EXTENSIVE,
        global_commodities=market.commodity,
        sector_commodities=market.commodity,
        data_ts=market.timeslice,
        dims=dims,
        regions=array(market.region),
    )

    expected, actual = broadcast(market.supply, result)
    assert actual.values == approx(expected.values)


def test_round_trip(market):
    from numpy import array
    from xarray import broadcast

    from muse.sectors.legacy_sector import ndarray_to_xarray, xarray_to_ndarray
    from muse.timeslices import QuantityType

    dims = ("commodity", "region", "year", "timeslice")

    arr = xarray_to_ndarray(
        years=market.year,
        xdata=market.supply,
        ts=market.timeslice,
        qt=QuantityType.EXTENSIVE,
        global_commodities=market.commodity,
        dims=dims,
        regions=array(market.region),
    )

    result = ndarray_to_xarray(
        years=market.year,
        data=arr,
        ts=market.timeslice,
        qt=QuantityType.EXTENSIVE,
        global_commodities=market.commodity,
        sector_commodities=market.commodity,
        data_ts=market.timeslice,
        dims=dims,
        regions=array(market.region),
    )

    expected, actual = broadcast(market.supply, result)
    assert actual.values == approx(expected.values)


@mark.legacy
@mark.sgidata
@mark.regression
@mark.parametrize("sector,filepath", legacy_inputs())
def test_legacy_sector_regression(sector, filepath, sectors_dir, tmpdir, compare_dirs):
    """Test the execution of the next method in the legacy sectors for 1 year."""
    from muse.mca import MCA
    from muse.readers import read_settings

    settings = read_settings(filepath)
    settings = update_settings(settings, sectors_dir, tmpdir)

    mca = MCA.factory(settings)
    mca.run()

    regression_dir = filepath.parent
    compare_dirs(tmpdir, regression_dir / "output")
