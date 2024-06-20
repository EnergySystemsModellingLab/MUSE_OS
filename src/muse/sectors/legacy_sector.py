"""This module defines the LegacySector class.

This is needed to interface the new MCA with the old MUSE sectors. It can be deleted
once accessing those sectors is no longer needed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from logging import getLogger
from typing import Any, Union

import numpy as np
import pandas as pd
from xarray import DataArray, Dataset

from muse.readers import read_csv_timeslices, read_initial_market
from muse.sectors.abstract import AbstractSector
from muse.sectors.register import register_sector
from muse.timeslices import QuantityType, new_to_old_timeslice


@dataclass
class LegacyMarket:
    BaseYear: int
    EndYear: int
    Foresight: np.ndarray
    TimeFramework: np.ndarray
    YearlyTimeFramework: np.ndarray
    NYears: list
    GlobalCommoditiesAttributes: np.ndarray
    CommoditiesBudget: list
    macro_drivers: pd.DataFrame
    dfRegions: pd.DataFrame
    Regions: np.ndarray
    interpolation_mode: str


@register_sector(name="legacy")
class LegacySector(AbstractSector):  # type: ignore
    @classmethod
    def factory(cls, name: str, settings: Any, **kwargs) -> "LegacySector":
        from pathlib import Path

        from muse_legacy.sectors import SECTORS

        from muse.readers import read_technologies

        sector = getattr(settings.sectors, name)

        settings_dir = sector.userdata_path
        sectors_dir = Path(sector.technodata_path).parent
        excess = sector.excess

        base_year = settings.time_framework[0]
        end_year = settings.time_framework[-1]

        path = settings.global_input_files.macrodrivers
        macro_drivers = pd.read_csv(path).sort_index(ascending=True)

        path = settings.global_input_files.regions
        regions = pd.read_csv(path).sort_index(ascending=True)
        global_commodities = read_technologies(
            Path(sector.technodata_path) / f"technodata{name.title()}.csv",
            None,
            Path(sector.technodata_path) / f"commOUTtechnodata{name.title()}.csv",
            Path(sector.technodata_path) / f"commINtechnodata{name.title()}.csv",
            commodities=settings.global_input_files.global_commodities,
        )[["heat_rate", "unit", "emmission_factor"]]

        interpolation_mode = (
            "Active" if settings.interpolation_mode == "linear" else "off"
        )

        market = LegacyMarket(
            BaseYear=base_year,
            EndYear=end_year,
            Foresight=np.array([settings.foresight]),
            TimeFramework=settings.time_framework,
            YearlyTimeFramework=np.arange(base_year, end_year + 1, 1, dtype=int),
            NYears=list(np.diff(settings.time_framework)),
            GlobalCommoditiesAttributes=global_commodities.commodity.values,
            CommoditiesBudget=settings.carbon_budget_control.commodities,
            macro_drivers=macro_drivers,
            dfRegions=regions,
            Regions=np.array(settings.regions),
            interpolation_mode=interpolation_mode,
        )

        timeslices, aggregation = cls.load_timeslices_and_aggregation(
            settings.timeslices, settings.sectors
        )
        timeslices = {
            "prices": timeslices["prices"],
            "finest": timeslices["finest"],
            "finest aggregation": aggregation,
            name: timeslices[name],
        }

        initial = (
            read_initial_market(
                settings.global_input_files.projections,
                base_year_export=getattr(
                    settings.global_input_files, "base_year_export", None
                ),
                base_year_import=getattr(
                    settings.global_input_files, "base_year_import", None
                ),
                timeslices=timeslices["prices"],
            )
            .sel(region=settings.regions)
            .interp(year=settings.time_framework, method=settings.interpolation_mode)
        )
        commodity_price = initial["prices"]
        static_trade = initial["static_trade"]

        old_sector = SECTORS[name](
            market=market, sectors_dir=sectors_dir, settings_dir=settings_dir
        )

        old_sector.SectorCommoditiesOUT = commodities_idx(old_sector, "OUT")
        old_sector.SectorCommoditiesIN = commodities_idx(old_sector, "IN")
        old_sector.SectorCommoditiesNotENV = commodities_idx(old_sector, "NotENV")

        sector_comm = list(
            set(old_sector.SectorCommoditiesOUT).union(old_sector.SectorCommoditiesIN)
        )

        commodities = {
            "global": global_commodities,
            name: global_commodities.isel(commodity=sector_comm),
        }

        msg = f"LegacySector {name} created successfully."
        getLogger(__name__).info(msg)
        return cls(
            name,
            old_sector,
            timeslices,
            commodities,
            commodity_price,
            static_trade,
            settings.regions,
            settings.time_framework,
            "Calibration" if getattr(settings, "calibration", False) else "Iteration",
            excess,
            "converged",
            str(sectors_dir),
            str(sector.output_path),
        )

    def __init__(
        self,
        name: str,
        old_sector,
        timeslices: dict,
        commodities: dict,
        commodity_price: DataArray,
        static_trade: DataArray,
        regions: Sequence,
        time_framework: np.ndarray,
        mode: str,
        excess: Union[int, float],
        market_iterative: str,
        sectors_dir: str,
        output_dir: str,
    ):
        super().__init__()
        self.name = name
        """Name of the sector"""
        self.old_sector = old_sector
        """Legacy sector method to run the calculation"""
        assert "prices" in timeslices
        assert "finest" in timeslices
        assert name in timeslices
        self.timeslices = timeslices
        """Timeslices for sectors and mca."""
        self.commodities = commodities
        """Commodities for each sector, as well as global commodities."""
        self.commodity_price = commodity_price
        """Initial price of all the commodities."""
        self.static_trade = static_trade
        """Static trade needed for the conversion and supply sectors."""
        self.regions = regions
        """Regions taking part in the simulation."""
        self.time_framework = time_framework
        """Time framework of the complete simulation."""
        self.mode = mode
        """If 'Calibration', the sector runs in calibration mode"""
        self.excess = excess
        """Allowed excess of capacity."""
        self.market_iterative = market_iterative
        """ -----> TODO what's this parameter?"""
        self.sectors_dir = sectors_dir
        """Sectors directory."""
        self.output_dir = output_dir
        """Outputs directory."""
        self.dims = ("commodity", "region", "year", "timeslice")
        """Order of the input and output dimensions."""
        self.calibrated = False
        """Flag if the sector has gone through the calibration process."""

    def next(self, market: Dataset) -> Dataset:
        """Adapter between the old and the new."""
        from muse_legacy.sectors.sector import Demand

        self.commodity_price.loc[{"year": market.year}] = market.prices

        # Consumption in Conversion and Supply sectors depend on the static trade
        # TODO This might need to go outside, in the MCA since it will affect all
        #  sectors, not just the legacy ones. But static trade seems to be always zero,
        #  so not sure how useful it might be.
        if not issubclass(type(self.old_sector), Demand):
            consumption = (
                market.consumption - self.static_trade.sel(year=market.year)
            ).clip(min=0.0)
        else:
            consumption = market.consumption.copy()

        converted = self.inputs(
            consumption=consumption, supply=market.supply, prices=self.commodity_price
        )

        idx = int(np.argwhere(self.time_framework == market.year.values[0]))

        result = self.runprocessmodule(
            converted.consumption,
            converted.supplycost,
            converted.supply,
            (idx, market.year.values[0]),
        )

        result = self.outputs(
            consumption=result.consumption,
            supply=result.supply,
            prices=result.supplycost,
        ).sel(year=market.year)

        result["comm_usage"] = self.commodities[self.name].comm_usage
        result = result.set_coords("comm_usage")

        # Prices in Demand sectors should not change.
        if issubclass(type(self.old_sector), Demand):
            result["prices"] = self.commodity_price.copy()

        return result

    def runprocessmodule(self, consumption, supplycost, supply, t):
        params = [
            consumption,
            supplycost,
            supply,
            new_to_old_timeslice(self.timeslices["prices"]),
            new_to_old_timeslice(
                self.timeslices["finest"], self.timeslices["finest aggregation"]
            ),
            t,
            self.mode,
        ]

        inputs = {"output_dir": self.output_dir, "sectors_dir": self.sectors_dir}

        if self.name == "Power":
            if self.mode == "Calibration":
                params += [self.market_iterative]
                result = self.old_sector.power_calibration(*params, **inputs)
                self.mode = "Iteration"
            else:
                self.mode = "Iteration"
                params += [self.old_sector.instance, self.market_iterative, self.excess]
                result = self.old_sector.runprocessmodule(*params, **inputs)
        else:
            params += [self.market_iterative, self.excess]
            result = self.old_sector.runprocessmodule(*params, **inputs)

        self.old_sector.report(result, t[1], self.output_dir)

        return result

    @staticmethod
    def load_timeslices_and_aggregation(timeslices, sectors) -> tuple[dict, str]:
        """Loads all sector timeslices and finds the finest one."""
        timeslices = {"prices": timeslices.rename("prices timeslices")}
        finest = timeslices["prices"].copy()
        aggregation = "month"

        for sector in sectors.list:
            sector_ts = read_csv_timeslices(
                getattr(sectors, sector).timeslices_path
            ).rename(sector + " timeslice")
            timeslices[sector] = sector_ts

            # Now we get the finest
            if len(finest) < len(sector_ts):
                finest = timeslices[sector]
                aggregation = getattr(sectors, sector).agregation_level
            elif len(finest) == len(sector_ts) and any(
                finest.get_index("timeslice") != sector_ts.get_index("timeslice")
            ):
                raise ValueError("Timeslice order do not match")

        timeslices["finest"] = finest
        timeslices["finest"] = timeslices["finest"].rename("finest timeslice")

        return timeslices, aggregation

    @property
    def global_commodities(self):
        """List of all commodities used by the MCA."""
        return self.commodities["global"].commodity.values

    @property
    def sector_commodities(self):
        """List of all commodities used by the Sector."""
        return self.commodities[self.name].commodity.values

    @property
    def sector_timeslices(self):
        """List of all commodities used by the MCA."""
        return self.timeslices[self.name]

    def _to(self, data: np.ndarray, data_ts, ts: pd.MultiIndex, qt: QuantityType):
        """From ndarray to dataarray."""
        return ndarray_to_xarray(
            years=self.time_framework,
            data=data,
            ts=ts,
            qt=qt,
            global_commodities=self.global_commodities,
            sector_commodities=self.sector_commodities,
            data_ts=data_ts,
            dims=self.dims,
            regions=self.regions,
        )

    def _from(self, xdata: DataArray, ts: pd.MultiIndex, qt: QuantityType):
        """From dataarray to ndarray."""
        return xarray_to_ndarray(
            years=self.time_framework,
            xdata=xdata,
            ts=ts,
            qt=qt,
            global_commodities=self.global_commodities,
            dims=self.dims,
            regions=self.regions,
        )

    def outputs(
        self, consumption: np.ndarray, prices: np.ndarray, supply: np.ndarray
    ) -> Dataset:
        """Converts MUSE numpy outputs to xarray."""
        from muse.timeslices import QuantityType

        finest, prices_ts = self.timeslices["finest"], self.timeslices["prices"]
        c = self._to(consumption, finest, prices_ts, QuantityType.EXTENSIVE)
        s = self._to(supply, self.sector_timeslices, prices_ts, QuantityType.EXTENSIVE)
        p = self._to(prices, self.sector_timeslices, prices_ts, QuantityType.INTENSIVE)
        return Dataset({"consumption": c, "supply": s, "costs": p})

    def inputs(self, consumption: DataArray, prices: DataArray, supply: DataArray):
        """Converts xarray to MUSE numpy input arrays."""
        from muse_legacy.sectors.sector import Sector as OriginalSector

        MarketVars = OriginalSector.MarketVars

        finest, prices_ts = self.timeslices["finest"], self.timeslices["prices"]
        c = self._from(consumption, finest, QuantityType.EXTENSIVE)
        s = self._from(supply, finest, QuantityType.EXTENSIVE)
        p = self._from(prices, prices_ts, QuantityType.INTENSIVE)

        return MarketVars(consumption=c, supply=s, supplycost=p)


def ndarray_to_xarray(
    years: np.ndarray,
    data: np.ndarray,
    ts: pd.MultiIndex,
    qt: QuantityType,
    global_commodities: DataArray,
    sector_commodities: DataArray,
    data_ts: pd.MultiIndex,
    dims: Sequence[str],
    regions: Sequence[str],
) -> DataArray:
    """From ndarray to dataarray."""
    from collections.abc import Hashable, Mapping

    from muse.timeslices import convert_timeslice

    coords: Mapping[Hashable, Any] = {
        "year": years,
        "commodity": global_commodities,
        "region": regions,
        "timeslice": data_ts,
    }
    result = convert_timeslice(DataArray(data, coords=coords, dims=dims), ts, qt)
    assert isinstance(result, DataArray)
    return result.sel(commodity=sector_commodities).transpose(*dims)


def xarray_to_ndarray(
    years: np.ndarray,
    xdata: DataArray,
    ts: pd.MultiIndex,
    qt: QuantityType,
    global_commodities: DataArray,
    dims: Sequence[str],
    regions: Sequence[str],
) -> np.ndarray:
    """From dataarray to ndarray."""
    from collections.abc import Hashable, Mapping

    from muse.timeslices import convert_timeslice

    coords: Mapping[Hashable, Any] = {
        "year": years,
        "commodity": global_commodities,
        "region": regions,
        "timeslice": ts,
    }
    warp = np.zeros((len(global_commodities), len(regions), len(years), len(ts)))
    result = DataArray(warp, coords=coords, dims=dims)
    result.loc[{"year": xdata.year}] = convert_timeslice(xdata, ts, qt).transpose(*dims)

    return result.values


def commodities_idx(sector, comm: str) -> Sequence:
    """Gets the indices of the commodities involved in the processes of the sector.

    Arguments:
        sector: The old MUSE sector of interest
        comm: Either "OUT", "IN" or "NotENV"

    Returns:
        A list with the indexes
    """
    comm = {
        "OUT": "listIndexCommoditiesOUT",
        "IN": "listIndexCommoditiesIN",
        "NotENV": "listIndexNotEnvironmental",
    }[comm]

    comm_list = chain.from_iterable(
        chain.from_iterable(
            [[c for c in p.__dict__[comm]] for p in wp.processes + wp.OtherProcesses]
            for wp in sector
        )
    )

    return list({item for item in comm_list})
