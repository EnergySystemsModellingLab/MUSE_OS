"""Sector with preset behaviours."""
from __future__ import annotations

from typing import Any, Text

from xarray import DataArray, Dataset

from muse.sectors.register import AbstractSector, register_sector


@register_sector(name=("preset", "presets"))
class PresetSector(AbstractSector):  # type: ignore
    """Sector with outcomes fixed from the start."""

    @classmethod
    def factory(cls, name: Text, settings: Any) -> PresetSector:
        """Constructs a PresetSectors from input data."""
        from typing import Sequence

        from xarray import DataArray, zeros_like

        from muse.commodities import CommodityUsage
        from muse.readers import (
            read_attribute_table,
            read_csv_outputs,
            read_macro_drivers,
            read_regression_parameters,
            read_timeslice_shares,
            read_timeslices,
        )
        from muse.regressions import endogenous_demand
        from muse.timeslices import QuantityType, convert_timeslice

        sector_conf = getattr(settings.sectors, name)
        presets = Dataset()

        presets["timeslice"] = read_timeslices(
            getattr(sector_conf, "timeslice_levels", None)
        ).timeslice
        if getattr(sector_conf, "consumption_path", None) is not None:
            consumption = read_csv_outputs(sector_conf.consumption_path)
            consumption.coords["timeslice"] = presets.timeslice
            presets["consumption"] = consumption
        elif getattr(sector_conf, "demand_path", None) is not None:
            presets["consumption"] = read_attribute_table(sector_conf.demand_path)
        elif (
            getattr(sector_conf, "macrodrivers_path", None) is not None
            and getattr(sector_conf, "regression_path", None) is not None
        ):
            macro_drivers = read_macro_drivers(
                getattr(sector_conf, "macrodrivers_path", None)
            )
            regression_parameters = read_regression_parameters(
                getattr(sector_conf, "regression_path", None)
            )
            forecast = getattr(sector_conf, "forecast", 0)
            if isinstance(forecast, Sequence):
                forecast = DataArray(
                    forecast, coords={"forecast": forecast}, dims="forecast"
                )
            consumption = endogenous_demand(
                drivers=macro_drivers,
                regression_parameters=regression_parameters,
                forecast=forecast,
            )
            if hasattr(sector_conf, "filters"):
                consumption = consumption.sel(sector_conf.filters._asdict())
            if "sector" in consumption.dims:
                consumption = consumption.sum("sector")

            if getattr(sector_conf, "timeslice_shares_path", None) is not None:
                timeslice = presets["timeslice"]
                assert isinstance(timeslice, DataArray)
                shares = read_timeslice_shares(
                    sector_conf.timeslice_shares_path, timeslice=timeslice
                )
                assert consumption.commodity.isin(shares.commodity).all()
                assert consumption.region.isin(shares.region).all()
                consumption = consumption * shares.sel(
                    region=consumption.region, commodity=consumption.commodity
                )
            presets["consumption"] = consumption

        if getattr(sector_conf, "supply_path", None) is not None:
            supply = read_csv_outputs(sector_conf.supply_path)
            supply.coords["timeslice"] = presets.timeslice
            presets["supply"] = supply

        if getattr(sector_conf, "costs_path", None) is not None:
            presets["costs"] = read_attribute_table(sector_conf.costs_path)
        elif (
            getattr(sector_conf, "lcoe_path", None) is not None and "supply" in presets
        ):
            costs = (
                read_csv_outputs(
                    sector_conf.lcoe_path,
                    indices=("RegionName", "ProcessName"),
                    columns="timeslices",
                )
                * presets["supply"]
            )
            presets["costs"] = costs

        if len(presets.data_vars) == 0:
            raise IOError("None of supply, consumption, costs given")

        # add missing data as zeros: we only need one of conumption, costs, supply
        components = {"supply", "consumption", "costs"}
        for component in components:
            others = components.intersection(presets.data_vars).difference({component})
            if component not in presets and len(others) > 0:
                presets[component] = zeros_like(presets[others.pop()])
        # add timeslice, if missing
        for component in {"supply", "consumption"}:
            if "timeslice" not in presets[component].dims:
                presets[component] = convert_timeslice(
                    presets[component], presets.timeslice, QuantityType.EXTENSIVE
                )

        comm_usage = (presets.costs > 0).any(set(presets.costs.dims) - {"commodity"})
        presets["comm_usage"] = (
            "commodity",
            [CommodityUsage.PRODUCT if u else CommodityUsage.OTHER for u in comm_usage],
        )
        presets = presets.set_coords("comm_usage")
        if "process" in presets.dims:
            presets = presets.sum("process")

        interpolation_mode = getattr(sector_conf, "interpolation_mode", "linear")
        return cls(presets, interpolation_mode=interpolation_mode, name=name)

    def __init__(
        self,
        presets: Dataset,
        interpolation_mode: Text = "linear",
        name: Text = "preset",
    ):
        super().__init__()

        self.presets: Dataset = presets
        """Market across time and space."""
        self.interpolation_mode: Text = interpolation_mode
        """Interpolation method"""
        self.name = name
        """Name by which to identify a sector"""

    def next(self, mca_market: Dataset) -> Dataset:
        """Advance sector by one time period."""
        from muse.timeslices import QuantityType, convert_timeslice

        presets = self.presets.sel(region=mca_market.region)
        supply = self._interpolate(presets.supply, mca_market.year)
        consumption = self._interpolate(presets.consumption, mca_market.year)
        costs = self._interpolate(presets.costs, mca_market.year)

        result = convert_timeslice(
            Dataset({"supply": supply, "consumption": consumption}),
            mca_market.timeslice,
            QuantityType.EXTENSIVE,
        )
        result["costs"] = convert_timeslice(
            costs, mca_market.timeslice, QuantityType.INTENSIVE
        )
        assert isinstance(result, Dataset)
        return result

    def _interpolate(self, data: DataArray, years: DataArray) -> DataArray:
        """Chooses interpolation depending on whether forecast is available."""

        if "forecast" in data.dims:
            baseyear = int(years.min())
            forecasted = (years - baseyear).values
            result = (
                data.interp(
                    year=baseyear,
                    method=self.interpolation_mode,
                    kwargs={"fill_value": "extrapolate"},
                )
                .interp(
                    forecast=forecasted,
                    method=self.interpolation_mode,
                    kwargs={"fill_value": "extrapolate"},
                )
                .drop_vars(("year", "forecast"))
            )
            result["year"] = "forecast", years.values
            return result.set_index(forecast="year").rename(forecast="year")
        return data.interp(year=years, method=self.interpolation_mode).ffill("year")
