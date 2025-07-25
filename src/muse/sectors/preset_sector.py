"""Sector with preset behaviours."""

from __future__ import annotations

from typing import Any

from xarray import DataArray, Dataset

from muse.sectors.register import AbstractSector, register_sector


@register_sector(name=("preset", "presets"))
class PresetSector(AbstractSector):  # type: ignore
    """Sector with outcomes fixed from the start."""

    @classmethod
    def factory(cls, name: str, settings: Any) -> PresetSector:
        """Constructs a PresetSectors from input data."""
        from muse.readers.toml import read_presets_sector

        presets = read_presets_sector(settings, name)
        sector_conf = getattr(settings.sectors, name)
        interpolation_mode = getattr(sector_conf, "interpolation_mode", "linear")
        return cls(presets, interpolation_mode=interpolation_mode, name=name)

    def __init__(
        self,
        presets: Dataset,
        interpolation_mode: str = "linear",
        name: str = "preset",
    ):
        super().__init__()

        self.presets: Dataset = presets
        """Market across time and space."""
        self.interpolation_mode: str = interpolation_mode
        """Interpolation method"""
        self.name = name
        """Name by which to identify a sector"""
        self.commodities: list[str] = []

    def next(self, mca_market: Dataset) -> Dataset:
        """Advance sector by one time period."""
        presets = self.presets.sel(region=mca_market.region)
        supply = self._interpolate(presets.supply, mca_market.year)
        consumption = self._interpolate(presets.consumption, mca_market.year)
        costs = self._interpolate(presets.costs, mca_market.year)

        result = Dataset({"supply": supply, "consumption": consumption, "costs": costs})
        assert isinstance(result, Dataset)
        return result

    def _interpolate(self, data: DataArray, years: DataArray) -> DataArray:
        return data.interp(year=years, method=self.interpolation_mode).ffill("year")
