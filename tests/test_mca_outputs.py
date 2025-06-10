"""Tests for MCA output quantities."""

import pandas as pd
import xarray as xr
from pytest import fixture

from muse import examples
from muse.outputs.mca import (
    capacity,
    consumption,
    metric_capital_costs,
    metric_eac,
    metric_emission_costs,
    metric_fuel_costs,
    metric_lcoe,
    prices,
    supply,
)
from muse.utilities import broadcast_over_assets

YEAR = 2020


@fixture
def market() -> xr.Dataset:
    """Create a test market."""
    return examples.residential_market(model="default")


@fixture
def sectors(market) -> list:
    """Create test sectors using MUSE's examples module."""
    residential_sector = examples.sector("residential", model="default")
    agent = next(residential_sector.agents)
    technologies = residential_sector.technologies
    tech_data = broadcast_over_assets(technologies, agent.assets)

    # Make up supply data
    supply_data = xr.DataArray(
        data=1.0,
        dims=["timeslice", "commodity", "year", "asset"],
        coords={
            "timeslice": market.timeslice,
            "commodity": tech_data.commodity,
            "year": market.year,
            "asset": agent.assets.asset,
        },
    )
    agent.supply = supply_data

    # Make up consumption data
    consumption_data = xr.DataArray(
        data=1.0,
        dims=["timeslice", "commodity", "year", "asset"],
        coords={
            "timeslice": market.timeslice,
            "commodity": tech_data.commodity,
            "year": market.year,
            "asset": agent.assets.asset,
        },
    )
    agent.consumption = consumption_data

    return [residential_sector]


def test_consumption(market, sectors):
    """Test consumption output quantity."""
    result = consumption(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "region",
        "commodity",
        "year",
        "month",
        "day",
        "hour",
        "timeslice",
        "consumption",
    }
    assert set(result.columns) == expected_cols


def test_supply(market, sectors):
    """Test supply output quantity."""
    result = supply(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "region",
        "commodity",
        "year",
        "month",
        "day",
        "hour",
        "timeslice",
        "supply",
    }
    assert set(result.columns) == expected_cols


def test_prices(market, sectors):
    """Test prices output quantity."""
    result = prices(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "region",
        "commodity",
        "year",
        "month",
        "day",
        "hour",
        "timeslice",
        "prices",
    }
    assert set(result.columns) == expected_cols


def test_capacity(market, sectors):
    """Test capacity output quantity."""
    result = capacity(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "capacity",
        "region",
        "asset",
        "installed",
    }
    assert set(result.columns) == expected_cols


def test_fuel_costs(market, sectors):
    """Test fuel costs output quantity."""
    result = metric_fuel_costs(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "region",
        "month",
        "day",
        "hour",
        "timeslice",
        "asset",
        "fuel_consumption_costs",
    }
    assert set(result.columns) == expected_cols


def test_capital_costs(market, sectors):
    """Test capital costs output quantity."""
    result = metric_capital_costs(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "capital_costs",
        "region",
        "asset",
        "installed",
    }
    assert set(result.columns) == expected_cols


def test_emission_costs(market, sectors):
    """Test emission costs output quantity."""
    result = metric_emission_costs(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "region",
        "month",
        "day",
        "hour",
        "timeslice",
        "emission_costs",
        "asset",
    }
    assert set(result.columns) == expected_cols


def test_lcoe(market, sectors):
    """Test LCOE output quantity."""
    result = metric_lcoe(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "region",
        "tech_type",
        "fuel",
        "month",
        "day",
        "hour",
        "timeslice",
        "lcoe",
    }
    assert set(result.columns) == expected_cols


def test_eac(market, sectors):
    """Test EAC output quantity."""
    result = metric_eac(market, sectors, YEAR)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "region",
        "tech_type",
        "fuel",
        "month",
        "day",
        "hour",
        "timeslice",
        "eac",
    }
    assert set(result.columns) == expected_cols
