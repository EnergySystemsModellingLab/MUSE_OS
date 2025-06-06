"""Tests for MCA output quantities."""

import pandas as pd
from pytest import fixture

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


@fixture
def mock_sectors(model) -> list:
    """Create test sectors using MUSE's examples module."""
    from muse import examples

    return [examples.sector("residential", model=model)]


def test_consumption(market, mock_sectors):
    """Test consumption output quantity."""
    result = consumption(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Market quantities include timeslice-related dimensions
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


def test_supply(market, mock_sectors):
    """Test supply output quantity."""
    result = supply(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Market quantities include timeslice-related dimensions
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


def test_prices(market, mock_sectors):
    """Test prices output quantity."""
    result = prices(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Market quantities include timeslice-related dimensions
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


def test_capacity(market, mock_sectors):
    """Test capacity output quantity."""
    result = capacity(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "capacity",
    }
    assert set(result.columns) == expected_cols


def test_fuel_costs(market, mock_sectors):
    """Test fuel costs output quantity."""
    result = metric_fuel_costs(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Cost quantities include timeslice-related dimensions and region
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
        "fuel_consumption_costs",
    }
    assert set(result.columns) == expected_cols


def test_capital_costs(market, mock_sectors):
    """Test capital costs output quantity."""
    result = metric_capital_costs(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Capital costs include technology attributes
    expected_cols = {
        "technology",
        "agent",
        "category",
        "sector",
        "year",
        "tech_type",
        "fuel",
        "capital_costs",
        "region",
    }
    assert set(result.columns) == expected_cols


def test_emission_costs(market, mock_sectors):
    """Test emission costs output quantity."""
    result = metric_emission_costs(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Cost quantities include timeslice-related dimensions and region
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
    }
    assert set(result.columns) == expected_cols


def test_lcoe(market, mock_sectors):
    """Test LCOE output quantity."""
    result = metric_lcoe(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Cost quantities include timeslice-related dimensions and technology attributes
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


def test_eac(market, mock_sectors):
    """Test EAC output quantity."""
    result = metric_eac(market, mock_sectors, 2010)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Cost quantities include timeslice-related dimensions and technology attributes
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
