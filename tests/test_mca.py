from collections.abc import Sequence
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from pytest import approx
from xarray import DataArray, Dataset, broadcast

from muse.commodities import (
    CommodityUsage,
    is_consumable,
    is_enduse,
    is_other,
)
from muse.mca import check_demand_fulfillment, check_equilibrium, find_equilibrium
from muse.timeslices import drop_timeslice


def test_check_equilibrium(market: Dataset):
    """Test market equilibrium checking for both demand and prices."""
    years = [2010, 2020]
    tol = 0.1
    market = market.interp(year=years)
    new_market = market.copy(deep=True)

    # Test demand equilibrium
    assert check_equilibrium(new_market, market, tol, "demand")
    new_market["supply"] = drop_timeslice(new_market["supply"]) + tol * 1.5
    assert not check_equilibrium(new_market, market, tol, "demand")

    # Test price equilibrium
    assert check_equilibrium(new_market, market, tol, "prices")
    new_market["prices"] = drop_timeslice(new_market["prices"]) + tol * 1.5
    assert not check_equilibrium(new_market, market, tol, "prices")


def test_check_demand_fulfillment(market: Dataset):
    """Test if market demand is fulfilled within tolerance."""
    tolerance = -0.1
    market["supply"] = drop_timeslice(market.consumption.copy(deep=True))

    assert check_demand_fulfillment(market, tolerance)

    # Test with unfulfilled demand
    market["supply"] = drop_timeslice(market["supply"]) + tolerance * 1.5
    assert not check_demand_fulfillment(market, tolerance)


def sector_market(market: Dataset, comm_usage: Sequence[CommodityUsage]) -> Dataset:
    """Create a test market dataset with random supply/demand values.

    Args:
        market: Template market dataset with dimensions
        comm_usage: Sequence of commodity usage flags

    Returns:
        Dataset with supply, consumption and prices
    """
    shape = (
        len(market.year),
        len(market.commodity),
        len(market.region),
        len(market.timeslice),
    )

    values = np.random.randint(0, 5, shape) / np.random.randint(1, 5, shape)
    single = DataArray(
        values,
        dims=("year", "commodity", "region", "timeslice"),
        coords={
            "year": market.year,
            "commodity": market.commodity,
            "comm_usage": ("commodity", comm_usage),
            "region": market.region,
            "timeslice": market.timeslice,
        },
    )
    single = single.where(~is_other(single.comm_usage), 0)

    return Dataset(
        {
            "supply": single.where(is_enduse(single.comm_usage), 0),
            "consumption": single.where(is_consumable(single.comm_usage), 0),
            "prices": single,
        }
    )


def test_find_equilibrium(market: Dataset):
    """Test market equilibrium finding with mock sectors.

    Tests convergence behavior with different iteration limits and
    verifies market values match expected outcomes.
    """
    market = market.interp(year=[2010, 2015])

    # Setup test commodities
    a_enduses = np.random.choice(market.commodity.values, 5, replace=False).tolist()
    b_enduses = [a_enduses.pop(), a_enduses.pop()]

    # Define commodity usage patterns
    available = (
        CommodityUsage.CONSUMABLE,
        CommodityUsage.PRODUCT | CommodityUsage.ENVIRONMENTAL,
        CommodityUsage.OTHER,
    )

    def get_usage(enduses, commodities):
        return [
            CommodityUsage.PRODUCT if c in enduses else np.random.choice(available)
            for c in commodities
        ]

    a_usage = get_usage(a_enduses, market.commodity)
    b_usage = get_usage(b_enduses, market.commodity)

    # Create test markets
    a_market = sector_market(market, a_usage).rename(prices="costs")
    b_market = sector_market(market, b_usage).rename(prices="costs")

    # Initialize market values
    market["supply"][:] = 0
    market["consumption"][:] = 0
    market["prices"][:] = 1

    # Mock sector behavior
    def create_mock_sector(test_market, side_effect):
        sector = patch("muse.sectors.AbstractSector").start()()
        sector.next.side_effect = lambda *args, **kwargs: (
            test_market.sel(commodity=~is_other(test_market.comm_usage))
            * side_effect.pop(0)
        )
        return sector

    convergence_steps = [0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0]

    # Test with maxiter=2 (no convergence)
    a = create_mock_sector(a_market, convergence_steps.copy())
    b = create_mock_sector(b_market, convergence_steps.copy())

    result = find_equilibrium(market, deepcopy([a, b]), maxiter=2)
    assert not result.converged
    assert result.sectors[0].next.call_count == 1
    assert result.sectors[1].next.call_count == 1

    expected = a_market.supply + b_market.supply
    actual, expected = broadcast(result.market.supply, expected)
    assert actual.values == approx(0.7 * expected.values)

    # Test with maxiter=5 (partial convergence)
    a = create_mock_sector(a_market, convergence_steps.copy())
    b = create_mock_sector(b_market, convergence_steps.copy())

    result = find_equilibrium(market, deepcopy([a, b]), maxiter=5)
    assert not result.converged
    assert all(s.next.call_count == 1 for s in result.sectors)

    actual, expected = broadcast(
        result.market.supply, a_market.supply + b_market.supply
    )
    assert actual.values == approx(expected.values)

    # Test with maxiter=8 (full convergence)
    a = create_mock_sector(a_market, convergence_steps.copy())
    b = create_mock_sector(b_market, convergence_steps.copy())

    result = find_equilibrium(market, deepcopy([a, b]), maxiter=8)
    assert result.converged
    assert all(s.next.call_count == 1 for s in result.sectors)

    # Verify final market state
    actual, expected = broadcast(
        result.market.supply, a_market.supply + b_market.supply
    )
    assert actual.values == approx(expected.values)

    actual, expected = broadcast(
        result.market.consumption, a_market.consumption + b_market.consumption
    )
    assert actual.values == approx(expected.values)
