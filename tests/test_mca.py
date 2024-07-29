from collections.abc import Sequence

from muse.commodities import CommodityUsage
from muse.timeslices import drop_timeslice
from xarray import Dataset


def test_check_equilibrium(market: Dataset):
    """Test for the equilibrium function of the MCA."""
    from muse.mca import check_equilibrium

    years = [2010, 2020]
    tol = 0.1
    equilibrium_variable = "demand"

    market = market.interp(year=years)
    new_market = market.copy(deep=True)

    assert check_equilibrium(new_market, market, tol, equilibrium_variable)
    new_market["supply"] = drop_timeslice(new_market["supply"]) + tol * 1.5

    assert not check_equilibrium(new_market, market, tol, equilibrium_variable)

    equilibrium_variable = "prices"

    assert check_equilibrium(new_market, market, tol, equilibrium_variable)
    new_market["prices"] = drop_timeslice(new_market["prices"]) + tol * 1.5
    assert not check_equilibrium(new_market, market, tol, equilibrium_variable)


def test_check_demand_fulfillment(market):
    """Test for the demand fulfillment function of the MCA."""
    from muse.mca import check_demand_fulfillment

    tolerance_unmet_demand = -0.1

    market["supply"] = drop_timeslice(market.consumption.copy(deep=True))
    assert check_demand_fulfillment(
        market,
        tolerance_unmet_demand,
    )
    market["supply"] = drop_timeslice(market["supply"]) + tolerance_unmet_demand * 1.5
    assert not check_demand_fulfillment(
        market,
        tolerance_unmet_demand,
    )


def sector_market(market: Dataset, comm_usage: Sequence[CommodityUsage]) -> Dataset:
    """Creates a likely return market from a sector."""
    from muse.commodities import is_consumable, is_enduse, is_other
    from numpy.random import randint
    from xarray import DataArray

    shape = (
        len(market.year),
        len(market.commodity),
        len(market.region),
        len(market.timeslice),
    )
    single = DataArray(
        randint(0, 5, shape) / randint(1, 5, shape),
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
    from copy import deepcopy
    from unittest.mock import patch

    from muse.commodities import is_enduse, is_other
    from muse.mca import find_equilibrium
    from numpy.random import choice
    from pytest import approx
    from xarray import broadcast

    market = market.interp(year=[2010, 2015])
    a_enduses = choice(market.commodity.values, 5, replace=False).tolist()
    b_enduses = [a_enduses.pop(), a_enduses.pop()]

    # only "service" is currently truly meaningful and required here
    # "service" means non-environmental outputs.
    available = (
        CommodityUsage.CONSUMABLE,
        CommodityUsage.PRODUCT | CommodityUsage.ENVIRONMENTAL,
        CommodityUsage.OTHER,
    )
    a_usage = [
        CommodityUsage.PRODUCT if i in a_enduses else choice(available)
        for i in market.commodity
    ]
    b_usage = [
        CommodityUsage.PRODUCT if i in b_enduses else choice(available)
        for i in market.commodity
    ]

    a_market = sector_market(market, a_usage).rename(prices="costs")
    b_market = sector_market(market, b_usage).rename(prices="costs")

    market["supply"][:] = 0
    market["consumption"][:] = 0
    market["prices"][:] = 1

    cls = "muse.sectors.AbstractSector"
    with patch(cls) as SectorA, patch(cls) as SectorB:
        a = SectorA()

        side_effect_a = [0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0]
        a.next.side_effect = lambda *args, **kwargs: a_market.sel(
            commodity=~is_other(a_market.comm_usage)
        ) * side_effect_a.pop(0)

        b = SectorB()
        side_effect_b = [0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0]
        b.next.side_effect = lambda *args, **kwargs: b_market.sel(
            commodity=~is_other(b_market.comm_usage)
        ) * side_effect_b.pop(0)
        # maxiter equals 1 implicitly not convergence
        result = find_equilibrium(market, deepcopy([a, b]), maxiter=2)
        assert not result.converged
        assert result.sectors[0].next.call_count == 1
        assert result.sectors[1].next.call_count == 1
        expected = a_market.supply + b_market.supply
        actual, expected = broadcast(result.market.supply, expected)
        assert actual.values == approx(0.7 * expected.values)

        side_effect_a.clear()
        side_effect_a.extend([0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0])
        side_effect_b.clear()
        side_effect_b.extend([0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0])
        result = find_equilibrium(market, deepcopy([a, b]), maxiter=5)
        assert not result.converged
        # check statelessness ~ only one call to next
        assert result.sectors[0].next.call_count == 1
        assert result.sectors[1].next.call_count == 1
        expected = a_market.supply + b_market.supply
        actual, expected = broadcast(result.market.supply, expected)
        assert actual.values == approx(expected.values)

        side_effect_a.clear()
        side_effect_a.extend([0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0])
        side_effect_b.clear()
        side_effect_b.extend([0.5, 0.7, 0.9, 0.95, 1.0, 1.0, 1.0])
        sectors = deepcopy([a, b])
        result = find_equilibrium(market, sectors, maxiter=8)
        assert result.converged
        # check statelessness ~ only one call to next
        assert result.sectors[0].next.call_count == 1
        assert result.sectors[1].next.call_count == 1

        expected = a_market.supply + b_market.supply
        actual, expected = broadcast(result.market.supply, expected)
        assert actual.values == approx(expected.values)

        expected = a_market.consumption + b_market.consumption
        actual, expected = broadcast(result.market.consumption, expected)
        assert actual.values == approx(expected.values)

        expected = b_market.costs.where(is_enduse(b_market.comm_usage))
        expected = a_market.costs.where(is_enduse(a_market.comm_usage))
        expected = expected.where(expected > 1e-15, result.market.prices)
        actual, expected = broadcast(result.market.prices, expected)
        assert (actual.sel(year=2015)).values == approx(expected.sel(year=2015).values)
