import numpy as np
import xarray as xr
from pytest import fixture

from muse.commodities import is_enduse
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import broadcast_over_assets


@fixture
def technologies(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
) -> xr.Dataset:
    return broadcast_over_assets(technologies, capacity)


@fixture
def production(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
) -> xr.DataArray:
    return (
        broadcast_timeslice(capacity)
        * distribute_timeslice(technologies.fixed_outputs)
        * broadcast_timeslice(technologies.utilization_factor)
    )


def test_share_based_supply_single_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    from muse.dispatch import share_based_supply

    # Setup single region data
    region = "USA"
    technologies = technologies.where(technologies.region == region, drop=True)
    capacity = capacity.where(capacity.region == region, drop=True)
    production = production.where(production.region == region, drop=True)

    # Create random demand
    demand = production.sum("asset") * np.random.rand(*production.sum("asset").shape)
    assert "region" not in demand.dims

    # Test supply matches demand for end-use commodities
    spl = share_based_supply(demand, capacity, technologies).sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_share_based_supply_multi_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    from muse.dispatch import share_based_supply

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)
    assert "region" in demand.dims

    # Test supply matches demand for end-use commodities by region
    spl = (
        share_based_supply(demand, capacity, technologies)
        .groupby("region")
        .sum("asset")
    )
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_share_based_supply_with_min_service(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    from muse.dispatch import share_based_supply
    from muse.quantities import minimum_production

    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Test supply meets minimum production constraint
    spl = share_based_supply(demand, capacity, technologies)
    assert (spl >= minprod).all()


def test_merit_order_dispatch():
    from muse.dispatch import dispatch_by_merit_order

    demand = xr.DataArray(
        [140], dims=("commodity",), coords={"commodity": ["electricity"]}
    )
    minprod = xr.DataArray(
        [[10], [20], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    maxprod = xr.DataArray(
        [[100], [50], [30]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    technology_costs = xr.DataArray(
        [5, 10, 15], dims=("asset",), coords={"asset": ["A", "B", "C"]}
    )

    result = dispatch_by_merit_order(demand, minprod, maxprod, technology_costs)

    expected = xr.DataArray(
        [[100], [40], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    xr.testing.assert_equal(result, expected)
