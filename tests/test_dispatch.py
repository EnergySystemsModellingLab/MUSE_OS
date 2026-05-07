import numpy as np
import xarray as xr
from pytest import fixture, raises

from muse.commodities import is_enduse
from muse.timeslices import broadcast_timeslice, distribute_timeslice
from muse.utilities import broadcast_over_assets


@fixture
def capacity(capacity: xr.DataArray) -> xr.DataArray:
    return capacity.isel(year=0)


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


@fixture
def prices(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice: xr.DataArray
) -> xr.DataArray:
    # Make random prices for all commodities/timeslices/regions
    regions = xr.DataArray(
        capacity["region"].to_index().unique(),
        dims="region",
        name="region",
    )
    prices_by_region = xr.DataArray(
        np.random.rand(
            technologies.sizes["commodity"],
            timeslice.sizes["timeslice"],
            regions.size,
        ),
        coords={
            "commodity": technologies.coords["commodity"],
            "timeslice": timeslice.coords["timeslice"],
            "region": regions,
        },
        dims=("commodity", "timeslice", "region"),
    )
    return prices_by_region


def test_share_based_supply_single_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    from muse.dispatch import share_based_production

    # Setup single region data
    region = "USA"
    technologies = technologies.where(technologies.region == region, drop=True)
    capacity = capacity.where(capacity.region == region, drop=True)
    production = production.where(production.region == region, drop=True)

    # Create random demand
    demand = production.sum("asset") * np.random.rand(*production.sum("asset").shape)
    assert "region" not in demand.dims

    # Test supply matches demand for end-use commodities
    spl = share_based_production(demand, capacity, technologies).sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_merit_order_supply_single_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    prices: xr.DataArray,
    timeslice,
):
    from muse.dispatch import merit_order_production

    # Setup single region data
    region = "USA"
    technologies = technologies.where(technologies.region == region, drop=True)
    capacity = capacity.where(capacity.region == region, drop=True)
    production = production.where(production.region == region, drop=True)
    prices = prices.sel(region=region)

    # Create random demand
    demand = production.sum("asset") * np.random.rand(*production.sum("asset").shape)
    assert "region" not in demand.dims

    # Test supply matches demand for end-use commodities
    spl = merit_order_production(demand, capacity, technologies, prices=prices).sum(
        "asset"
    )
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_share_based_supply_multi_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    from muse.dispatch import share_based_production

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)
    assert "region" in demand.dims

    # Test supply matches demand for end-use commodities by region
    spl = (
        share_based_production(demand, capacity, technologies)
        .groupby("region")
        .sum("asset")
    )
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_merit_order_supply_multi_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    prices: xr.DataArray,
    timeslice,
):
    from muse.dispatch import merit_order_production

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)
    assert "region" in demand.dims

    # Test supply matches demand for end-use commodities by region
    spl = (
        merit_order_production(demand, capacity, technologies, prices=prices)
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
    from muse.dispatch import share_based_production
    from muse.quantities import minimum_production

    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Test supply meets minimum production constraint
    spl = share_based_production(demand, capacity, technologies)
    assert (spl >= minprod).all()


def test_merit_order_supply_with_min_service(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    prices: xr.DataArray,
    timeslice,
):
    from muse.dispatch import merit_order_production
    from muse.quantities import minimum_production

    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Test supply meets minimum production constraint
    spl = merit_order_production(demand, capacity, technologies, prices=prices)
    assert (spl >= minprod).all()


def test_dispatch_by_merit_order():
    """Follows the example in the docstring of `dispatch_by_merit_order`."""
    from muse.dispatch import dispatch_by_merit_order

    demand = xr.DataArray(
        [140], dims=("commodity",), coords={"commodity": ["electricity"]}
    )
    minprod = xr.DataArray(
        [[20], [10], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    maxprod = xr.DataArray(
        [[50], [100], [30]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    technology_costs = xr.DataArray(
        [10, 5, 15], dims=("asset",), coords={"asset": ["A", "B", "C"]}
    )

    result = dispatch_by_merit_order(demand, minprod, maxprod, technology_costs)

    expected = xr.DataArray(
        [[40], [100], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    xr.testing.assert_equal(result, expected)


def test_dispatch_by_merit_order_with_unlabelled_assets():
    """Dispatch remains correct when the asset dimension has no labels."""
    from muse.dispatch import dispatch_by_merit_order

    demand = xr.DataArray(
        [140], dims=("commodity",), coords={"commodity": ["electricity"]}
    )
    minprod = xr.DataArray(
        [[20], [10], [0]],
        dims=("asset", "commodity"),
        coords={"commodity": ["electricity"]},
    )
    maxprod = xr.DataArray(
        [[50], [100], [30]],
        dims=("asset", "commodity"),
        coords={"commodity": ["electricity"]},
    )
    technology_costs = xr.DataArray([10, 5, 15], dims=("asset",))

    result = dispatch_by_merit_order(demand, minprod, maxprod, technology_costs)

    expected = xr.DataArray(
        [[40], [100], [0]],
        dims=("asset", "commodity"),
        coords={"commodity": ["electricity"]},
    )
    xr.testing.assert_equal(result, expected)


def test_dispatch_by_merit_order_rejects_reordered_costs():
    """Reordered labelled cost inputs are rejected by exact alignment."""
    from muse.dispatch import dispatch_by_merit_order

    demand = xr.DataArray(
        [140], dims=("commodity",), coords={"commodity": ["electricity"]}
    )
    minprod = xr.DataArray(
        [[20], [10], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    maxprod = xr.DataArray(
        [[50], [100], [30]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    technology_costs = xr.DataArray(
        [5, 10, 15],
        dims=("asset",),
        coords={"asset": ["B", "A", "C"]},
    )

    with raises(ValueError, match="join='exact'"):
        dispatch_by_merit_order(demand, minprod, maxprod, technology_costs)


def test_dispatch_by_merit_order_preserves_input_order_for_ties():
    """Stable sorting keeps the original asset order when costs are tied."""
    from muse.dispatch import dispatch_by_merit_order

    demand = xr.DataArray(
        [45], dims=("commodity",), coords={"commodity": ["electricity"]}
    )
    minprod = xr.DataArray(
        [[0], [0], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    maxprod = xr.DataArray(
        [[30], [30], [30]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    technology_costs = xr.DataArray(
        [10, 10, 20], dims=("asset",), coords={"asset": ["A", "B", "C"]}
    )

    result = dispatch_by_merit_order(demand, minprod, maxprod, technology_costs)

    expected = xr.DataArray(
        [[30], [15], [0]],
        dims=("asset", "commodity"),
        coords={"asset": ["A", "B", "C"], "commodity": ["electricity"]},
    )
    xr.testing.assert_equal(result, expected)
