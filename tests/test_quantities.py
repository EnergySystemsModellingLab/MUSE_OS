import numpy as np
import xarray as xr
from pytest import approx, fixture, mark

from muse.commodities import is_enduse, is_pollutant
from muse.quantities import (
    capacity_in_use,
    consumption,
    emission,
    maximum_production,
    minimum_production,
    production_amplitude,
    supply,
)
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


def test_consumption(technologies: xr.Dataset, production: xr.DataArray, market):
    # Test without prices
    consump = consumption(technologies, production)
    assert set(production.dims) == set(consump.dims)

    # Test with prices but no flexible inputs
    technologies.flexible_inputs[:] = 0
    consump2 = consumption(technologies, production, market.prices)
    assert consump2.values == approx(consump.values)

    # Test with prices and flexible inputs
    consump3 = consumption(technologies, production, market.prices)
    assert set(production.dims) == set(consump3.dims)


def test_production_aggregate_asset_view(
    technologies: xr.Dataset, capacity: xr.DataArray
):
    """Test production when capacity has format of agent.sector."""
    technologies = technologies[["fixed_outputs", "utilization_factor"]]
    enduses = is_enduse(technologies.comm_usage)
    assert enduses.any()

    def check_production(fouts: float, ufact: float):
        technologies.fixed_outputs[:] = fouts
        technologies.utilization_factor[:] = ufact
        prod = maximum_production(technologies, capacity)

        assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
        assert prod.sel(commodity=~enduses).values == approx(0)

        prod, expected = xr.broadcast(
            prod.sel(commodity=enduses).sum("timeslice"), capacity
        )
        assert prod.values == approx(fouts * ufact * expected.values)

    # Test different combinations of fixed outputs and utilization factors
    check_production(1.0, 1.0)
    check_production(2.0, 0.5)
    check_production(3.0, 0.5)


@mark.xfail
def test_production_agent_asset_view(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
):
    """Test production when capacity has format of agent.assets.capacity."""
    from muse.utilities import coords_to_multiindex, reduce_assets

    capacity = coords_to_multiindex(reduce_assets(capacity)).unstack("asset").fillna(0)
    test_production_aggregate_asset_view(technologies, capacity)


def test_capacity_in_use(production: xr.DataArray, technologies: xr.Dataset):
    technologies = technologies[["fixed_outputs", "utilization_factor"]]
    production[:] = prod = 10
    technologies.fixed_outputs[:] = fout = 5
    technologies.utilization_factor[:] = ufac = 2

    enduses = is_enduse(technologies.comm_usage)

    # Test with max_dim=None
    capa = capacity_in_use(production, technologies, max_dim=None)
    assert "commodity" in capa.dims
    capa, expected = xr.broadcast(capa, enduses * prod / fout / ufac)
    assert capa.values == approx(expected.values)

    # Test without max_dim
    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)

    # Test with modified production for specific commodity
    maxcomm = np.random.choice(production.commodity.sel(commodity=enduses).values)
    production.loc[{"commodity": maxcomm}] = prod = 11
    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)


def test_emission(production: xr.DataArray, technologies: xr.Dataset):
    em = emission(production, technologies)
    envs = is_pollutant(technologies.comm_usage)

    # Check environmental commodities
    assert em.commodity.isin(envs.commodity).all()
    assert set(em.commodity.values) == set(envs.commodity[envs].values)

    # Test zero emissions cases
    techs = technologies.copy()
    techs.fixed_outputs.loc[{"commodity": envs}] = 0
    em_zero = emission(production, techs)
    # Check that all non-NaN values are zero
    assert (em_zero.where(~np.isnan(em_zero), 0) == 0).all()

    # Test zero production case
    em_zero_prod = emission(production * 0, technologies)
    assert (em_zero_prod.where(~np.isnan(em_zero_prod), 0) == 0).all()


def test_min_production(technologies: xr.Dataset, capacity: xr.DataArray, timeslice):
    """Test minimum production quantity."""
    # Test without minimum service factor
    technologies["minimum_service_factor"] = 0
    assert (minimum_production(technologies, capacity) == 0).all()

    # Test with minimum service factor
    technologies["minimum_service_factor"] = 0.5
    production = minimum_production(technologies, capacity)
    assert not (production == 0).all()
    assert (production <= maximum_production(technologies, capacity)).all()


def test_supply_single_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    # Setup single region data
    region = "USA"
    technologies = technologies.where(technologies.region == region, drop=True)
    capacity = capacity.where(capacity.region == region, drop=True)
    production = production.where(production.region == region, drop=True)

    # Create random demand
    demand = production.sum("asset") * np.random.rand(*production.sum("asset").shape)
    assert "region" not in demand.dims

    # Test supply matches demand for end-use commodities
    spl = supply(capacity, demand, technologies).sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_supply_multi_region(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)
    assert "region" in demand.dims

    # Test supply matches demand for end-use commodities by region
    spl = supply(capacity, demand, technologies).groupby("region").sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_supply_with_min_service(
    technologies: xr.Dataset,
    capacity: xr.DataArray,
    production: xr.DataArray,
    timeslice,
):
    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # Create random demand
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Test supply meets minimum production constraint
    spl = supply(capacity, demand, technologies)
    assert (spl >= minprod).all()


def test_production_amplitude(production: xr.DataArray, technologies: xr.Dataset):
    techs = broadcast_over_assets(technologies, production)
    result = production_amplitude(production, techs)
    assert set(result.dims) == set(production.dims) - {"commodity"}
