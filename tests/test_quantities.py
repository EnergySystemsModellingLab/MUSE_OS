import numpy as np
import xarray as xr
from pytest import approx, fixture, mark


@fixture
def technologies(technologies, capacity, timeslice):
    from muse.utilities import broadcast_over_assets

    return broadcast_over_assets(technologies, capacity)


@fixture
def production(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
) -> xr.DataArray:
    from muse.timeslices import broadcast_timeslice, distribute_timeslice

    return (
        broadcast_timeslice(capacity)
        * distribute_timeslice(technologies.fixed_outputs)
        * broadcast_timeslice(technologies.utilization_factor)
    )


def test_consumption(technologies, production, market):
    from muse.quantities import consumption

    # Prices not provided, so flexible inputs are ignored
    consump = consumption(technologies, production)
    assert set(production.dims) == set(consump.dims)

    # Prices provided, but no flexible inputs -> should be the same as above
    technologies.flexible_inputs[:] = 0
    consump2 = consumption(technologies, production, market.prices)
    assert consump2.values == approx(consump.values)

    # Flexible inputs considered
    consump3 = consumption(technologies, production, market.prices)
    assert set(production.dims) == set(consump3.dims)


def test_production_aggregate_asset_view(
    technologies: xr.Dataset, capacity: xr.DataArray
):
    """Production when capacity has format of agent.sector.

    E.g. capacity aggregated across agents.
    """
    from muse.commodities import is_enduse
    from muse.quantities import maximum_production

    technologies: xr.Dataset = technologies[  # type:ignore
        ["fixed_outputs", "utilization_factor"]
    ]

    enduses = is_enduse(technologies.comm_usage)
    assert enduses.any()

    technologies.fixed_outputs[:] = 1
    technologies.utilization_factor[:] = 1
    prod = maximum_production(technologies, capacity)
    assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
    assert prod.sel(commodity=~enduses).values == approx(0)
    prod, expected = xr.broadcast(
        prod.sel(commodity=enduses).sum("timeslice"), capacity
    )
    assert prod.values == approx(expected.values)

    technologies.fixed_outputs[:] = fouts = 2
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity)
    assert prod.sel(commodity=~enduses).values == approx(0)
    assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
    prod, expected = xr.broadcast(
        prod.sel(commodity=enduses).sum("timeslice"), capacity
    )
    assert prod.values == approx(fouts * ufact * expected.values)

    technologies.fixed_outputs[:] = fouts = 3
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity)
    assert prod.sel(commodity=~enduses).values == approx(0)
    assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
    prod, expected = xr.broadcast(
        prod.sel(commodity=enduses).sum("timeslice"), capacity
    )
    assert prod.values == approx(fouts * ufact * expected.values)


@mark.xfail
def test_production_agent_asset_view(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
):
    """Production when capacity has format of agent.assets.capacity.

    TODO: This requires a fully-explicit technologies dataset. Need to rework the
    fixtures.
    """
    from muse.utilities import coords_to_multiindex, reduce_assets

    capacity = coords_to_multiindex(reduce_assets(capacity)).unstack("asset").fillna(0)
    test_production_aggregate_asset_view(technologies, capacity)


def test_capacity_in_use(production: xr.DataArray, technologies: xr.Dataset):
    from muse.commodities import is_enduse
    from muse.quantities import capacity_in_use

    technologies: xr.Dataset = technologies[  # type: ignore
        ["fixed_outputs", "utilization_factor"]
    ]
    production[:] = prod = 10
    technologies.fixed_outputs[:] = fout = 5
    technologies.utilization_factor[:] = ufac = 2

    enduses = is_enduse(technologies.comm_usage)
    capa = capacity_in_use(production, technologies, max_dim=None)
    assert "commodity" in capa.dims
    capa, expected = xr.broadcast(capa, enduses * prod / fout / ufac)
    assert capa.values == approx(expected.values)

    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)

    maxcomm = np.random.choice(production.commodity.sel(commodity=enduses).values)
    production.loc[{"commodity": maxcomm}] = prod = 11
    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)


def test_emission(production: xr.DataArray, technologies: xr.Dataset):
    from muse.commodities import is_pollutant
    from muse.quantities import emission

    em = emission(production, technologies)

    # Check that all environmental commodities are in the result
    envs = is_pollutant(technologies.comm_usage)
    assert em.commodity.isin(envs.commodity).all()

    # Check that no non-environmental commodities are in the result
    assert set(em.commodity.values) == set(envs.commodity[envs].values)

    # If fixed_outputs for env commodities are zero, then emissions should be zero
    techs = technologies.copy()
    techs.fixed_outputs.loc[{"commodity": envs}] = 0
    em = emission(production, techs)

    # If production is zero, then emissions should be zero
    em = emission(production * 0, technologies)
    assert (em == 0).all()


def test_min_production(technologies, capacity, timeslice):
    """Test minimum production quantity."""
    from muse.quantities import maximum_production, minimum_production

    # If no minimum service factor is defined, the minimum production is zero
    assert "minimum_service_factor" not in technologies
    production = minimum_production(technologies, capacity)
    assert (production == 0).all()

    # If minimum service factor is defined, then the minimum production is not zero
    # and it is less than the maximum production
    technologies["minimum_service_factor"] = 0.5
    production = minimum_production(technologies, capacity)
    assert not (production == 0).all()
    assert (production <= maximum_production(technologies, capacity)).all()


def test_supply_single_region(technologies, capacity, production, timeslice):
    from muse.commodities import is_enduse
    from muse.quantities import supply

    # Select data for a single region
    region = "USA"
    technologies = technologies.where(technologies.region == region, drop=True)
    capacity = capacity.where(capacity.region == region, drop=True)
    production = production.where(production.region == region, drop=True)

    # Random demand within the bounds of the maximum production
    demand = production.sum("asset")
    demand = demand * np.random.rand(*demand.shape)
    assert "region" not in demand.dims

    # Calculate supply
    spl = supply(capacity, demand, technologies)

    # Total supply across assets should equal demand (for end-use commodities)
    spl = spl.sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_supply_multi_region(technologies, capacity, production, timeslice):
    from muse.commodities import is_enduse
    from muse.quantities import supply

    # Random demand within the bounds of the maximum production
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Calculate supply
    assert "region" in demand.dims
    spl = supply(capacity, demand, technologies)

    # Total supply across assets within each region should equal demand
    # (for end-use commodities)
    spl = spl.groupby("region").sum("asset")
    enduses = is_enduse(technologies.comm_usage)
    assert abs(spl.sel(commodity=enduses) - demand.sel(commodity=enduses)).sum() < 1e-5


def test_supply_with_min_service(technologies, capacity, production, timeslice):
    from muse.quantities import minimum_production, supply

    # Calculate minimum production
    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # Random demand within the bounds of the maximum production
    demand = production.groupby("region").sum("asset")
    demand = demand * np.random.rand(*demand.shape)

    # Calculate supply
    spl = supply(capacity, demand, technologies)

    # Supply should be greater than or equal to the minimum production
    assert (spl >= minprod).all()


def test_production_amplitude(production, technologies):
    from muse.quantities import production_amplitude
    from muse.utilities import broadcast_over_assets

    techs = broadcast_over_assets(technologies, production)
    result = production_amplitude(production, techs)
    assert set(result.dims) == set(production.dims) - {"commodity"}
