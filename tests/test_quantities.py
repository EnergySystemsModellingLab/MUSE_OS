import numpy as np
import xarray as xr
from pytest import approx, fixture


@fixture
def production(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
) -> xr.DataArray:
    from muse.timeslices import broadcast_timeslice, distribute_timeslice
    from muse.utilities import broadcast_techs

    techs = broadcast_techs(technologies, capacity)
    return (
        broadcast_timeslice(capacity)
        * distribute_timeslice(techs.fixed_outputs)
        * broadcast_timeslice(techs.utilization_factor)
    )


def test_supply_enduse(technologies, capacity, timeslice):
    """End-use part of supply."""
    from muse.commodities import is_enduse
    from muse.quantities import maximum_production, supply

    production = maximum_production(technologies, capacity)
    demand = production.sum("asset") + 1
    spl = supply(capacity, demand, technologies).where(
        is_enduse(technologies.comm_usage), 0
    )
    assert (abs(spl - production) < 1e-12).all()
    assert (spl.sum("asset") < demand).all()

    demand = production.sum("asset") * 0.7
    spl = supply(capacity, demand, technologies).where(
        is_enduse(technologies.comm_usage), 0
    )
    assert (spl <= production + 1e-12).all()
    assert (
        abs(spl.sum("asset") - demand.where(production.sum("asset") > 0, 0)) < 1e-12
    ).all()


def test_supply_emissions(technologies, capacity, timeslice):
    """Emission part of supply."""
    from muse.commodities import is_enduse, is_pollutant
    from muse.quantities import emission, maximum_production, supply

    production = maximum_production(technologies, capacity)
    spl = supply(capacity, production.sum("asset") + 1, technologies)
    msn = emission(spl.where(is_enduse(spl.comm_usage), 0), technologies.fixed_outputs)
    actual, expected = xr.broadcast(
        spl.sel(commodity=is_pollutant(spl.comm_usage)), msn
    )
    assert actual.values == approx(expected.values)


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
    capacity: xr.DataArray, technologies: xr.Dataset
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


def test_production_agent_asset_view(
    capacity: xr.DataArray, technologies: xr.Dataset, timeslice
):
    """Production when capacity has format of agent.assets.capacity."""
    from muse.utilities import coords_to_multiindex, reduce_assets

    capacity = coords_to_multiindex(reduce_assets(capacity)).unstack("asset").fillna(0)
    test_production_aggregate_asset_view(capacity, technologies)


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


def test_supply_capped_by_min_service(technologies, capacity, timeslice):
    """Test supply is capped by the minimum service."""
    from muse.commodities import CommodityUsage
    from muse.quantities import minimum_production, supply

    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity)

    # If minimum service factor is defined, then the minimum production is not zero
    assert not (minprod == 0).all()

    # And even if the demand is smaller than the minimum production, the supply
    # should be equal to the minimum production
    demand = minprod / 2
    spl = supply(capacity, demand, technologies)
    spl = spl.sel(commodity=spl.comm_usage == CommodityUsage.PRODUCT).sum(
        ["year", "asset"]
    )
    minprod = minprod.sel(commodity=minprod.comm_usage == CommodityUsage.PRODUCT).sum(
        ["year", "asset"]
    )
    assert (spl == approx(minprod)).all()

    # But if there is not minimum service factor, the supply should be equal to the
    # demand and should not be capped by the minimum production
    del technologies["minimum_service_factor"]
    spl = supply(capacity, demand, technologies)
    spl = spl.sel(commodity=spl.comm_usage == CommodityUsage.PRODUCT).sum(
        ["year", "asset"]
    )
    demand = demand.sel(commodity=demand.comm_usage == CommodityUsage.PRODUCT).sum(
        ["year", "asset"]
    )
    assert (spl == approx(demand)).all()
    assert (spl <= minprod).all()


def test_production_amplitude(production, technologies):
    from muse.quantities import production_amplitude
    from muse.utilities import broadcast_techs

    techs = broadcast_techs(technologies, production)
    result = production_amplitude(production, techs)
    assert set(result.dims) == set(production.dims) - {"commodity"}
