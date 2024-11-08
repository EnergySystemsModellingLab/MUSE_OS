from typing import cast

import numpy as np
import xarray as xr
from pytest import approx, fixture


@fixture
def demand(
    technologies: xr.Dataset, capacity: xr.DataArray, market: xr.DataArray
) -> xr.DataArray:
    from collections.abc import Hashable, Mapping
    from typing import Any

    region = xr.DataArray(list(set(capacity.region.values)), dims="region")
    coords: Mapping[Hashable, Any] = {
        "commodity": technologies.commodity,
        "year": capacity.year,
        "region": region,
        "timeslice": market.timeslice,
    }
    data = np.random.randint(0, 5, tuple(len(u) for u in coords.values()))
    return xr.DataArray(data, coords=coords, dims=tuple(coords.keys()))


@fixture
def production(
    technologies: xr.Dataset, capacity: xr.DataArray, timeslice
) -> xr.DataArray:
    from numpy.random import random

    from muse.timeslices import QuantityType, convert_timeslice

    comms = xr.DataArray(
        random(len(technologies.commodity)),
        coords={"commodity": technologies.commodity},
        dims="commodity",
    )
    return capacity * convert_timeslice(comms, timeslice, QuantityType.EXTENSIVE)


def make_array(array):
    data = np.random.randint(1, 5, len(array))
    return xr.DataArray(data, dims=array.dims, coords=array.coords)


def test_supply_enduse(technologies, capacity, timeslice):
    """End-use part of supply."""
    from muse.commodities import is_enduse
    from muse.quantities import maximum_production, supply
    from muse.timeslices import QuantityType, convert_timeslice

    production = maximum_production(technologies, capacity, timeslice)
    demand = convert_timeslice(
        production.sum("asset") + 1, timeslice, QuantityType.EXTENSIVE
    )
    spl = supply(capacity, demand, technologies).where(
        is_enduse(technologies.comm_usage), 0
    )
    assert (abs(spl - production) < 1e-12).all()
    assert (spl.sum("asset") < demand).all()

    demand = convert_timeslice(
        production.sum("asset") * 0.7, timeslice, QuantityType.EXTENSIVE
    )
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

    production = maximum_production(technologies, capacity, timeslices=timeslice)
    spl = supply(capacity, production.sum("asset") + 1, technologies)
    msn = emission(spl.where(is_enduse(spl.comm_usage), 0), technologies.fixed_outputs)
    actual, expected = xr.broadcast(
        spl.sel(commodity=is_pollutant(spl.comm_usage)), msn
    )
    assert actual.values == approx(expected.values)


def test_gross_margin(technologies, capacity, market, timeslice):
    from muse.commodities import is_enduse, is_fuel, is_pollutant
    from muse.quantities import gross_margin

    """
    Gross margin refers to the calculation
    .. _here:
    https://www.investopedia.com/terms/g/grossmargin.asp
    """
    # we modify the variables to have just the values we want for the testing
    selected = capacity.technology.values[0]

    technologies = technologies.sel(technology=technologies.technology == selected)
    capa = capacity.where(capacity.technology == selected, drop=True)

    # Filtering commodity outputs
    usage = technologies.comm_usage

    technologies.var_par[:] = vp = 2
    technologies.var_exp[:] = ve = 0.5
    technologies.fixed_inputs[{"commodity": is_fuel(usage)}] = fuels = 2
    technologies.fixed_outputs[{"commodity": is_pollutant(usage)}] = envs = 10
    technologies.fixed_outputs[{"commodity": is_enduse(usage)}] = prod = 5

    market.prices[:] = prices = 3
    market.prices[{"commodity": is_pollutant(usage)}] = env_prices = 6
    # We expect a xr.DataArray with 1 replacement technology
    actual = gross_margin(technologies, capa, market.prices)

    revenues = prices * prod * sum(is_enduse(usage))
    env_costs = env_prices * envs * sum(is_pollutant(usage))
    cons_costs = prices * fuels * sum(is_fuel(usage))
    var_costs = vp * ((prod * sum(is_enduse(usage))) ** ve)

    expected = revenues - env_costs - cons_costs - var_costs
    expected *= 100 / revenues

    expected, actual = xr.broadcast(expected, actual)
    assert actual.values == approx(expected.values)


def test_decommissioning_demand(technologies, capacity, timeslice):
    from muse.commodities import is_enduse
    from muse.quantities import decommissioning_demand

    years = [2010, 2015]
    capacity = capacity.interp(year=years)
    capacity.loc[{"year": 2010}] = current = 1.3
    capacity.loc[{"year": 2015}] = forecast = 1.0
    technologies.fixed_outputs[:] = fouts = 0.5
    technologies.utilization_factor[:] = ufac = 0.4
    decom = decommissioning_demand(technologies, capacity, timeslice, years)
    assert set(decom.dims) == {"asset", "commodity", "year", "timeslice"}
    assert decom.sel(commodity=is_enduse(technologies.comm_usage)).sum(
        "timeslice"
    ).values == approx(ufac * fouts * (current - forecast))


def test_consumption_no_flex(technologies, production, market):
    from muse.commodities import is_enduse, is_fuel
    from muse.quantities import consumption

    fins = (
        technologies.fixed_inputs.where(is_fuel(technologies.comm_usage), 0)
        .interp(year=sorted(set(production.installed.values)), method="slinear")
        .sel(
            technology=production.technology,
            region=production.region,
            year=production.installed,
        )
    )
    services = technologies.commodity.sel(commodity=is_enduse(technologies.comm_usage))
    expected = (
        (production.rename(commodity="comm_in") * fins)
        .sel(comm_in=production.commodity.isin(services).rename(commodity="comm_in"))
        .sum("comm_in")
    )

    actual = consumption(technologies, production)
    assert set(actual.dims) == set(expected.dims)
    assert actual.values == approx(expected.values)

    technologies.flexible_inputs[:] = 0
    actual = consumption(technologies, production, market.prices)
    assert actual.values == approx(expected.values)


def test_consumption_with_flex(technologies, production, market, timeslice):
    from itertools import product

    from muse.commodities import is_enduse, is_fuel
    from muse.quantities import consumption
    from muse.timeslices import QuantityType, convert_timeslice

    techs = technologies.copy()
    techs.fixed_inputs[:] = 0
    techs.flexible_inputs[:] = 0
    consumables = is_fuel(techs.comm_usage)
    while (techs.flexible_inputs.sel(commodity=consumables) == 0).all():
        techs.flexible_inputs[:] = (
            np.random.randint(0, 2, techs.flexible_inputs.shape) != 0
        )
        techs.flexible_inputs[{"commodity": ~consumables}] = 0

    def one_dim(dimension):
        from numpy import arange
        from numpy.random import shuffle

        data = arange(len(dimension), dtype="int")
        shuffle(data)
        return xr.DataArray(data, coords=dimension.coords, dims=dimension.dims)

    year = one_dim(production.year)
    asset = one_dim(production.asset)
    region = one_dim(market.region)
    timeslice = one_dim(market.timeslice)
    commodity = one_dim(market.commodity)

    prices = timeslice + commodity + year * region
    assert set(prices.dims) == set(market.prices.dims)
    noenduse = ~is_enduse(techs.comm_usage)
    production = convert_timeslice(
        asset * year + commodity, timeslice, QuantityType.EXTENSIVE
    )
    production.loc[{"commodity": noenduse}] = 0

    actual = consumption(technologies, production, prices)
    assert set(actual.dims) == {"year", "timeslice", "asset", "commodity"}
    assert (year.year == actual.year).all()
    assert (timeslice.timeslice == actual.timeslice).all()
    assert (asset.asset == actual.asset).all()
    assert (commodity.commodity == actual.commodity).all()

    fuels = techs.commodity.loc[{"commodity": consumables}].values
    dims = ("timeslice", "asset", "year")
    allprods = list(product(*(actual[u] for u in dims)))
    allprods = [
        allprods[i] for i in np.random.choice(range(len(allprods)), 50, replace=False)
    ]
    for ts, asset, year in allprods:
        flexs = techs.flexible_inputs.sel(
            region=asset.region, technology=asset.technology
        ).interp(year=asset.installed, method="slinear")
        comm_prices = prices.sel(region=asset.region, year=year, timeslice=ts)
        comm_prices = [int(p) for p, f in zip(comm_prices, flexs) if f > 0]
        min_price = min(comm_prices) if comm_prices else None
        ncomms = max(len([u for u in comm_prices if u == min_price]), 1)
        for comm in fuels:
            current_price = prices.sel(
                region=asset.region, year=year, timeslice=ts, commodity=comm
            )
            coords = dict(timeslice=ts, year=year, asset=asset, commodity=comm)
            if current_price != min_price:
                assert actual.sel(coords).values == approx(0)
                continue
            prod = production.sel(asset=asset, year=year).sum("commodity")
            expected = prod.sel(timeslice=ts) / ncomms * flexs.sel(commodity=comm)
            assert expected.values == approx(actual.sel(coords).values)


def test_production_aggregate_asset_view(
    capacity: xr.DataArray, technologies: xr.Dataset, timeslice: xr.DataArray
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
    prod = maximum_production(technologies, capacity, timeslices=timeslice)
    assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
    assert prod.sel(commodity=~enduses).values == approx(0)
    prod, expected = xr.broadcast(
        prod.sel(commodity=enduses).sum("timeslice"), capacity
    )
    assert prod.values == approx(expected.values)

    technologies.fixed_outputs[:] = fouts = 2
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity, timeslices=timeslice)
    assert prod.sel(commodity=~enduses).values == approx(0)
    assert set(prod.dims) == set(capacity.dims).union({"commodity", "timeslice"})
    prod, expected = xr.broadcast(
        prod.sel(commodity=enduses).sum("timeslice"), capacity
    )
    assert prod.values == approx(fouts * ufact * expected.values)

    technologies.fixed_outputs[:] = fouts = 3
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity, timeslices=timeslice)
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
    test_production_aggregate_asset_view(capacity, technologies, timeslice)


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


def test_supply_cost(production: xr.DataArray, timeslice: xr.Dataset):
    from numpy import average
    from numpy.random import random

    from muse.costs import supply_cost

    timeslice = timeslice.timeslice
    production = production.sel(year=production.year.min(), drop=True)
    # no zero production, because it does not sit well with np.average
    production[:] = random(production.shape)
    lcoe = xr.DataArray(
        random((len(production.asset), len(timeslice))),
        coords={"timeslice": timeslice, "asset": production.asset},
        dims=("asset", "timeslice"),
    )

    production, lcoe = xr.broadcast(production, lcoe)
    actual = supply_cost(production, lcoe, asset_dim="asset")
    for region in set(production.region.values):
        expected = average(
            lcoe.sel(asset=production.region == region),
            weights=production.sel(asset=production.region == region),
            axis=production.get_axis_num("asset"),
        )

        assert actual.sel(region=region).values == approx(expected)


def test_supply_cost_zero_prod(production: xr.DataArray, timeslice: xr.Dataset):
    from numpy.random import randn

    from muse.costs import supply_cost

    timeslice = timeslice.timeslice
    production = production.sel(year=production.year.min(), drop=True)
    production[:] = 0
    lcoe = xr.DataArray(
        randn(len(production.asset), len(timeslice)),
        coords={"timeslice": timeslice, "asset": production.asset},
        dims=("asset", "timeslice"),
    )
    production, lcoe = xr.broadcast(production, lcoe)
    actual = supply_cost(production, lcoe, asset_dim="asset")
    assert actual.values == approx(0e0)


def test_emission(production: xr.DataArray, technologies: xr.Dataset):
    from muse.commodities import is_enduse, is_pollutant
    from muse.quantities import emission

    envs = is_pollutant(technologies.comm_usage)
    technologies = cast(xr.Dataset, technologies[["fixed_outputs"]])
    technologies.fixed_outputs[{"commodity": envs}] = fout = 1.5
    technologies.fixed_outputs[{"commodity": ~envs}] = 2

    enduses = is_enduse(technologies.comm_usage.sel(commodity=production.commodity))
    production[{"commodity": enduses}] = prod = 0.5
    production[{"commodity": ~enduses}] = 5

    em = emission(production, technologies)
    assert em.commodity.isin(envs.commodity).all()
    assert em.values == approx(fout * enduses.sum().values * prod)


def test_demand_matched_production(
    demand: xr.DataArray, capacity: xr.DataArray, technologies: xr.Dataset
):
    from muse.commodities import CommodityUsage, is_enduse
    from muse.quantities import demand_matched_production, maximum_production

    # try and make sure we have a few more outputs than the default fixture
    technologies.comm_usage[:] = np.random.choice(
        [CommodityUsage.PRODUCT] * 3 + list(set(technologies.comm_usage.values)),
        technologies.comm_usage.shape,
    )
    technologies.fixed_outputs[:] = np.random.random(technologies.fixed_outputs.shape)
    technologies.fixed_outputs[:] *= is_enduse(technologies.comm_usage)

    capacity = capacity.sel(year=capacity.year.min(), drop=True)
    max_prod = maximum_production(technologies, capacity, timeslices=demand.timeslice)
    demand = max_prod.sum("asset")
    demand[:] *= np.random.choice([0, 1, 1 / 2, 1 / 3, 1 / 10], demand.shape)
    prices = xr.zeros_like(demand)
    prices[:] = np.random.randint(1, 10, prices.shape)
    production = demand_matched_production(demand, prices, capacity, technologies)
    assert set(production.dims) == set(max_prod.dims).union(prices.dims, capacity.dims)
    assert (production <= max_prod + 1e-8).all()


def test_min_production(technologies, capacity, timeslice):
    """Test minimum production quantity."""
    from muse.quantities import maximum_production, minimum_production

    # If no minimum service factor is defined, the minimum production is zero
    assert "minimum_service_factor" not in technologies
    production = minimum_production(technologies, capacity, timeslice)
    assert (production == 0).all()

    # If minimum service factor is defined, then the minimum production is not zero
    # and it is less than the maximum production
    technologies["minimum_service_factor"] = 0.5
    production = minimum_production(technologies, capacity, timeslice)
    assert not (production == 0).all()
    assert (production <= maximum_production(technologies, capacity, timeslice)).all()


def test_supply_capped_by_min_service(technologies, capacity, timeslice):
    """Test supply is capped by the minimum service."""
    from muse.commodities import CommodityUsage
    from muse.quantities import minimum_production, supply

    technologies["minimum_service_factor"] = 0.3
    minprod = minimum_production(technologies, capacity, timeslice)

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
