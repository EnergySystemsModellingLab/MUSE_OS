from pytest import approx, fixture
from xarray import DataArray, Dataset


@fixture
def demand(technologies: Dataset, capacity: DataArray, market: DataArray) -> DataArray:
    from typing import Mapping, Hashable, Any
    from numpy.random import randint

    region = DataArray(list(set(capacity.region.values)), dims="region")
    coords: Mapping[Hashable, Any] = {
        "commodity": technologies.commodity,
        "year": capacity.year,
        "region": region,
        "timeslice": market.timeslice,
    }
    data = randint(0, 5, tuple(len(u) for u in coords.values()))
    return DataArray(data, coords=coords, dims=tuple(coords.keys()))


@fixture
def production(technologies: Dataset, capacity: DataArray) -> DataArray:
    from numpy.random import random

    comms = DataArray(
        random(len(technologies.commodity)),
        coords={"commodity": technologies.commodity},
        dims="commodity",
    )
    return capacity * comms


def make_array(array):
    from numpy.random import randint
    from xarray import DataArray

    data = randint(1, 5, len(array))
    return DataArray(data, dims=array.dims, coords=array.coords)


def test_supply_enduse(technologies, capacity, timeslice):
    """End-use part of supply."""
    from muse.quantities import supply, maximum_production
    from muse.commodities import is_enduse
    from numpy.random import random

    production = maximum_production(technologies, capacity)
    share = DataArray(
        random(timeslice.timeslice.shape),
        coords={"timeslice": timeslice.timeslice},
        dims="timeslice",
    )
    demand = (production.sum("asset") + 1) * share / share.sum()
    spl = supply(capacity, demand, technologies).where(
        is_enduse(technologies.comm_usage), 0
    )
    assert (abs(spl.sum("timeslice") - production) < 1e-12).all()
    assert (spl.sum("asset") < demand).all()

    demand = production.sum("asset") * 0.7 * share / share.sum()
    spl = supply(capacity, demand, technologies).where(
        is_enduse(technologies.comm_usage), 0
    )
    assert (spl.sum("timeslice") <= production + 1e-12).all()
    assert (
        abs(spl.sum("asset") - demand.where(production.sum("asset") > 0, 0)) < 1e-12
    ).all()


def test_supply_emissions(technologies, capacity):
    """Emission part of supply."""
    from xarray import broadcast
    from muse.quantities import supply, maximum_production, emission
    from muse.commodities import is_enduse, is_pollutant

    production = maximum_production(technologies, capacity)
    spl = supply(capacity, production.sum("asset") + 1, technologies)
    msn = emission(spl.where(is_enduse(spl.comm_usage), 0), technologies.fixed_outputs)
    actual, expected = broadcast(spl.sel(commodity=is_pollutant(spl.comm_usage)), msn)
    assert actual.values == approx(expected.values)


def test_gross_margin(technologies, capacity, market):
    from muse.quantities import gross_margin
    from muse.commodities import is_pollutant, is_fuel, is_enduse
    from xarray import broadcast

    # we modify the variables to have just the values we want for the testing
    technologies = technologies.sel(technology=technologies.technology == "soda_shaker")
    capacity = capacity.sel(asset=capacity.technology == "soda_shaker")
    capacity[:] = capa = 9

    # This will leave 2 environmental outputs and 4 fuel inputs.
    usage = technologies.comm_usage

    technologies.var_par[:] = vp = 2
    technologies.var_exp[:] = ve = 0.5
    technologies.fixed_inputs[{"commodity": is_fuel(usage)}] = fuels = 2
    technologies.fixed_outputs[{"commodity": is_pollutant(usage)}] = envs = 10
    technologies.fixed_outputs[{"commodity": is_enduse(usage)}] = prod = 5

    market.prices[:] = prices = 3
    market.prices[{"commodity": is_pollutant(usage)}] = env_prices = 6
    # We expect a DataArray with 1 replacement technology
    actual = gross_margin(technologies, capacity, market.prices)

    revenues = prices * prod * sum(is_enduse(usage))
    env_costs = env_prices * envs * sum(is_pollutant(usage))
    cons_costs = prices * fuels * sum(is_fuel(usage))
    var_costs = vp * (capa ** ve) * market.represent_hours / sum(market.represent_hours)
    expected = revenues - env_costs - cons_costs - var_costs

    expected, actual = broadcast(expected, actual)
    assert actual.values == approx(expected.values)


def test_decommissioning_demand(technologies, capacity):
    from muse.quantities import decommissioning_demand
    from muse.commodities import is_enduse

    years = [2010, 2015]
    capacity = capacity.interp(year=years)
    capacity.loc[{"year": 2010}] = current = 1.3
    capacity.loc[{"year": 2015}] = forecast = 1.0
    technologies.fixed_outputs[:] = fouts = 0.5
    technologies.utilization_factor[:] = ufac = 0.4
    decom = decommissioning_demand(technologies, capacity, years)
    assert set(decom.dims) == {"asset", "commodity", "year"}
    assert decom.sel(commodity=is_enduse(technologies.comm_usage)).values == approx(
        ufac * fouts * (current - forecast)
    )


def test_consumption_no_flex(technologies, production, market):
    from xarray import broadcast
    from muse.quantities import consumption
    from muse.commodities import is_fuel, is_enduse

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
    assert "timeslice" not in actual.dims
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)

    technologies.flexible_inputs[:] = 0
    actual = consumption(technologies, production, market.prices)
    expected = expected * market.represent_hours / market.represent_hours.sum()
    actual, expected = broadcast(actual, expected)
    assert actual.values == approx(expected.values)


def test_consumption_with_flex(technologies, production, market):
    from itertools import product
    from numpy.random import randint, choice
    from muse.quantities import consumption
    from muse.commodities import is_fuel, is_enduse

    techs = technologies.copy()
    techs.fixed_inputs[:] = 0
    techs.flexible_inputs[:] = 0
    consumables = is_fuel(techs.comm_usage)
    while (techs.flexible_inputs.sel(commodity=consumables) == 0).all():
        techs.flexible_inputs[:] = randint(0, 2, techs.flexible_inputs.shape) != 0
        techs.flexible_inputs[{"commodity": ~consumables}] = 0

    def one_dim(dimension):
        from numpy import arange
        from numpy.random import shuffle

        data = arange(len(dimension), dtype="int")
        shuffle(data)
        return DataArray(data, coords=dimension.coords, dims=dimension.dims)

    year = one_dim(production.year)
    asset = one_dim(production.asset)
    region = one_dim(market.region)
    timeslice = one_dim(market.timeslice)
    commodity = one_dim(market.commodity)
    hours = market.represent_hours / market.represent_hours.sum()

    prices = timeslice + commodity + year * region
    assert set(prices.dims) == set(market.prices.dims)
    assert set((asset + year + commodity).dims) == set(production.dims)
    noenduse = ~is_enduse(techs.comm_usage)
    production = asset * year + commodity
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
    allprods = [allprods[i] for i in choice(range(len(allprods)), 50, replace=False)]
    for (ts, asset, year) in allprods:
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
            expected = (
                prod * hours.sel(timeslice=ts) / ncomms * flexs.sel(commodity=comm)
            )
            assert expected.values == approx(actual.sel(coords).values)


def test_lifetime_LCOE_annual_cap_costs(market: Dataset, technologies: Dataset):
    from muse.quantities import lifetime_levelized_cost_of_energy as LCOE
    from xarray import broadcast

    technologies.fix_par[:] = 0
    technologies.var_par[:] = 0
    technologies.fixed_inputs[:] = 0
    technologies.fixed_outputs[:] = 0

    technologies.technical_life[:] = tf = 2
    technologies.interest_rate[:] = itr = 0.02
    technologies.cap_par[:] = cap = 3

    lcoe = LCOE(market.prices, technologies)
    hours = market.prices.represent_hours / market.prices.represent_hours.sum()
    expected, lcoe = broadcast(
        cap * itr * (1 + itr) ** tf / ((1 + itr) ** tf - 1) * hours, lcoe
    )
    assert lcoe.values == approx(expected.values)


def test_lifetime_LCOE_om(market: Dataset, technologies: Dataset):
    from muse.quantities import lifetime_levelized_cost_of_energy as LCOE

    technologies.fixed_inputs[:] = 0
    technologies.fixed_outputs[:] = 0
    technologies.cap_par[:] = 0

    technologies.fix_par[:] = fp = 1
    technologies.var_par[:] = vp = 2
    technologies.technical_life[:] = tf = 2
    technologies.interest_rate[:] = itr = 0.02

    lcoe = LCOE(market.prices, technologies)
    rates = sum(1 / (1 + itr) ** y for y in range(1, tf + 1))
    assert lcoe.values == approx(rates * (fp + vp))


def test_lifetime_LCOE_fuel(market: Dataset, technologies: Dataset):
    from muse.quantities import lifetime_levelized_cost_of_energy as LCOE

    technologies.fix_par[:] = 0
    technologies.var_par[:] = 0
    technologies.fixed_outputs[:] = 0
    technologies.cap_par[:] = 0

    finputs = technologies.fixed_inputs
    isfuel = finputs.any(u for u in finputs.dims if u != "commodity")
    finputs.loc[{"commodity": isfuel}] = fuels = 0.9
    technologies.technical_life[:] = tf = 5
    technologies.interest_rate[:] = itr = 0.02
    market.prices.loc[{"commodity": isfuel}] = p = 0.6

    lcoe = LCOE(market.prices, technologies)
    rates = sum(1 / (1 + itr) ** y for y in range(1, tf + 1))
    assert lcoe.values == approx(rates * p * fuels * isfuel.sum().values)


def test_lifetime_LCOE_envs(market: Dataset, technologies: Dataset):
    from muse.quantities import lifetime_levelized_cost_of_energy as LCOE
    from muse.commodities import is_pollutant

    technologies.fix_par[:] = 0
    technologies.var_par[:] = 0
    technologies.fixed_inputs[:] = 0
    technologies.cap_par[:] = 0

    isenv = is_pollutant(technologies.comm_usage)
    technologies.fixed_outputs.loc[{"commodity": isenv}] = envs = 0.9
    technologies.technical_life[:] = tf = 5
    technologies.interest_rate[:] = itr = 0.02
    market.prices.loc[{"commodity": isenv}] = p = 0.6

    lcoe = LCOE(market.prices, technologies)
    rates = sum(1 / (1 + itr) ** y for y in range(1, tf + 1))
    assert lcoe.values == approx(rates * p * envs * isenv.sum().values)


def test_lifetime_vs_annual_LCOE(market: Dataset, technologies: Dataset):
    from muse.quantities import lifetime_levelized_cost_of_energy as lifetime
    from muse.quantities import annual_levelized_cost_of_energy as annual
    from xarray import broadcast

    technologies.interest_rate[:] = 0
    technologies.technical_life[:] = 1

    base_year = int(market.year.min().values)
    life = lifetime(market.prices, technologies, base_year=base_year)
    annum = annual(market.prices.sel(year=base_year), technologies.sel(year=base_year))
    assert set(life.dims) == set(annum.dims)
    life, annum = broadcast(life, annum)
    assert life.values == approx(annum.values)


def test_production_aggregate_asset_view(capacity: DataArray, technologies: Dataset):
    """Production when capacity has format of agent.sector.

    E.g. capacity aggregated across agents.
    """
    from xarray import broadcast
    from muse.quantities import maximum_production
    from muse.commodities import is_enduse

    technologies: Dataset = technologies[  # type:ignore
        ["fixed_outputs", "utilization_factor"]
    ]

    enduses = is_enduse(technologies.comm_usage)
    assert enduses.any()

    technologies.fixed_outputs[:] = 1
    technologies.utilization_factor[:] = 1
    prod = maximum_production(technologies, capacity)
    assert set(prod.dims) == set(capacity.dims).union({"commodity"})
    assert prod.sel(commodity=~enduses).values == approx(0)
    prod, expected = broadcast(prod.sel(commodity=enduses), capacity)
    assert prod.values == approx(expected.values)

    technologies.fixed_outputs[:] = fouts = 2
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity)
    assert prod.sel(commodity=~enduses).values == approx(0)
    assert set(prod.dims) == set(capacity.dims).union({"commodity"})
    prod, expected = broadcast(prod.sel(commodity=enduses), capacity)
    assert prod.values == approx(fouts * ufact * expected.values)

    technologies.fixed_outputs[:] = fouts = 3
    technologies.utilization_factor[:] = ufact = 0.5
    prod = maximum_production(technologies, capacity)
    assert prod.sel(commodity=~enduses).values == approx(0)
    assert set(prod.dims) == set(capacity.dims).union({"commodity"})
    prod, expected = broadcast(prod.sel(commodity=enduses), capacity)
    assert prod.values == approx(fouts * ufact * expected.values)


def test_production_agent_asset_view(capacity: DataArray, technologies: Dataset):
    """Production when capacity has format of agent.assets.capacity."""
    from muse.utilities import reduce_assets, coords_to_multiindex

    capacity = coords_to_multiindex(reduce_assets(capacity)).unstack("asset").fillna(0)
    test_production_aggregate_asset_view(capacity, technologies)


def test_capacity_in_use(production: DataArray, technologies: Dataset):
    from xarray import broadcast
    from numpy.random import choice
    from muse.quantities import capacity_in_use
    from muse.commodities import is_enduse

    technologies: Dataset = technologies[  # type: ignore
        ["fixed_outputs", "utilization_factor"]
    ]
    production[:] = prod = 10
    technologies.fixed_outputs[:] = fout = 5
    technologies.utilization_factor[:] = ufac = 2

    enduses = is_enduse(technologies.comm_usage)
    capa = capacity_in_use(production, technologies, max_dim=None)
    assert "commodity" in capa.dims
    capa, expected = broadcast(capa, enduses * prod / fout / ufac)
    assert capa.values == approx(expected.values)

    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)

    maxcomm = choice(production.commodity.sel(commodity=enduses).values)
    production.loc[{"commodity": maxcomm}] = prod = 11
    capa = capacity_in_use(production, technologies)
    assert "commodity" not in capa.dims
    assert capa.values == approx(prod / fout / ufac)


def test_supply_cost(production: DataArray, timeslice: Dataset):
    from xarray import DataArray, broadcast
    from numpy import average
    from numpy.random import random
    from muse.quantities import supply_cost

    timeslice = timeslice.timeslice
    production = production.sel(year=production.year.min(), drop=True)
    # no zero production, because it does not sit well with np.average
    production[:] = random(production.shape)
    lcoe = DataArray(
        random((len(production.asset), len(timeslice))),
        coords={"timeslice": timeslice, "asset": production.asset},
        dims=("asset", "timeslice"),
    )
    production, lcoe = broadcast(production, lcoe)
    actual = supply_cost(production, lcoe, asset_dim="asset")
    expected = average(lcoe, weights=production, axis=production.get_axis_num("asset"))

    assert actual.values == approx(expected)


def test_supply_cost_zero_prod(production: DataArray, timeslice: Dataset):
    from xarray import DataArray, broadcast
    from numpy.random import randn
    from muse.quantities import supply_cost

    timeslice = timeslice.timeslice
    production = production.sel(year=production.year.min(), drop=True)
    production[:] = 0
    lcoe = DataArray(
        randn(len(production.asset), len(timeslice)),
        coords={"timeslice": timeslice, "asset": production.asset},
        dims=("asset", "timeslice"),
    )
    production, lcoe = broadcast(production, lcoe)
    actual = supply_cost(production, lcoe, asset_dim="asset")
    assert actual.values == approx(0e0)


def test_emission(production: DataArray, technologies: Dataset):
    from muse.quantities import emission
    from muse.commodities import is_pollutant, is_enduse

    envs = is_pollutant(technologies.comm_usage)
    technologies = technologies[["fixed_outputs"]]
    technologies.fixed_outputs[{"commodity": envs}] = fout = 1.5
    technologies.fixed_outputs[{"commodity": ~envs}] = 2

    enduses = is_enduse(technologies.comm_usage.sel(commodity=production.commodity))
    production[{"commodity": enduses}] = prod = 0.5
    production[{"commodity": ~enduses}] = 5

    em = emission(production, technologies)
    assert em.commodity.isin(envs.commodity).all()
    assert em.values == approx(fout * enduses.sum().values * prod)


def test_demand_matched_production(
    demand: DataArray, capacity: DataArray, technologies: Dataset
):
    from numpy.random import choice, randint, random
    from xarray import zeros_like
    from muse.quantities import maximum_production, demand_matched_production
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.commodities import is_enduse, CommodityUsage

    # try and make sure we have a few more outputs than the default fixture
    technologies.comm_usage[:] = choice(
        [CommodityUsage.PRODUCT] * 3 + list(set(technologies.comm_usage.values)),
        technologies.comm_usage.shape,
    )
    technologies.fixed_outputs[:] = random(technologies.fixed_outputs.shape)
    technologies.fixed_outputs[:] *= is_enduse(technologies.comm_usage)

    capacity = capacity.sel(year=capacity.year.min(), drop=True)
    max_prod = convert_timeslice(
        maximum_production(technologies, capacity),
        demand.timeslice,
        QuantityType.EXTENSIVE,
    )
    demand = max_prod.sum("asset")
    demand[:] *= choice([0, 1, 1 / 2, 1 / 3, 1 / 10], demand.shape)
    prices = zeros_like(demand)
    prices[:] = randint(1, 10, prices.shape)
    production = demand_matched_production(demand, prices, capacity, technologies)
    assert set(production.dims) == set(max_prod.dims).union(prices.dims, capacity.dims)
    assert (production <= max_prod + 1e-8).all()
