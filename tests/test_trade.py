import xarray as xr


def _matching_market(technologies, assets, timeslice):
    """A market which matches stocks exactly."""
    from muse.timeslices import convert_timeslice, QuantityType
    from muse.quantities import maximum_production, consumption
    from numpy.random import random

    market = xr.Dataset()
    production = convert_timeslice(
        maximum_production(technologies, assets.capacity),
        timeslice,
        QuantityType.EXTENSIVE,
    )
    market["supply"] = production.sum("asset").rename(dst_region="region")
    market["consumption"] = (
        consumption(technologies, production)
        .groupby("region")
        .sum(("asset", "dst_region"))
        + market.supply
    )
    market["prices"] = market.supply.dims, random(market.supply.shape)
    return market


def test_demand_constraint():
    from muse import examples, constraints as cs
    from muse.utilities import agent_concatenation

    power = examples.sector("power", "trade")
    assets = agent_concatenation({u.uuid: u.assets for u in list(power.agents)})
    market = _matching_market(power.technologies, assets, power.timeslices)

    constraint = cs.demand(
        demand=market.consumption.sel(year=2020, drop=True),
        assets=assets,
        search_space=xr.DataArray(0),
        market=None,
        technologies=power.technologies,
    )

    assert set(constraint.b.dims) == {"timeslice", "dst_region", "commodity"}
