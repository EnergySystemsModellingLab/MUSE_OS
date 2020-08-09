import numpy as np
import xarray as xr


def test_demand_constraint():
    from muse import examples, constraints as cs
    from muse.utilities import agent_concatenation

    power = examples.sector("power", "trade")
    market = examples.matching_market("power", "trade")
    assets = agent_concatenation({u.uuid: u.assets for u in list(power.agents)})

    constraint = cs.demand(
        demand=market.consumption.sel(year=2020, drop=True),
        assets=assets,
        search_space=xr.DataArray(0),
        market=None,
        technologies=power.technologies,
    )

    assert set(constraint.b.dims) == {"timeslice", "dst_region", "commodity"}


def test_lp_costs():
    from muse import examples
    from muse.constraints import lp_costs

    technologies = examples.technodata("power", model="trade")
    search_space = examples.search_space("power", model="trade")
    timeslices = examples.sector("power", model="trade").timeslices
    costs = (
        search_space
        * np.arange(np.prod(search_space.shape)).reshape(search_space.shape)
        * xr.ones_like(technologies.dst_region)
    )

    lpcosts = lp_costs(technologies.sel(year=2020), costs, timeslices)
    assert "capacity" in lpcosts.data_vars
    assert "production" in lpcosts.data_vars
    assert set(lpcosts.capacity.dims) == {"asset", "replacement", "dst_region"}
    assert set(lpcosts.production.dims) == {
        "asset",
        "replacement",
        "dst_region",
        "timeslice",
        "commodity",
    }
    assert "region" in lpcosts.asset.coords
    assert "agent" in lpcosts.asset.coords
    assert "installed" not in lpcosts.asset.coords
