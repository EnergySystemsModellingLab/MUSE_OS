from typing import Any, Mapping, Text

import numpy as np
import xarray as xr
from pytest import fixture


@fixture
def constraints_args(sector="power", model="trade") -> Mapping[Text, Any]:
    from muse import examples
    from muse.utilities import reduce_assets, agent_concatenation

    power = examples.sector(model=model, sector=sector)
    market = examples.matching_market("power", "trade")
    search_space = examples.search_space("power", model="trade")
    market = examples.matching_market("power", "trade")
    assets = reduce_assets(
        agent_concatenation({u.uuid: u.assets for u in list(power.agents)}),
        coords=["agent", "technology", "region"],
    ).set_coords(["agent", "technology", "region"])
    return dict(
        demand=market.consumption.sel(year=2020, drop=True),
        assets=assets,
        search_space=search_space,
        market=market,
        technologies=power.technologies,
    )


def test_demand_constraint(constraints_args):
    from muse import constraints as cs

    constraint = cs.demand(**constraints_args)
    assert set(constraint.b.dims) == {"timeslice", "dst_region", "commodity"}


def test_max_capacity_constraints(constraints_args):
    from muse import constraints as cs

    constraint = cs.max_capacity_expansion(**constraints_args)
    assert constraint.capacity == 1
    assert constraint.production == 0
    assert set(constraint.b.dims) == {"replacement", "dst_region", "region"}


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
    assert "technology" not in lpcosts.asset.coords
