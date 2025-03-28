from collections.abc import Mapping
from typing import Any

from pytest import approx, fixture


@fixture
def constraints_args(sector="power", model="trade") -> Mapping[str, Any]:
    from muse import examples
    from muse.utilities import agent_concatenation, reduce_assets

    power = examples.sector(model=model, sector=sector)
    search_space = examples.search_space("power", model="trade")
    market = examples.matching_market("power", "trade")
    assets = agent_concatenation({u.uuid: u.assets for u in list(power.agents)})
    capacity = reduce_assets(assets.capacity, coords=("region", "technology"))
    return dict(
        demand=market.consumption.sel(year=2025, drop=True),
        capacity=capacity.sel(year=[2020, 2025]),
        search_space=search_space,
        market=market,
        technologies=power.technologies.sel(year=2025, drop=True),
    )


def test_demand_constraint(constraints_args):
    from muse import constraints as cs

    constraint = cs.demand(**constraints_args)
    assert set(constraint.b.dims) == {"timeslice", "dst_region", "commodity"}


def test_max_capacity_constraints(constraints_args):
    from muse import constraints as cs

    constraint = cs.max_capacity_expansion(**constraints_args)
    assert constraint.production == 0
    assert set(constraint.capacity.dims) == {"agent", "src_region"}
    assert ((constraint.region == constraint.src_region) == constraint.capacity).all()
    assert set(constraint.b.dims) == {"replacement", "dst_region", "src_region"}
    assert set(constraint.agent.coords) == {"region", "agent"}


def test_max_production(constraints_args):
    from muse import constraints as cs

    constraint = cs.max_production(**constraints_args)
    dims = {
        "timeslice",
        "commodity",
        "replacement",
        "agent",
        "timeslice",
        "dst_region",
        "src_region",
    }
    assert set(constraint.capacity.dims) == dims
    assert set(constraint.production.dims) == dims
    assert set(constraint.agent.coords) == {"region", "agent"}


def test_minimum_service(constraints_args):
    from muse import constraints as cs

    assert cs.minimum_service(**constraints_args) is None

    constraints_args["technologies"]["minimum_service_factor"] = 0.5
    constraint = cs.minimum_service(**constraints_args)
    dims = {"replacement", "agent", "commodity", "timeslice"}
    assert set(constraint.capacity.dims) == dims
    assert set(constraint.production.dims) == dims
    assert set(constraint.b.dims) == dims
    assert (constraint.capacity <= 0).all()
    assert set(constraint.agent.coords) == {"region", "agent"}


def test_search_space(constraints_args):
    from muse import constraints as cs

    search_space = constraints_args["search_space"]
    search_space[:] = 1
    assert cs.search_space(**constraints_args) is None

    search_space[:] = search_space * (search_space.region == "R1")
    constraint = cs.search_space(**constraints_args)
    assert constraint.b.values == approx(0)
    assert constraint.production == 0
    assert set(constraint.b.dims) == {"replacement", "agent"}
    assert set(constraint.capacity.dims) == {"replacement", "agent"}
    assert set(constraint.agent.coords) == {"region", "agent"}


def test_power_sector_no_investment():
    from muse import examples
    from muse.utilities import agent_concatenation

    power = examples.sector("power", "trade")
    market = examples.matching_market("power", "trade").sel(year=[2020, 2025])

    initial = agent_concatenation({u.uuid: u.assets.capacity for u in power.agents})
    power.next(market)
    final = agent_concatenation({u.uuid: u.assets.capacity for u in power.agents})

    assert (initial == final).all()


def test_power_sector_some_investment():
    from muse import examples
    from muse.utilities import agent_concatenation

    power = examples.sector("power", "trade")
    market = examples.matching_market("power", "trade").sel(year=[2020, 2025])
    market.consumption[:] *= 1.5

    initial = agent_concatenation({u.uuid: u.assets.capacity for u in power.agents})
    result = power.next(market)
    final = agent_concatenation({u.uuid: u.assets.capacity for u in power.agents})
    assert "windturbine" not in initial.technology
    assert final.sel(asset=final.technology == "windturbine", year=2025).sum() < 1
    assert "dst_region" not in result.dims
