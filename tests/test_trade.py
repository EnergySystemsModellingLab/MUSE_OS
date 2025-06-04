from collections.abc import Mapping
from typing import Any

from pytest import approx, fixture

from muse import constraints as cs
from muse import examples
from muse.utilities import agent_concatenation, reduce_assets


@fixture
def constraints_args(sector="power", model="trade") -> Mapping[str, Any]:
    power = examples.sector(model=model, sector=sector)
    search_space = examples.search_space("power", model="trade")
    market = examples.matching_market("power", "trade")
    assets = agent_concatenation({agent.uuid: agent.assets for agent in power.agents})
    capacity = reduce_assets(assets.capacity, coords=("region", "technology"))

    return dict(
        demand=market.consumption.sel(year=2025, drop=True),
        capacity=capacity.sel(year=[2020, 2025]),
        search_space=search_space,
        market=market,
        technologies=power.technologies.sel(year=2025, drop=True),
    )


def test_demand_constraint(constraints_args):
    constraint = cs.demand(**constraints_args)
    assert set(constraint.b.dims) == {"timeslice", "dst_region", "commodity"}


def test_max_capacity_constraints(constraints_args):
    constraint = cs.max_capacity_expansion(**constraints_args)
    assert constraint.production == 0
    assert set(constraint.capacity.dims) == {"agent", "src_region"}
    assert set(constraint.b.dims) == {"replacement", "dst_region", "src_region"}
    assert set(constraint.agent.coords) == {"region", "agent"}
    assert ((constraint.region == constraint.src_region) == constraint.capacity).all()


def test_max_production(constraints_args):
    constraint = cs.max_production(**constraints_args)
    production_dims = {
        "timeslice",
        "commodity",
        "replacement",
        "agent",
        "dst_region",
        "src_region",
    }
    assert set(constraint.capacity.dims) == production_dims
    assert set(constraint.production.dims) == production_dims
    assert set(constraint.agent.coords) == {"region", "agent"}


def test_minimum_service(constraints_args):
    assert cs.minimum_service(**constraints_args) is None

    constraints_args["technologies"]["minimum_service_factor"] = 0.5
    constraint = cs.minimum_service(**constraints_args)
    service_dims = {"replacement", "agent", "commodity", "timeslice"}
    assert set(constraint.capacity.dims) == service_dims
    assert set(constraint.production.dims) == service_dims
    assert set(constraint.b.dims) == service_dims
    assert set(constraint.agent.coords) == {"region", "agent"}
    assert (constraint.capacity <= 0).all()


def test_search_space(constraints_args):
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


def get_agent_capacities(sector):
    """Helper to get concatenated agent capacities."""
    return agent_concatenation(
        {agent.uuid: agent.assets.capacity for agent in sector.agents}
    )


def test_power_sector_no_investment():
    power = examples.sector("power", "trade")
    market = examples.matching_market("power", "trade").sel(year=[2020, 2025])

    initial_capacity = get_agent_capacities(power)
    power.next(market)
    final_capacity = get_agent_capacities(power)

    assert (initial_capacity == final_capacity).all()


def test_power_sector_some_investment():
    power = examples.sector("power", "trade")
    market = examples.matching_market("power", "trade").sel(year=[2020, 2025])
    market.consumption[:] *= 1.5

    initial_capacity = get_agent_capacities(power)
    result = power.next(market)
    final_capacity = get_agent_capacities(power)

    assert "windturbine" not in initial_capacity.technology
    assert (
        final_capacity.sel(
            asset=final_capacity.technology == "windturbine", year=2025
        ).sum()
        < 1
    )
    assert "dst_region" not in result.dims
