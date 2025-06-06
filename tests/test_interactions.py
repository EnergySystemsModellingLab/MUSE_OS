"""Test agent interactions."""

from collections import namedtuple
from itertools import chain

import pytest
from numpy.random import choice
from pytest import fixture, mark

from muse.errors import NoInteractionsFound
from muse.interactions import (
    agents_groupby,
    factory,
    new_to_retro_net,
    register_agent_interaction,
)


@fixture
def agents():
    Agent = namedtuple("Agent", ["region", "name", "category", "assets"])
    sample_data = {
        "regions": ["Area52", "Bermuda Triangle", "City of London"],
        "names": ["John", "Joe", "Jill"],
        "categories": ["yup", "nope"],
    }

    # Generate unique combinations of region, name, and category
    combinations = {
        tuple(items)
        for items in zip(*(choice(data, 40) for data in sample_data.values()))
    }

    return [Agent(*combo, [None]) for combo in combinations]


def test_groupby(agents):
    grouped = agents_groupby(agents, ("category", "name"))

    # Verify group sizes and agent preservation
    assert sum(len(group) for group in grouped.values()) == len(agents)
    assert set(map(id, chain(*grouped.values()))) == set(map(id, agents))

    # Verify correct grouping keys and contents
    assert set(grouped.keys()) == {(a.category, a.name) for a in agents}
    for (cat, name), group in grouped.items():
        assert all(a.category == cat and a.name == name for a in group)


def test_new_to_retro_net(agents):
    net = new_to_retro_net(agents, "nope")

    # Verify network properties
    assert sum(len(group) for group in net) <= len(agents)
    assert set(map(id, chain(*net))).issubset(map(id, agents))
    assert all(len(list(group)) == 2 for group in net)

    # Verify group constraints
    for group in net:
        assert len({a.region for a in group}) == 1
        assert len({a.name for a in group}) == 1
        categories = [a.category for a in group]
        yup_index = categories.index("yup") if "yup" in categories else 0
        assert "nope" not in categories[yup_index:]
        assert "yup" not in categories[:yup_index]


@mark.usefixtures("save_registries")
def test_compute_interactions(agents):
    @register_agent_interaction
    def dummy_interaction(a, b):
        assert all(agent.assets[0] is None for agent in (a, b))
        a.assets[0], b.assets[0] = b, a

    interactions = factory([("new_to_retro", "dummy_interaction")])
    interactions(agents)

    # Check unmatched agents
    unmatched = [agent for agent in agents if agent.assets[0] is None]
    assert len({(a.region, a.name) for a in unmatched}) == len(unmatched)

    # Check matched agents
    matched = [agent for agent in agents if agent.assets[0] is not None]
    assert len(new_to_retro_net(agents)) == 0 or matched
    assert all(agent.assets[0].assets[0] is agent for agent in matched)

    # Test that NoInteractionsFound is raised when all agents are 'nope' category
    nope_agents = [a for a in agents if a.category == "nope"]
    with pytest.raises(NoInteractionsFound):
        interactions(nope_agents)
