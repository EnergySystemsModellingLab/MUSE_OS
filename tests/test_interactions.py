"""Test agent interactions."""

import pytest
from pytest import fixture, mark


@fixture
def agents():
    from collections import namedtuple

    from numpy.random import choice

    Agent = namedtuple("Agent", ["region", "name", "category", "assets"])
    regions = ["Area52", "Bermuda Triangle", "City of London"]
    names = ["John", "Joe", "Jill"]
    categories = ["yup", "nope"]
    results = {
        (region, name, cat)
        for region, name, cat in zip(
            *(choice(data, 40) for data in [regions, names, categories])
        )
    }
    return [Agent(*u, [None]) for u in results]


def test_groupby(agents):
    from itertools import chain

    from muse.interactions import agents_groupby

    grouped = agents_groupby(agents, ("category", "name"))
    assert sum(len(u) for u in grouped.values()) == len(agents)
    assert set(id(u) for u in chain(*grouped.values())) == set(id(u) for u in agents)
    assert set(grouped.keys()) == set((u.category, u.name) for u in agents)
    for (category, name), group_agents in grouped.items():
        assert all(agent.category == category for agent in group_agents)
        assert all(agent.name == name for agent in group_agents)


def test_new_to_retro_net(agents):
    from itertools import chain

    from muse.interactions import new_to_retro_net

    net = new_to_retro_net(agents, "nope")
    assert sum(len(u) for u in net) <= len(agents)
    assert set(id(u) for u in chain(*net)).issubset(id(u) for u in agents)
    assert all(len(list(u)) == 2 for u in net)
    for agents in net:
        assert len(set(u.region for u in agents)) == 1
        assert len(set(u.name for u in agents)) == 1
        categories = [u.category for u in agents]
        i = categories.index("yup") if "yup" in categories else 0
        assert "nope" not in categories[i:]
        assert "yup" not in categories[:i]


@mark.usefixtures("save_registries")
def test_compute_interactions(agents):
    from muse.errors import NoInteractionsFound
    from muse.interactions import factory, new_to_retro_net, register_agent_interaction

    @register_agent_interaction
    def dummy_interaction(a, b):
        assert a.assets[0] is None
        assert b.assets[0] is None
        a.assets[0] = b
        b.assets[0] = a

    interactions = factory([("new_to_retro", "dummy_interaction")])
    interactions(agents)

    are_none = [agent for agent in agents if agent.assets[0] is None]
    assert len({(u.region, u.name) for u in are_none}) == len(are_none)
    not_none = [agent for agent in agents if agent.assets[0] is not None]
    assert len(new_to_retro_net(agents)) == 0 or len(not_none) != 0
    for agent in not_none:
        assert agent.assets[0].assets[0] is agent

    agents2 = [a for a in agents if a.category == "nope"]
    with pytest.raises(NoInteractionsFound):
        interactions(agents2)
