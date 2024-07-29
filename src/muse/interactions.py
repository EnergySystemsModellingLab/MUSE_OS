"""Modes of interactions between agents.

Interactions between agents are modelled via two orthogonal concepts:

- a *net* is a set of agents which interact in some way
- an *interaction* proper is a function that takes a net and actually performs the
  interaction.

Hence, there are two registrators in this this module,
:py:func:`register_interaction_net`, and :py:func:`register_agent_interaction`. The
first registers functions that take the full set of agents as input and returns a
sequence of nets. It is expected each net of the sequence will be applied the same
interaction. The second registrator registers the interaction proper: it takes agents as
arguments and returns nothing. It is expected to modify the agents in-place.
"""

__all__ = [
    "register_interaction_net",
    "register_agent_interaction",
    "factory",
    "new_to_retro_net",
    "transfer_assets",
]

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

from muse.agents import AbstractAgent, Agent
from muse.errors import NoInteractionsFound
from muse.registration import registrator

AGENT_INTERACTIONS: Mapping[str, Callable] = {}
"""All interaction between a single agents and its interactees."""
INTERACTION_NETS: Mapping[str, Callable] = {}
"""All functions to computes lists of agents interaction with each other."""

INTERACTION_NET = Sequence[Sequence[Agent]]
"""Defines interaction sets between agents.

An interaction set is a mapping an agent to the agents with which it interacts. For
instance, it could, for each region, map a new agent with the retro agent with the same
name as itself.
"""
INTERACTION_NET_SIGNATURE = Callable[[Sequence[Agent]], INTERACTION_NET]
"""Signature of the interaction net function.

The interaction net functions should create a list of lists of interacting agents, given
as input the list of all agents.
"""
AGENT_INTERACTION_SIGNATURE = Callable[[Agent, Agent], None]
"""Signature of a single agent to agent(s) interaction.

An interaction function takes as argument the agents that are to interact. It can modify
the parameters and assets of these agents.
"""


@registrator(registry=INTERACTION_NETS, loglevel="debug")
def register_interaction_net(function: INTERACTION_NET_SIGNATURE):
    """Decorator to register a function computing interaction nets.

    An interaction net function takes as input the list of all agents and
    returns the list of all interactions, where an interaction is a list of at
    least two interacting agents.

    An interactiont-net function also takes as argument a sector object.
    This object should not be modified in any way. But it can be queried for
    parameters, if the specific interaction-net function requires it.
    """
    return function


@registrator(registry=AGENT_INTERACTIONS, loglevel="debug")
def register_agent_interaction(function: AGENT_INTERACTION_SIGNATURE):
    """Decorator to register an agent to agent(s) interaction function.

    An agent interaction function takes at least two agents and makes them
    interact in some way.

    An agent interaction function also takes as argument a sector object.
    This object should not be modified in any way. But it can be queried for
    parameters, if the specific agent interaction function requires it. This is
    most likely the same configuration object passed on to the interaction net
    function.
    """
    return function


def factory(
    inputs: Optional[Sequence[Union[Mapping, tuple[str, str]]]] = None,
) -> Callable[[Sequence[AbstractAgent]], None]:
    """Creates an interaction functor."""
    if inputs is None:
        inputs = tuple()

    interactions = []
    for params in inputs:
        if isinstance(params, Mapping):
            net_params = params["net"]
            action_params = params["interaction"]
        else:
            net_params, action_params = params

        if isinstance(net_params, str):
            net_params = {"name": net_params}

        if isinstance(action_params, str):
            action_params = {"name": action_params}

        net = INTERACTION_NETS[net_params["name"]]
        interaction = AGENT_INTERACTIONS[action_params["name"]]

        interactions.append(((net, net_params), (interaction, action_params)))

    def compute_interactions(agents: Sequence[AbstractAgent]) -> None:
        """Applies interaction net and agent interaction functions.

        If a network is found with no interactions, an error is raised.
        """
        from logging import getLogger

        for (net, net_params), (interaction, interaction_params) in interactions:
            nparams = {k: v for k, v in net_params.items() if k != "name"}
            sets = net(agents, **nparams)

            if len(sets) == 0:
                raise NoInteractionsFound

            getLogger(__name__).info(
                f"Net {net_params['name']} of {len(sets)} interactions interacting "
                f"via {interaction_params['name']}"
            )

            iparams = {k: v for k, v in interaction_params.items() if k != "name"}
            for agents in sets:
                interaction(*agents, **iparams)

    return compute_interactions


def agents_groupby(
    agents: Sequence[Agent], attributes: Sequence[str]
) -> Mapping[tuple, list[Agent]]:
    attr_list = [tuple(getattr(agent, attr) for attr in attributes) for agent in agents]
    result: Mapping[tuple, list[Agent]] = {tuple(n): [] for n in attr_list}
    for attrs, agent in zip(attr_list, agents):
        result[attrs].append(agent)
    return result


@register_interaction_net(name=["default", "new_to_retro"])
def new_to_retro_net(
    agents: Sequence[Agent], first_category: str = "newcapa"
) -> INTERACTION_NET:
    """Interactions between new and retrofit agents."""
    groups = agents_groupby(agents, ("region", "name"))

    def comparison(a, b):
        if a is None:
            return b is None
        if a == b:
            return True
        if hasattr(a, "lower") and hasattr(b, "lower"):
            return a.lower() == b.lower()
        return False

    return [
        [a for a in group if comparison(a.category, first_category)]
        + [a for a in group if not comparison(a.category, first_category)]
        for group in groups.values()
        if len(group) == 2
    ]


@register_agent_interaction(name="transfer")
def transfer_assets(from_: Agent, to_: Agent) -> None:
    """Transfer assets from first agent to second agent."""
    from xarray import zeros_like

    from muse.utilities import merge_assets

    to_.assets = merge_assets(to_.assets, from_.assets)
    from_.assets = zeros_like(to_.assets)
