"""Various search-space filters.

Search-space filters return a modified matrix of booleans, with dimension
`(asset, replacement)`, where `asset` refer to technologies currently managed by
the agent, and `replacement` to all technologies the agent could consider, prior
to filtering.

Filters should be registered using the decorator :py:func:`register_filter`. The
registration makes it possible to call then from the agent by specifying the
`search_rule` attribute. The `search_rule` attribute is string or list of
strings specifying the filters to apply one after the other when considering the
search space.

Filters are not expected to modify any of their arguments. They should all
follow the same signature:

.. code-block:: Python

    @register_filter
    def search_space_filter(
        agent: Agent,
        search_space: xr.DataArray,
        technologies: xr.Dataset,
        market: xr.Dataset
    ) -> xr.DataArray:
        pass

Arguments:
    agent: the agent relevant to the search space. The filters may need to query
        the agent for parameters, e.g. the current year, the interpolation
        method, the tolerance, etc.
    search_space: the current search space.
    technologies: A data set characterising the technologies from which the
        agent can draw assets.
    market: Market variables, such as prices or current capacity and retirement
        profile.

Returns:
    A new search space with the same data-type as the input search-space, but
    with potentially different values.


In practice, an initial search space is created by calling a function with the signature
given below, and registered with :py:func:`register_initializer`. The initializer
function returns a search space which is passed on to a chain of filters, as done in the
:py:func:`factory` function.

Functions creating initial search spaces should have the following signature:

.. code-block:: Python

    @register_initializer
    def search_space_initializer(
        agent: Agent,
        demand: xr.DataArray,
        technologies: xr.Dataset,
        market: xr.Dataset
    ) -> xr.DataArray:
        pass

Arguments:
    agent: the agent relevant to the search space. The filters may need to query
        the agent for parameters, e.g. the current year, the interpolation
        method, the tolerance, etc.
    demand: share of the demand per existing reference technology (e.g.
        assets).
    technologies: A data set characterising the technologies from which the
        agent can draw assets.
    market: Market variables, such as prices or current capacity and retirement
        profile.

Returns:
    An initial search space
"""

__all__ = [
    "factory",
    "register_filter",
    "register_initializer",
    "identity",
    "reduce_asset",
    "similar_technology",
    "same_enduse",
    "same_fuels",
    "currently_existing_tech",
    "currently_referenced_tech",
    "maturity",
    "compress",
    "with_asset_technology",
    "initialize_from_technologies",
]

from collections.abc import Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import numpy as np
import xarray as xr

from muse.agents import Agent
from muse.registration import registrator

SSF_SIGNATURE = Callable[[Agent, xr.DataArray, xr.Dataset, xr.Dataset], xr.DataArray]
""" Search space filter signature """

SEARCH_SPACE_FILTERS: MutableMapping[str, SSF_SIGNATURE] = {}
"""Filters for selecting technology search spaces."""


SSI_SIGNATURE = Callable[[Agent, xr.DataArray, xr.Dataset, xr.Dataset], xr.DataArray]
""" Search space initializer signature """

SEARCH_SPACE_INITIALIZERS: MutableMapping[str, SSI_SIGNATURE] = {}
"""Functions to create an initial search-space."""


@registrator(registry=SEARCH_SPACE_FILTERS, loglevel="info")
def register_filter(function: SSF_SIGNATURE) -> Callable:
    """Decorator to register a function as a filter.

    Registers a function as a filter so that it can be applied easily
    when constraining the technology search-space.

    The name that the function is registered with defaults to the function name.
    However, it can also be specified explicitly as a *keyword* argument. In any
    case, it must be unique amongst all search-space filters.
    """
    from functools import wraps

    @wraps(function)
    def decorated(
        agent: Agent, search_space: xr.DataArray, *args, **kwargs
    ) -> xr.DataArray:
        result = function(agent, search_space, *args, **kwargs)  # type: ignore
        if isinstance(result, xr.DataArray):
            result.name = search_space.name
        return result

    return decorated


@registrator(
    registry=SEARCH_SPACE_INITIALIZERS, logname="initial search-space", loglevel="info"
)
def register_initializer(function: SSI_SIGNATURE) -> Callable:
    """Decorator to register a function as a search-space initializer."""
    from functools import wraps

    @wraps(function)
    def decorated(agent: Agent, *args, **kwargs) -> xr.DataArray:
        result = function(agent, *args, **kwargs)  # type: ignore
        if isinstance(result, xr.DataArray):
            result.name = "search_space"
        return result

    return decorated


def factory(
    settings: Optional[Union[str, Mapping, Sequence[Union[str, Mapping]]]] = None,
    separator: str = "->",
):
    """Creates filters from input TOML data.

    The input data is standardized to a list of dictionaries where each dictionary
    contains at least one member, "name".

    The first dictionary specifies the initial function which creates the search space
    from the demand share, the market, and the dataset describing technologies in the
    sectors.

    The next entries are applied in turn and transform the search space in some way.
    In other words the process is more or less:

    .. code-block:: Python

        search_space = initial_filter(
            agent, demand, technologies=technologies, market=market
        )
        for afilter in filters:
            search_space = afilter(
                agent, search_space, technologies=technologies, market=market
            )
        return search_space

    ``initial_filter`` is simply first filter given on input, if that filter is
    registered with :py:func:`register_initializer`. Otherwise,
    :py:func:`initialize_from_technologies` is automatically inserted.
    """
    from functools import partial

    if settings is None:
        parameters: Sequence[Mapping[str, Any]] = []
    elif isinstance(settings, Mapping):
        parameters = [settings]
    elif isinstance(settings, str):
        parameters = [{"name": name.strip()} for name in settings.split(separator)]
    else:
        parameters = [
            {"name": item} if isinstance(item, str) else item for item in settings
        ]
    if len(parameters) == 0 or parameters[0]["name"] not in SEARCH_SPACE_INITIALIZERS:
        initial_settings: Mapping[str, str] = {"name": "initialize_from_technologies"}
    else:
        initial_settings, parameters = cast(Mapping, parameters[0]), parameters[1:]

    functions = [
        partial(
            SEARCH_SPACE_INITIALIZERS[initial_settings["name"]],
            **{k: v for k, v in initial_settings.items() if k != "name"},
        ),
        *(
            partial(
                SEARCH_SPACE_FILTERS[setting["name"]],
                **{k: v for k, v in setting.items() if k != "name"},
            )
            for setting in parameters
        ),
    ]

    def filters(agent: Agent, demand: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        """Applies a series of filter to determine the search space."""
        result = functions[0](agent, demand, *args, **kwargs)
        for function in functions[1:]:
            result = function(agent, result, *args, **kwargs)
        return result

    return filters


@register_filter
def same_enduse(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    *args,
    **kwargs,
) -> xr.DataArray:
    """Only allow for technologies with at least the same end-use."""
    from muse.commodities import is_enduse

    tech_enduses = agent.filter_input(
        technologies.fixed_outputs,
        year=agent.year,
        commodity=is_enduse(technologies.comm_usage),
    )
    tech_enduses = (tech_enduses > 0).astype(int).rename(technology="replacement")
    asset_enduses = tech_enduses.sel(replacement=search_space.asset)
    return search_space & ((tech_enduses - asset_enduses) >= 0).all("commodity")


@register_filter(name="all")
def identity(agent: Agent, search_space: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    """Returns search space as given."""
    return search_space


@register_filter(name="similar")
def similar_technology(
    agent: Agent, search_space: xr.DataArray, technologies: xr.Dataset, *args, **kwargs
):
    """Filters technologies with the same type."""
    tech_type = agent.filter_input(technologies.tech_type)
    asset_types = tech_type.sel(technology=search_space.asset)
    tech_types = tech_type.sel(technology=search_space.replacement)
    return search_space & (asset_types == tech_types)


@register_filter(name="fueltype")
def same_fuels(
    agent: Agent, search_space: xr.DataArray, technologies: xr.Dataset, *args, **kwargs
):
    """Filters technologies with the same fuel type."""
    fuel = agent.filter_input(technologies.fuel)
    asset_fuel = fuel.sel(technology=search_space.asset)
    tech_fuel = fuel.sel(technology=search_space.replacement)
    return search_space & (asset_fuel == tech_fuel)


@register_filter(name="existing")
def currently_existing_tech(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
) -> xr.DataArray:
    """Only consider technologies that currently exist in the market.

    This filter only allows technologies that exists in the market and have non- zero
    capacity in the current year. See `currently_referenced_tech` for a similar filter
    that does not check the capacity.
    """
    capacity = agent.filter_input(market.capacity, year=agent.year).rename(
        technology="replacement"
    )
    result = search_space & search_space.replacement.isin(capacity.replacement)
    both = (capacity.replacement.isin(search_space.replacement)).replacement
    result.loc[{"replacement": both.values}] &= capacity.sel(
        replacement=both
    ) > getattr(agent, "tolerance", 1e-8)
    return result


@register_filter
def currently_referenced_tech(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
) -> xr.DataArray:
    """Only consider technologies that are currently referenced in the market.

    This filter will allow any technology that exists in the market, even if it
    currently sits at zero capacity (unlike `currently_existing_tech` which requires
    non-zero capacity in the current year).
    """
    capacity = agent.filter_input(market.capacity, year=agent.year).rename(
        technology="replacement"
    )
    return search_space & search_space.replacement.isin(capacity.replacement)


@register_filter
def maturity(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    enduse_label: str = "service",
    **kwargs,
) -> xr.DataArray:
    """Only allows technologies that have achieve a given market share.

    Specifically, the market share refers to the capacity for each end- use.
    """
    capacity = agent.filter_input(market.capacity, year=agent.year)
    total_capacity = capacity.sum("technology")
    enduse_market_share = agent.maturity_threshold * total_capacity
    condition = enduse_market_share <= capacity
    techs = (
        condition.technology.where(condition, drop=True).drop_vars("technology").values
    )

    # Generate a boolean mask where 'True' corresponds to entries in
    # 'search_space.replacement' that are in 'techs'
    mask = search_space.replacement.isin(techs)

    # Apply this mask to 'search_space', turning all fields where the condition is not
    # met to False
    replacement = search_space.where(mask, False)

    return search_space & replacement


@register_filter
def spend_limit(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    enduse_label: str = "service",
    **kwargs,
) -> xr.DataArray:
    """Only allows technologies that have achieve a given market share.

    Specifically, the market share refers to the capacity for each end- use.
    """
    spend_limit = agent.spend_limit
    unit_capex = agent.filter_input(technologies.cap_par, year=agent.year)
    condition = (unit_capex <= spend_limit).rename("spend_limit")
    techs = (
        condition.technology.where(condition, drop=True).drop_vars("technology").values
    )

    # Generate a boolean mask where 'True' corresponds to entries in
    # 'search_space.replacement' that are in 'techs'
    mask = search_space.replacement.isin(techs)

    # Apply this mask to 'search_space', turning all fields where the condition is not
    # met to False
    replacement = search_space.where(mask, False)

    return search_space & replacement


@register_filter
def compress(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    **kwargs,
) -> xr.DataArray:
    """Compress search space to include only potential technologies.

    This operation reduces the *size* of the search space along the
    `replacement` dimension, such that are left only technologies that
    will be considered as replacement for at least by one asset. Unlike
    most filters, it does not change the data, but rather changes how
    the data is represented. In other words, this is mostly an
    *optimization* for later steps, to avoid unnecessary computations.
    """
    if len(search_space.dims) == 1 and search_space.dims[0] == "replacement":
        condition = search_space
    else:
        condition = search_space.any("asset")
    return search_space.sel(replacement=condition)


@register_filter(name=["reduce_asset", "reduce_assets"])
def reduce_asset(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    **kwargs,
) -> xr.DataArray:
    """Reduce over assets."""
    return search_space.any("asset") if "asset" in search_space.dims else search_space


@register_filter
def with_asset_technology(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    **kwargs,
) -> xr.DataArray:
    """Search space *also* contains its asset technology for each asset."""
    return search_space | (search_space.asset == search_space.replacement)


@register_initializer(name="from_techs")
def initialize_from_technologies(
    agent: Agent, demand: xr.DataArray, technologies: xr.Dataset, *args, **kwargs
):
    """Initialize a search space from existing technologies."""
    coords = (
        ("asset", demand.asset.values),
        ("replacement", technologies.technology.values),
    )
    return xr.DataArray(
        np.ones(tuple(len(u[1]) for u in coords), dtype=bool),
        coords=coords,
        dims=[u[0] for u in coords],
        name="search_space",
    )


@register_initializer(name="from_assets")
def initialize_from_assets(
    agent: Agent,
    demand: xr.DataArray,
    technologies: xr.Dataset,
    *args,
    coords: Sequence[str] = ("region", "technology"),
    **kwargs,
):
    """Initialize a search space from existing technologies."""
    from muse.utilities import reduce_assets

    replacement = xr.DataArray(
        np.ones_like(technologies.technology, dtype=bool),
        coords={"replacement": technologies.technology.values},
        dims="replacement",
    )
    if "asset" not in agent.assets.dims or len(agent.assets.asset) == 0:
        return replacement

    assets = (
        xr.ones_like(reduce_assets(agent.assets.asset, coords=coords), dtype=bool)
        .rename(technology="asset")
        .set_index()
    )
    return (assets * replacement).transpose("asset", "replacement")
