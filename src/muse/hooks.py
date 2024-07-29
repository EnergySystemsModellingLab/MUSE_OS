"""Pre and post hooks on agents."""

__all__ = [
    "register_initial_asset_transform",
    "register_final_asset_transform",
    "noop",
    "clean",
    "old_assets_only",
    "merge_assets",
    "new_assets_only",
    "housekeeping_factory",
    "asset_merge_factory",
]
from collections.abc import Mapping, MutableMapping
from typing import Callable, Union

from xarray import Dataset

from muse.agents import Agent
from muse.registration import registrator

INITIAL_ASSET_TRANSFORM: MutableMapping[str, Callable] = {}
""" Transform at the start of each step. """
FINAL_ASSET_TRANSFORM: MutableMapping[str, Callable] = {}
""" Transform at the end of each step, including new assets. """


def housekeeping_factory(settings: Union[str, Mapping] = "noop") -> Callable:
    """Returns a function for performing initial housekeeping.

    For instance, remove technologies with no capacity now or in the future.
    Available housekeeping functions should be registered with
    :py:func:`@register_initial_asset_transform<register_initial_asset_transform>`.
    """
    from muse.agents import AbstractAgent

    if isinstance(settings, str):
        name = settings
        params: Mapping = {}
    else:
        params = {k: v for k, v in settings.items() if k != "name"}
        name = settings["name"]

    transform = INITIAL_ASSET_TRANSFORM[name]

    def initial_assets_transform(agent: AbstractAgent, assets: Dataset) -> Dataset:
        return transform(agent, assets, **params)

    return initial_assets_transform


def asset_merge_factory(settings: Union[str, Mapping] = "new") -> Callable:
    """Returns a function for merging new investments into assets.

    Available merging functions should be registered with
    :py:func:`@register_final_asset_transform<register_final_asset_transform>`.
    """
    """Returns a function for performing initial housekeeping.

    For instance, remove technologies with no capacity now or in the future.
    Available housekeeping functions should be registered with
    :py:func:`@register_initial_asset_transform<register_initial_asset_transform>`.
    """
    if isinstance(settings, str):
        name = settings
        params: Mapping = {}
    else:
        params = {k: v for k, v in settings.items() if k != "name"}
        name = settings["name"]

    transform = FINAL_ASSET_TRANSFORM[name]

    def final_assets_transform(old_assets: Dataset, new_assets):
        return transform(old_assets, new_assets, **params)

    final_assets_transform.__name__ = name
    return final_assets_transform


@registrator(registry=INITIAL_ASSET_TRANSFORM, loglevel="info")
def register_initial_asset_transform(
    function: Callable[[Agent, Dataset], Dataset],
) -> Callable:
    """Decorator to register a function for cleaning or transforming assets.

    The transformation is applied at the start of each iteration. It any function which
    take an agent and assets as input and any number of keyword arguments, and returns
    the transformed assets. The agent should not be modified. It is only there to query
    the current year, the region, etc.
    """
    return function


@registrator(registry=FINAL_ASSET_TRANSFORM, loglevel="info")
def register_final_asset_transform(
    function: Callable[[Dataset, Dataset], Dataset],
) -> Callable:
    """Decorator to register a function to merge new investments into current assets.

    The transform is applied a the very end of the agent iteration. It can be any
    function which takes as input the current set of assets, the new assets, and any
    number of keyword arguments. The function must return a "merge" of the two assets.

    For instance, the new assets could completely replace the old assets
    (:py:func:`new_assets_only`), or they could be summed to the old assets
    (:py:func:`merge_assets`).
    """
    from functools import wraps

    @wraps(function)
    def decorated(old_assets: Dataset, new_assets: Dataset) -> Dataset:
        result = function(old_assets, new_assets)
        # missing values -> NaN -> integers become floats
        for variable in set(result.variables).intersection(old_assets.variables):
            result[variable] = result[variable].astype(old_assets[variable].dtype)
        result = result.drop_vars(set(result.coords) - set(old_assets.coords))
        return result

    return decorated


@register_initial_asset_transform(name="default")
def noop(agent: Agent, assets: Dataset) -> Dataset:
    """Return assets as they are."""
    return assets


@register_initial_asset_transform
def clean(agent: Agent, assets: Dataset) -> Dataset:
    """Removes empty assets."""
    from muse.utilities import clean_assets

    years = [agent.year, agent.forecast_year]
    return clean_assets(assets, years)


@register_final_asset_transform(name="new")
def new_assets_only(old_assets: Dataset, new_assets: Dataset) -> Dataset:
    """Returns newly invested assets and ignores old assets."""
    return new_assets


@register_final_asset_transform(name="old")
def old_assets_only(old_assets: Dataset, new_assets: Dataset) -> Dataset:
    """Returns old assets and ignores newly invested assets."""
    return old_assets


@register_final_asset_transform(name="merge")
def merge_assets(old_assets: Dataset, new_assets: Dataset) -> Dataset:
    """Adds new assets to old along asset dimension.

    New assets are assumed to be nonequivalent to any old_assets. Indeed,
    it is expected that the asset dimension does not have coordinates (i.e.
    it is a combination of coordinates, such as technology and installation
    year).

    After merging the new assets, quantities are back-filled along the year
    dimension. Further missing values (i.e. future years the old_assets
    did not take into account) are set to zero.
    """
    from logging import getLogger

    from muse.utilities import merge_assets

    assert "asset" not in old_assets

    if "asset" not in new_assets.dims and "replacement" in new_assets.dims:
        new_assets = new_assets.rename(replacement="asset")
    if len(new_assets.capacity) == 0:
        getLogger(__name__).critical("there are no new assets")
        return old_assets
    return merge_assets(old_assets, new_assets)
