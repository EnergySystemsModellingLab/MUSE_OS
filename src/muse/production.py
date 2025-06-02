"""Various ways and means to compute production.

Production is the amount of commodities produced by an asset. However, depending on the
context, it could be computed several ways. For  instance, it can be obtained straight
from the capacity of the asset. Or it can be obtained by matching for the same
commodities with a set of assets.

Production methods can be registered via the :py:func:`@register_production
<register_production>` production decorator.  Registering a function makes the function
accessible from MUSE's input file. Production methods are not expected to modify their
arguments. Furthermore they should conform the
following signatures:

.. code-block:: python

    @register_production
    def production(
        market: xr.Dataset, capacity: xr.DataArray, technologies: xr.Dataset, **kwargs
    ) -> xr.DataArray:
        pass


Arguments:
    market: Market, including demand and prices.
    capacity: The capacity of each asset within a market.
    technologies: A dataset characterising the technologies of the same assets.
    **kwargs: Any number of keyword arguments

Returns:
    A `xr.DataArray` with the amount produced for each good from each asset.
"""

from __future__ import annotations

__all__ = [
    "PRODUCTION_SIGNATURE",
    "factory",
    "maximum_production",
    "register_production",
    "supply",
]

from collections.abc import MutableMapping
from typing import Callable

import xarray as xr

from muse.registration import registrator

"""Production signature."""
PRODUCTION_SIGNATURE = Callable[
    [xr.Dataset, xr.DataArray, xr.Dataset, str], xr.DataArray
]

"""Dictionary of production methods. """
PRODUCTION_METHODS: MutableMapping[str, PRODUCTION_SIGNATURE] = {}


@registrator(registry=PRODUCTION_METHODS, loglevel="info")
def register_production(function: PRODUCTION_SIGNATURE):
    """Decorator to register a function as a production method.

    .. seealso::

        :py:mod:`muse.production`
    """
    return function


def factory(name) -> PRODUCTION_SIGNATURE:
    from muse.production import PRODUCTION_METHODS

    return PRODUCTION_METHODS[name]


@register_production(name=("max", "maximum"))
def maximum_production(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilization factor. For more details, see
    :py:func:`muse.quantities.maximum_production`.
    """
    from muse.quantities import maximum_production

    return maximum_production(technologies, capacity, timeslice_level)


@register_production(name=("share", "shares"))
def supply(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: str | None = None,
) -> xr.DataArray:
    """Service current demand equally from all assets.

    "Equally" means that equivalent technologies are used to the same percentage of
    their respective capacity.
    """
    from muse.quantities import supply

    return supply(
        capacity, market.consumption, technologies, timeslice_level=timeslice_level
    )
