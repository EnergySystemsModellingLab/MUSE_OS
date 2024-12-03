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

__all__ = [
    "PRODUCTION_SIGNATURE",
    "factory",
    "maximum_production",
    "register_production",
    "supply",
]
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, Optional, Union, cast

import xarray as xr

from muse.registration import registrator

PRODUCTION_SIGNATURE = Callable[[xr.DataArray, xr.DataArray, xr.Dataset], xr.DataArray]
"""Production signature."""

PRODUCTION_METHODS: MutableMapping[str, PRODUCTION_SIGNATURE] = {}
"""Dictionary of production methods. """


@registrator(registry=PRODUCTION_METHODS, loglevel="info")
def register_production(function: PRODUCTION_SIGNATURE = None):
    """Decorator to register a function as a production method.

    .. seealso::

        :py:mod:`muse.production`
    """
    return function


def factory(
    settings: Union[str, Mapping] = "maximum_production", **kwargs
) -> PRODUCTION_SIGNATURE:
    """Creates a production functor.

    This function's raison d'être is to convert the input from a TOML file into an
    actual functor usable within the model, i.e. it converts data into logic.

    Arguments:
        settings: Registered production method to create. The name is resolved when the
            function returned by the factory is called. Hence, it could refer to a
            function yet to be registered when this factory method is called.
        **kwargs: any keyword argument the production method accepts.
    """
    from functools import partial

    from muse.production import PRODUCTION_METHODS

    if isinstance(settings, str):
        name = settings
        keywords: MutableMapping[str, Any] = dict()
    else:
        keywords = dict(**settings)
        name = keywords.pop("name")

    keywords.update(**kwargs)
    name = keywords.pop("name", name)

    method = PRODUCTION_METHODS[name]
    return cast(
        PRODUCTION_SIGNATURE, method if not keywords else partial(method, **keywords)
    )


@register_production(name=("max", "maximum"))
def maximum_production(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    timeslice_level: Optional[str] = None,
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
    timeslice_level: Optional[str] = None,
) -> xr.DataArray:
    """Service current demand equally from all assets.

    "Equally" means that equivalent technologies are used to the same percentage of
    their respective capacity.
    """
    from muse.quantities import supply

    return supply(
        capacity, market.consumption, technologies, timeslice_level=timeslice_level
    )
