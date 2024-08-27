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
    "demand_matched_production",
    "factory",
    "maximum_production",
    "register_production",
    "supply",
    "PRODUCTION_SIGNATURE",
]
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, Union, cast

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
    market: xr.Dataset, capacity: xr.DataArray, technologies: xr.Dataset
) -> xr.DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilization factor. For more details, see
    :py:func:`muse.quantities.maximum_production`.
    """
    from muse.quantities import maximum_production

    return maximum_production(technologies, capacity)


@register_production(name=("share", "shares"))
def supply(
    market: xr.Dataset, capacity: xr.DataArray, technologies: xr.Dataset
) -> xr.DataArray:
    """Service current demand equally from all assets.

    "Equally" means that equivalent technologies are used to the same percentage of
    their respective capacity.
    """
    from muse.quantities import supply

    return supply(capacity, market.consumption, technologies)


@register_production(name="match")
def demand_matched_production(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    costs: str = "prices",
) -> xr.DataArray:
    """Production from matching demand via annual lcoe."""
    from muse.costs import annual_levelized_cost_of_energy as lcoe
    from muse.quantities import demand_matched_production, gross_margin
    from muse.utilities import broadcast_techs

    if costs == "prices":
        prices = market.prices
    elif costs == "gross_margin":
        prices = gross_margin(technologies, capacity, market.prices)
    elif costs == "lcoe":
        prices = lcoe(
            market.prices, cast(xr.Dataset, broadcast_techs(technologies, capacity))
        )
    else:
        raise ValueError(f"Unknown costs option {costs}")

    return demand_matched_production(
        demand=market.consumption,
        prices=prices,
        capacity=capacity,
        technologies=technologies,
    )


@register_production(name="costed")
def costed_production(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    costs: Union[xr.DataArray, Callable, str] = "alcoe",
    with_minimum_service: bool = True,
    with_emission: bool = True,
) -> xr.DataArray:
    """Computes production from ranked assets.

    The assets are ranked according to their cost. The cost can be provided as an
    xarray, a callable creating an xarray, or as "alcoe". The asset with least cost are
    allowed to service the demand first, up to the maximum production. By default, the
    minimum service is applied first.
    """
    from muse.commodities import CommodityUsage, check_usage, is_pollutant
    from muse.costs import annual_levelized_cost_of_energy
    from muse.quantities import (
        costed_production,
        emission,
    )
    from muse.utilities import broadcast_techs

    if isinstance(costs, str) and costs.lower() == "alcoe":
        costs = annual_levelized_cost_of_energy
    elif isinstance(costs, str):
        raise ValueError(f"Unknown cost {costs}")
    if callable(costs):
        technodata = cast(xr.Dataset, broadcast_techs(technologies, capacity))
        costs = costs(
            prices=market.prices.sel(region=technodata.region), technologies=technodata
        )
    else:
        costs = costs
    assert isinstance(costs, xr.DataArray)

    production = costed_production(
        market.consumption,
        costs,
        capacity,
        technologies,
        with_minimum_service=with_minimum_service,
    )
    # add production of environmental pollutants
    if with_emission:
        env = is_pollutant(technologies.comm_usage)
        production[dict(commodity=env)] = emission(
            production, technologies.fixed_outputs
        ).transpose(*production.dims)
        production[
            dict(
                commodity=~check_usage(technologies.comm_usage, CommodityUsage.PRODUCT)
            )
        ] = 0
    return production
