"""Various ways and means to compute production.

Production is the amount of commodities produced by an asset. However, depending on the
context, it could be computed several ways. For  instace, it can be obtained straight
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
        market: Dataset, capacity: DataArray, technologies: Dataset, **kwargs
    ) -> DataArray:
        pass


Arguments:
    market: Market, including demand and prices.
    capacity: The capacity of each asset within a market.
    technologies: A dataset characterising the technologies of the same assets.
    **kwargs: Any number of keyword arguments

Returns:
    A `DataArray` with the amount produced for each good from each asset.
"""
__all__ = [
    "factory",
    "maximum_production",
    "demand_matched_production",
    "register_production",
    "PRODUCTION_SIGNATURE",
]
from typing import Any, Callable, Mapping, MutableMapping, Text, Union

from xarray import DataArray, Dataset

from muse.registration import registrator

PRODUCTION_SIGNATURE = Callable[[DataArray, DataArray, Dataset], DataArray]
"""Production signature."""

PRODUCTION_METHODS: MutableMapping[Text, PRODUCTION_SIGNATURE] = {}
"""Dictionary of production methods. """


@registrator(registry=PRODUCTION_METHODS, loglevel="info")
def register_production(function: PRODUCTION_SIGNATURE = None):
    """Decorator to register a function as a production method.

    .. seealso::

        :py:mod:`muse.production`
    """
    return function


def factory(
    settings: Union[Text, Mapping] = "maximum_production", **kwargs
) -> PRODUCTION_SIGNATURE:
    """Creates a production functor.

    This function's raison d'être is to convert the input from a TOML file into an
    actual functor usable within the model, i.e. it converts data into logic.

    Arguments:
        name: Registered production method to create. The name is resolved when the
            function returned by the factory is called. Hence, it could refer to a
            function yet to be registered when this factory method is called.
        **kwargs: any keyword argument the production method accepts.
    """
    if isinstance(settings, Text):
        name = settings
        keywords: MutableMapping[Text, Any] = dict()
    else:
        keywords = dict(**settings)
        name = keywords.pop("name")

    keywords.update(**kwargs)
    name = keywords.pop("name", name)

    def production_method(market, capacity, technologies) -> DataArray:
        from muse.production import PRODUCTION_METHODS

        return PRODUCTION_METHODS[name](  # type: ignore
            market=market, capacity=capacity, technologies=technologies, **keywords
        )

    return production_method


@register_production(name=("max", "maximum"))
def maximum_production(
    _: Dataset, capacity: DataArray, technologies: Dataset
) -> DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilitization factor. For more details, see
    :py:func:`muse.quantities.maximum_production`.
    """
    from muse.quantities import maximum_production

    return maximum_production(technologies, capacity)


@register_production(name=("share", "shares"))
def supply(market: Dataset, capacity: DataArray, technologies: Dataset) -> DataArray:
    """Service current demand equally from all assets.

    "Equally" means that equivalent technologies are used to the same percentage of
    their respective capacity.
    """
    from muse.quantities import supply

    return supply(capacity, market.consumption, technologies)


@register_production(name="match")
def demand_matched_production(
    market: Dataset,
    capacity: DataArray,
    technologies: Dataset,
    costing: Text = "prices",
) -> DataArray:
    """Production from matching demand via annual lcoe."""
    from muse.quantities import (
        demand_matched_production,
        gross_margin,
        annual_levelized_cost_of_energy as lcoe,
    )
    from muse.utilities import broadcast_techs

    if costing == "prices":
        prices = market.prices
    elif costing == "gross_margin":
        prices = gross_margin(technologies, capacity, market.prices)
    elif costing == "lcoe":
        prices = lcoe(market.prices, broadcast_techs(technologies, capacity))
    else:
        raise ValueError(f"Unknown costing option {costing}")

    return demand_matched_production(
        demand=market.consumption,
        prices=prices,
        capacity=capacity,
        technologies=technologies,
    )
