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
    "factory",
    "maximum_production",
    "demand_matched_production",
    "register_production",
    "PRODUCTION_SIGNATURE",
]
from typing import Any, Callable, Mapping, MutableMapping, Optional, Text, Union, cast

import numpy as np
import xarray as xr

from muse.registration import registrator

PRODUCTION_SIGNATURE = Callable[[xr.DataArray, xr.DataArray, xr.Dataset], xr.DataArray]
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

    def production_method(market, capacity, technologies) -> xr.DataArray:
        from muse.production import PRODUCTION_METHODS

        return PRODUCTION_METHODS[name](  # type: ignore
            market=market, capacity=capacity, technologies=technologies, **keywords
        )

    return production_method


@register_production(name=("max", "maximum"))
def maximum_production(
    market: xr.Dataset, capacity: xr.DataArray, technologies: xr.Dataset
) -> xr.DataArray:
    """Production when running at full capacity.

    *Full capacity* is limited by the utilitization factor. For more details, see
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
    costing: Text = "prices",
) -> xr.DataArray:
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
        prices = lcoe(
            market.prices, cast(xr.Dataset, broadcast_techs(technologies, capacity))
        )
    else:
        raise ValueError(f"Unknown costing option {costing}")

    return demand_matched_production(
        demand=market.consumption,
        prices=prices,
        capacity=capacity,
        technologies=technologies,
    )


@register_production
def costed_dispatch(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    costing: Text = "prices",
    cost_function: Union[Callable, Text] = "llcoe",
    year: Optional[int] = None,
) -> xr.DataArray:
    """Computes production from ranked assets.

    The assets are ranked according to their cost. Currently only llcoe and alcoe are
    allowed. The asset with least cost are allowed to service the demand first, up to
    the maximum production and above their minimum service.
    """

    from muse.quantities import (
        lifetime_levelized_cost_of_energy,
        annual_levelized_cost_of_energy,
        maximum_production,
    )
    from muse.utilities import broadcast_techs
    from muse.timeslices import convert_timeslice, QuantityType

    if callable(cost_function):
        cost_callable = cost_function
    elif cost_function.lower() == "llcoe":
        cost_callable = lifetime_levelized_cost_of_energy
    elif cost_function.lower() == "alcoe":
        cost_callable = annual_levelized_cost_of_energy
    else:
        raise ValueError(f"Unknown cost {cost_function}")

    if year is None:
        year = market.year.min()
    technodata = broadcast_techs(technologies, capacity)
    costs = cost_callable(market.prices.sel(region=technodata.region), technodata).rank(
        "asset"
    )
    maxprod = convert_timeslice(
        maximum_production(technodata, capacity.sel(year=year)),
        market.timeslice,
        QuantityType.EXTENSIVE,
    )
    minprod = getattr(technodata, "minimum_service_factor", 0) * maxprod
    commodity = (maxprod > 0).any([i for i in maxprod.dims if i != "commodity"])
    demand = market.consumption.sel(year=year, commodity=commodity).copy()

    visited = (maxprod <= 0).sel(commodity=commodity)
    constraints = (
        xr.Dataset(dict(maxprod=maxprod, minprod=minprod, costs=costs))
        .set_coords("costs")
        .sel(commodity=commodity)
    )
    production = xr.zeros_like(constraints.maxprod)
    for cost in sorted(set(constraints.costs.values.flatten())):
        condition = (constraints.costs == cost) & (constraints.maxprod > 0)
        assert ((~visited) & condition).sum() == condition.sum()
        visited |= condition
        cost_constraints = constraints.where(condition, 0)
        fullprod = cost_constraints.groupby("region").sum("asset")
        if (fullprod.maxprod <= demand + 1e-10).all():
            demand -= fullprod.maxprod
            production += cost_constraints.maxprod
        else:
            demand_prod = (
                broadcast_techs(demand, production)
                * (cost_constraints.maxprod / cost_constraints.maxprod.sum("asset"))
            ).where(condition, 0)
            current_prod = np.maximum(
                np.minimum(demand_prod, cost_constraints.maxprod),
                cost_constraints.minprod,
            )
            demand = np.maximum(
                (
                    demand
                    - xr.Dataset(dict(current_prod=current_prod))
                    .groupby("region")
                    .sum("asset")
                    .current_prod
                ),
                0,
            )
            production += current_prod

    assert visited.all()

    result = xr.zeros_like(maxprod)
    result[dict(commodity=commodity)] += production
    return result
