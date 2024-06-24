"""Output cached quantities.

Functions that output the state of diverse quantities at intermediate steps of the
calculation.

The core of the method is the OutputCache class that initiated by the MCA with input
parameters defined in the TOML file, much like the existing `output` options but in a
`outputs_cache` list, enables listening for data to be cached and, after
each period, saved into disk via the `consolidate_cache` method.

Anywhere in the code, you can write:

.. code-block:: python

    cache_quantity(quantity_name=some_data)

If the quantity has been set as something to cache, the data will be stored and,
eventually, save to disk after - possibly - aggregating the data and removing those
entries corresponding to non-convergent investment attempts. This process of cleaning
and aggregation is quantity specific.

See documentation for the :py:func:`muse.outputs.cache.cache_quantity` function as well
as how to setup the toml input file to cache quantities. Users can customize and create
further output quantities by registering with MUSE via
:py:func:`muse.outputs.cache.register_cached_quantity`.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, MutableMapping, Sequence
from functools import reduce
from operator import attrgetter
from typing import (
    Callable,
    Union,
)

import pandas as pd
import xarray as xr
from pubsub import pub

from muse.registration import registrator
from muse.sectors import AbstractSector

OUTPUT_QUANTITY_SIGNATURE = Callable[
    [list[xr.DataArray]], Union[xr.DataArray, pd.DataFrame]
]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: MutableMapping[str, OUTPUT_QUANTITY_SIGNATURE] = {}
"""Quantity for post-simulation analysis."""

CACHE_TOPIC_CHANNEL = "cache_quantity"
"""Topic channel to use with the pubsub messaging system."""


@registrator(registry=OUTPUT_QUANTITIES)
def register_cached_quantity(function: OUTPUT_QUANTITY_SIGNATURE) -> Callable:
    """Registers a function to compute an output quantity."""
    from functools import wraps

    @wraps(function)
    def decorated(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, (pd.DataFrame, xr.DataArray)):
            result.name = function.__name__
        return result

    return decorated


def cache_quantity(
    function: Callable | None = None,
    quantity: str | Sequence[str] | None = None,
    **kwargs: xr.DataArray,
) -> Callable | None:
    """Cache one or more quantities to be post-processed later on.

    This function can be used as a decorator, in which case the quantity input argument
    must be set, or directly called with any number of keyword arguments. In the former
    case, the matching between quantities and values to cached is done by the function
    'match_quantities'. When used in combination with other decorators, care must be
    taken to decide the order in which they are applied to make sure the appropriate
    output is cached.

    Note that if the quantity has NOT been selected to be cached when configuring the
    MUSE simulation, it will be silently ignored if present as an input to this
    function.

    Example:
        As a decorator, the quantity argument must be set:

        >>> @cache_quantity(quantity="capacity")
        ... def some_calculation():
        ...     return xr.DataArray()

        If returning a sequence of DataArrays, the number of quantities to record must
        be the same as the number of arrays. They are paired in the same order they are
        given and the 'name' attribute of the arrays, if present, is ignored.

        >>> @cache_quantity(quantity=["capacity", "production"])
        ... def other_calculation():
        ...     return xr.DataArray(), xr.DataArray()

        For a finer control of what is cached when there is a complex output, combine
        the DataArrays in a Dataset. In this case, the 'quantity' input argument can be
        either a string or a sequence of strings to record multiple variables in the
        Dataset.

        >>> @cache_quantity(quantity=["capacity", "production"])
        ... def and_another_one():
        ...     return xr.Dataset(
        ...         {
        ...             "not cached": xr.DataArray(),
        ...             "capacity": xr.DataArray(),
        ...             "production": xr.DataArray(),
        ...         }
        ...     )

        When this function is called directly and not used as a decorator, simply
        provide the name of the quantities and the DataArray to record as keyword
        arguments:

        >>> cache_quantity(capacity=xr.DataArray(), production=xr.DataArray())

    Args:
        function (Optional[Callable]): The decorated function, if any. Its output must
            be a DataArray, a sequence of DataArray or a Dataset. See 'match_quantities'
        quantity (Union[str, List[str], None]): The name of the quantities to record.
        **kwargs (xr.DataArray): Keyword arguments of the form
            'quantity_name=quantity_value'.

    Raises:
        ValueError: If a function input argument is provided at the same time than
        keyword arguments.

    Return:
        (Optional[Callable]) The decorated function (or a dummy function if called
        directly).
    """
    from functools import wraps

    # When not used as a decorator
    if len(kwargs) > 0:
        if function is not None:
            raise ValueError(
                "If keyword arguments are provided, then 'function' must be None"
            )
        pub.sendMessage(CACHE_TOPIC_CHANNEL, data=kwargs)
        return None

    # When used as a decorator
    if function is None:
        return lambda x: cache_quantity(x, quantity=quantity)

    if quantity is None:
        raise ValueError(
            "When 'cache_quantity' is used as a decorator the 'quantity' input argument"
            " must be a string or sequence of strings. None found."
        )

    @wraps(function)
    def decorated(*args, **kwargs):
        result = function(*args, **kwargs)
        cache_quantity(**match_quantities(quantity, result))
        return result

    return decorated


def match_quantities(
    quantity: str | Sequence[str],
    data: xr.DataArray | xr.Dataset | Sequence[xr.DataArray],
) -> Mapping[str, xr.DataArray]:
    """Matches the quantities with the corresponding data.

    The possible name attribute in the DataArrays is ignored.

    Args:
        quantity (Union[str, Sequence[str]]): The name(s) of the quantity(ies) to cache.
        data (Union[xr.DataArray, xr.Dataset, Sequence[xr.DataArray]]): The structure
            containing the data to cache.

    Raises:
        TypeError: If there is an invalid combination of input argument types.
        ValueError: If the number of quantities does not match the length of the data.
        KeyError: If the required quantities do not exist as variables in the dataset.

    Returns:
        (Mapping[str, xr.DataArray]) A dictionary matching the quantity names with the
        corresponding data.
    """
    if isinstance(quantity, str) and isinstance(data, xr.DataArray):
        return {quantity: data}

    elif isinstance(quantity, str) and isinstance(data, xr.Dataset):
        return {quantity: data[quantity]}

    elif isinstance(quantity, Sequence) and isinstance(data, xr.Dataset):
        return {q: data[q] for q in quantity}

    elif isinstance(quantity, Sequence) and isinstance(data, Sequence):
        if len(quantity) != len(data):
            msg = f"{len(quantity)} != {len(data)}"
            raise ValueError(
                f"The number of quantities does not match the length of the data {msg}."
            )
        return {q: v for q, v in zip(quantity, data)}

    else:
        msg = f"{type(quantity)} and {type(data)}"
        raise TypeError(f"Invalid combination of input argument types {msg}")


class OutputCache:
    """Creates outputs functions for post-mortem analysis of cached quantities.

    Each parameter is a dictionary containing the following:

    - quantity (mandatory): name of the quantity to output. Mandatory.
    - sink (optional): name of the storage procedure, e.g. the file format
      or database format. When it cannot be guessed from `filename`, it defaults to
      "csv".
    - filename (optional): path to a directory or a file where to store the quantity. In
      the latter case, if sink is not given, it will be determined from the file
      extension. The filename can incorporate markers. By default, it is
      "{default_output_dir}/{sector}{year}{quantity}{suffix}".
    - any other parameter relevant to the sink, e.g. `pandas.to_csv` keyword
      arguments.

    For simplicity, it is also possible to given lone strings as input.
    They default to `{'quantity': string}` (and the sink will default to
    "csv").

    Raises:
        ValueError: If unknown quantities are requested to be cached.
    """

    def __init__(
        self,
        *parameters: Mapping,
        output_quantities: MutableMapping[str, OUTPUT_QUANTITY_SIGNATURE] | None = None,
        sectors: list[AbstractSector] | None = None,
        topic: str = CACHE_TOPIC_CHANNEL,
    ):
        from muse.outputs.sector import _factory

        output_quantities = (
            OUTPUT_QUANTITIES if output_quantities is None else output_quantities
        )
        self.agents: MutableMapping[str, MutableMapping[str, str]] = (
            extract_agents(sectors) if sectors is not None else {}
        )

        missing = [
            p["quantity"] for p in parameters if p["quantity"] not in output_quantities
        ]

        if len(missing) != 0:
            raise ValueError(
                f"There are unknown quantities to cache: {missing}. "
                f"Valid quantities are: {list(output_quantities.keys())}"
            )

        self.to_save: Mapping[str, list[xr.DataArray]] = {
            p["quantity"]: [] for p in parameters if p["quantity"] in output_quantities
        }

        self.factory: Mapping[str, Callable] = {
            p["quantity"]: _factory(output_quantities, p, sector_name="Cache")
            for p in parameters
            if p["quantity"] in self.to_save
        }

        self.normalized = {}
        for p in self.to_save:
            for alt, val in output_quantities.items():
                if val != output_quantities[p]:
                    continue
                self.normalized[alt] = p

        pub.subscribe(self.cache, topic)

    def cache(self, data: Mapping[str, xr.DataArray]) -> None:
        """Caches the data into memory.

        If the quantity has not been selected to be cached when configuring the
        MUSE simulation, it will be silently ignored if present as an input to this
        function.

        Args:
            data (Mapping[str, xr.DataArray]): Dictionary with the quantities and
            DataArray values to save.
        """
        for quantity, value in data.items():
            normalized = self.normalized.get(quantity, "")
            if normalized not in self.to_save:
                continue

            self.to_save[normalized].append(value.copy())
            self.to_save[normalized][-1].name = normalized

    def consolidate_cache(self, year: int) -> None:
        """Save the cached data into disk and flushes cache.

        This method is meant to be called after each time period in the main loop of the
        MCA, just after market and sector quantities are saved.

        Args:
            year (int): Year of interest.
        """
        for quantity, cache in self.to_save.items():
            if len(cache) == 0:
                continue

            self.factory[quantity](cache, self.agents, year=year)
        self.to_save = {q: [] for q in self.to_save}


def extract_agents(
    sectors: list[AbstractSector],
) -> MutableMapping[str, MutableMapping[str, str]]:
    """_summary_.

    Args:
        sectors (List[AbstractSector]): _description_

    Returns:
        Mapping[Text, Text]: _description_
    """
    return ChainMap(*[extract_agents_internal(sector) for sector in sectors])


def extract_agents_internal(
    sector: AbstractSector,
) -> MutableMapping[str, MutableMapping[str, str]]:
    """Extract simple agent metadata from a sector.

    Args:
        sector (AbstractSector): Sector to extract the metadata from.

    Returns:
        Mapping[Text, Text]: A dictionary with the uuid of each agent as keys and a
        dictionary with the name, agent type and agent sector as values.
    """
    info: MutableMapping[str, MutableMapping[str, str]] = {}
    sector_name = getattr(sector, "name", "unnamed")
    agents = sorted(getattr(sector, "agents", []), key=attrgetter("name"))
    for agent in agents:
        aid = agent.uuid
        info[aid] = {}
        info[aid]["agent"] = agent.name
        info[aid]["category"] = agent.category
        info[aid]["sector"] = sector_name
        info[aid]["dst_region"] = agent.region
        info[aid]["year"] = agent.forecast_year
        info[aid]["installed"] = agent.year

    return info


def _aggregate_cache(quantity: str, data: list[xr.DataArray]) -> pd.DataFrame:
    """Combine a list of DataArrays in a dataframe.

    The merging gives precedence to the last entries of the list over the first ones.
    I.e, the records of the arrays cached last will overwrite those of the ones cached
    before in the case of having identical index.

    Args:
        quantity: The quantity to cache.
        data: The list of DataArrays to combine.

    Returns:
        pd.DataFrame: A Dataframe with the data aggregated.
    """
    data = [da.to_dataframe().reset_index() for da in data]
    cols = sorted(
        set.intersection(
            *map(set, [[c for c in d.columns if c != quantity] for d in data])
        )
    )

    def check_col(colname: str) -> str:
        if colname.endswith("_x") or colname.endswith("_y"):
            return colname.rsplit("_", maxsplit=1)[0]
        return colname

    return reduce(
        lambda left, right: pd.DataFrame.merge(left, right, how="outer", on=cols)
        .T.groupby(check_col)
        .last()
        .T,
        data,
    )


def consolidate_quantity(
    quantity: str,
    cached: list[xr.DataArray],
    agents: MutableMapping[str, MutableMapping[str, str]],
) -> pd.DataFrame:
    """Consolidates the cached quantity into a single DataFrame to save.

    Args:
        quantity (Text): The quantity to cache.
        cached (List[xr.DataArray]): The list of cached arrays
        agents (MutableMapping[Text, MutableMapping[Text, Text]]): Agents' metadata.

    Returns:
        pd.DataFrame: DataFrame with the consolidated data.
    """
    data = _aggregate_cache(quantity, cached)

    ignore_dst_region = "dst_region" in data.columns
    for agent in tuple(agents):
        filter = data.agent == agent
        for key, value in agents[agent].items():
            if key == "dst_region" and ignore_dst_region:
                continue
            data.loc[filter, key] = value

    data = data.rename(columns={"replacement": "technology"})

    group_cols = [c for c in data.columns if c not in [quantity, "asset"]]
    data = (
        data.groupby(group_cols)
        .sum()
        .infer_objects()
        .fillna(0)
        .reset_index()
        .drop("asset", axis=1, errors="ignore")
    )
    data = data[data[quantity] != 0]
    return data[sorted(data.columns)]


@register_cached_quantity
def capacity(
    cached: list[xr.DataArray],
    agents: MutableMapping[str, MutableMapping[str, str]],
) -> pd.DataFrame:
    """Consolidates the cached capacities into a single DataFrame to save.

    Args:
        cached (List[xr.DataArray]): The list of cached arrays
        agents (MutableMapping[Text, MutableMapping[Text, Text]]): Agents' metadata.

    Returns:
        pd.DataFrame: DataFrame with the consolidated data.
    """
    return consolidate_quantity("capacity", cached, agents)


@register_cached_quantity
def production(
    cached: list[xr.DataArray],
    agents: MutableMapping[str, MutableMapping[str, str]],
) -> pd.DataFrame:
    """Consolidates the cached production into a single DataFrame to save.

    Args:
        cached (List[xr.DataArray]): The list of cached arrays
        agents (MutableMapping[Text, MutableMapping[Text, Text]]): Agents' metadata.

    Returns:
        pd.DataFrame: DataFrame with the consolidated data.
    """
    return consolidate_quantity("production", cached, agents)


@register_cached_quantity(name="lifetime_levelized_cost_of_energy")
def lcoe(
    cached: list[xr.DataArray],
    agents: MutableMapping[str, MutableMapping[str, str]],
) -> pd.DataFrame:
    """Consolidates the cached LCOE into a single DataFrame to save.

    Args:
        cached (List[xr.DataArray]): The list of cached arrays
        agents (MutableMapping[Text, MutableMapping[Text, Text]]): Agents' metadata.

    Returns:
        pd.DataFrame: DataFrame with the consolidated data.
    """
    """Consolidates the cached LCOE into a single DataFrame to save."""
    if "timeslice" in cached[0].dims:
        cached = [c.assign_coords(timeslice=c.timeslice.data) for c in cached]
    return consolidate_quantity("lcoe", cached, agents)
