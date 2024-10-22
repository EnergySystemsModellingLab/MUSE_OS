"""Timeslice utility functions."""

__all__ = [
    "read_timeslices",
    "convert_timeslice",
    "drop_timeslice",
    "setup_module",
]

from collections.abc import Mapping, Sequence
from enum import Enum, unique
from typing import Union

from numpy import ndarray
from pandas import MultiIndex
from xarray import DataArray

TIMESLICE: DataArray = None  # type: ignore
"""Array with the finest timeslice."""
TRANSFORMS: dict[tuple, ndarray] = None  # type: ignore
"""Transforms from each aggregate to the finest timeslice."""


def read_timeslices(
    settings: Union[Mapping, str],
    level_names: Sequence[str] = ("month", "day", "hour"),
    name: str = "timeslice",
) -> DataArray:
    '''Reads reference timeslice from toml like input.

    Arguments:
        settings: A dictionary of nested dictionaries or a string that toml will
            interpret as such. The nesting specifies different levels of the timeslice.
            If a dictionary and it contains "timeslices" key, then the associated value
            is used as the root dictionary. Ultimately, the most nested values should be
            relative weights for each slice in the timeslice, e.g. the corresponding
            number of hours.
        level_names: Hints indicating the names of each level. Can also be given a
            "level_names" key in ``settings``.
        name: name of the reference array

    Return:
        A ``DataArray`` with dimension *timeslice* and values representing the relative
        weight of each timeslice.

    Example:
        >>> from muse.timeslices import read_timeslices
        >>> read_timeslices(
        ...     """
        ...     [timeslices]
        ...     spring.weekday = 5
        ...     spring.weekend = 2
        ...     autumn.weekday = 5
        ...     autumn.weekend = 2
        ...     winter.weekday = 5
        ...     winter.weekend = 2
        ...     summer.weekday = 5
        ...     summer.weekend = 2
        ...     level_names = ["season", "week"]
        ...     """
        ... )  # doctest: +SKIP
        <xarray.DataArray (timeslice: 8)> Size: 32B
        array([5, 2, 5, 2, 5, 2, 5, 2])
        Coordinates:
          * timeslice  (timeslice) object 64B MultiIndex
          * season     (timeslice) object 64B 'spring' 'spring' ... 'summer' 'summer'
          * week       (timeslice) object 64B 'weekday' 'weekend' ... 'weekend'
    '''
    from functools import reduce

    from toml import loads

    if isinstance(settings, str):
        settings = loads(settings)
    settings = dict(**settings.get("timeslices", settings))
    if "level_names" in settings:
        level_names = settings.pop("level_names")
    settings.pop("aggregates", {})

    # figures out levels
    levels: list[tuple] = [(level,) for level in settings]
    ts = list(settings.values())
    while all(isinstance(v, Mapping) for v in ts):
        levels = [(*previous, b) for previous, a in zip(levels, ts) for b in a]
        ts = reduce(list.__add__, (list(u.values()) for u in ts), [])

    nln = min(len(levels[0]), len(level_names))
    level_names = (
        list(level_names[:nln]) + [str(i) for i in range(len(levels[0]))][nln:]
    )
    indices = MultiIndex.from_tuples(levels, names=level_names)

    if any(
        reduce(set.union, indices.levels[:i], set()).intersection(indices.levels[i])
        for i in range(1, indices.nlevels)
    ):
        raise ValueError("Names from different levels should not overlap.")

    return DataArray(ts, coords={"timeslice": indices}, dims=name)


def setup_module(settings: Union[str, Mapping]):
    """Sets up module singletons."""
    global TIMESLICE
    TIMESLICE = read_timeslices(settings)


@unique
class QuantityType(Enum):
    """Underlying transformation when performing time-slice conversion.

    The meaning of a quantity vs the time-slice can be different:

    - intensive: when extending the period of interest, quantities should be
      added together. For instance the number of hours should be summed across
      months.
    - extensive: when extending the period of interest, quantities should be
      broadcasted. For instance when extending a price from a one week period to
      a two week period, the price should remain the same. Going in the opposite
      direction (reducing the length of the time period), quantities should be
      averaged.
    """

    INTENSIVE = "intensive"
    EXTENSIVE = "extensive"


def convert_timeslice(x, ts=None, quantity=QuantityType.INTENSIVE):
    from xarray import Coordinates

    if ts is None:
        ts = TIMESLICE

    if hasattr(x, "timeslice"):
        x = x.sel(timeslice=ts["timeslice"])
        return x

    mindex_coords = Coordinates.from_pandas_multiindex(ts.timeslice, "timeslice")
    extensive = x.expand_dims(timeslice=ts["timeslice"]).assign_coords(mindex_coords)
    if quantity is QuantityType.EXTENSIVE:
        return extensive

    if quantity is QuantityType.INTENSIVE:
        return extensive * (ts / ts.sum())


def drop_timeslice(data: DataArray) -> DataArray:
    """Drop the timeslice variable from a DataArray.

    If the array doesn't contain the timeslice variable, return the input unchanged.
    """
    if "timeslice" not in data.dims:
        return data

    return data.drop_vars(data.timeslice.indexes)
