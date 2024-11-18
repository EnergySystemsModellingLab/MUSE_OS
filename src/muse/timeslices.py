"""Timeslice utility functions."""

__all__ = [
    "read_timeslices",
    "broadcast_timeslice",
    "distribute_timeslice",
    "drop_timeslice",
    "setup_module",
]

from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
from xarray import DataArray

TIMESLICE: DataArray = None  # type: ignore


def read_timeslices(
    settings: Union[Mapping, str],
    level_names: Sequence[str] = ("month", "day", "hour"),
) -> pd.DataFrame:
    from functools import reduce

    from toml import loads

    # Read timeslice settings
    if isinstance(settings, str):
        settings = loads(settings)
    settings = dict(**settings.get("timeslices", settings))

    # Extract level names
    if "level_names" in settings:
        level_names = settings.pop("level_names")

    # Extract timeslice levels and lengths
    ts = list(settings.values())
    levels: list[tuple] = [(level,) for level in settings]
    while all(isinstance(v, Mapping) for v in ts):
        levels = [(*previous, b) for previous, a in zip(levels, ts) for b in a]
        ts = reduce(list.__add__, (list(u.values()) for u in ts), [])

    # Prepare multiindex
    nln = min(len(levels[0]), len(level_names))
    level_names = (
        list(level_names[:nln]) + [str(i) for i in range(len(levels[0]))][nln:]
    )
    indices = pd.MultiIndex.from_tuples(levels, names=level_names)

    # Make sure names from different levels don't overlap
    if any(
        reduce(set.union, indices.levels[:i], set()).intersection(indices.levels[i])
        for i in range(1, indices.nlevels)
    ):
        raise ValueError("Names from different levels should not overlap.")

    # Create DataArray
    return DataArray(ts, coords={"timeslice": indices}, dims="timeslice")


def setup_module(settings: Union[str, Mapping]):
    """Sets up module singletons."""
    global TIMESLICE
    TIMESLICE = read_timeslices(settings)


def broadcast_timeslice(
    x: DataArray, ts: Optional[DataArray] = None, level: Optional[str] = None
) -> DataArray:
    """Convert a non-timesliced array to a timesliced array by broadcasting.

    If x is already timesliced in the appropriate scheme, it will be returned unchanged.

    Args:
        x: Array to broadcast.
        ts: Dataarray with timeslice lengths. If None, defaults to the global timeslice.
        level: Level to broadcast to. If None, use the finest level of ts.

    """
    from xarray import Coordinates

    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = compress_timeslice(ts, ts=ts, level=level, operation="sum")

    # If x already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in x.dims:
        if x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
            return x
        raise ValueError("x has incompatible timeslicing.")

    mindex_coords = Coordinates.from_pandas_multiindex(ts.timeslice, "timeslice")
    broadcasted = x.expand_dims(timeslice=ts["timeslice"]).assign_coords(mindex_coords)
    return broadcasted


def distribute_timeslice(
    x: DataArray, ts: Optional[DataArray] = None, level=None
) -> DataArray:
    """Convert a non-timesliced array to a timesliced array by distribution.

    If x is already timesliced in the appropriate scheme, it will be returned unchanged.

    Args:
        x: Array to distribute.
        ts: Dataarray with timeslice lengths. If None, defaults to the global timeslice.
        level: Level to distribute to. If None, use the finest level of ts.

    """
    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = compress_timeslice(ts, ts=ts, level=level, operation="sum")

    # If x already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in x.dims:
        if x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
            return x
        raise ValueError("x has incompatible timeslicing.")

    broadcasted = broadcast_timeslice(x, ts=ts)
    timeslice_fractions = ts / broadcast_timeslice(ts.sum(), ts=ts)
    return broadcasted * timeslice_fractions


def compress_timeslice(
    x: DataArray,
    ts: Optional[DataArray] = None,
    level: Optional[str] = None,
    operation: str = "sum",
) -> DataArray:
    """Convert a fully timesliced array to a coarser level.

    The operation can be either 'sum', or 'mean':
    - sum: sum values at each compressed timeslice level
    - mean: take a weighted average of values at each compressed timeslice level

    Args:
        x: Timesliced array to compress. Must have the same timeslicing as ts.
        ts: Dataarray with timeslice lengths. If None, defaults to the global timeslice.
        level: Level to compress to. If None, don't compress.
        operation: Operation to perform ("sum" or "mean"). Defaults to "sum".

    """
    if ts is None:
        ts = TIMESLICE

    # Raise error if x is not timesliced appropriately
    if "timeslice" not in x.dims:
        raise ValueError("x must have a 'timeslice' dimension.")
    if not x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
        raise ValueError("x has incompatible timeslicing.")

    # If level is not specified, don't compress
    if level is None:
        return x

    # level must be a valid timeslice level
    x_levels = x.timeslice.to_index().names
    if level not in x_levels:
        raise ValueError(f"Unknown level: {level}. Must be one of {x_levels}.")

    # Return x unchanged if already at the desired level
    if get_level(x) == level:
        return x

    # Prepare mask
    idx = x_levels.index(level)
    kept_levels, compressed_levels = x_levels[: idx + 1], x_levels[idx + 1 :]
    mask = ts.unstack(dim="timeslice")
    if operation == "sum":
        mask = mask.where(np.isnan(mask), 1)
    elif operation == "mean":
        mask = mask / mask.sum(compressed_levels)
    else:
        raise ValueError(f"Unknown operation: {operation}. Must be 'sum' or 'mean'.")

    # Perform the operation
    result = (
        (x.unstack(dim="timeslice") * mask)
        .sum(compressed_levels)
        .stack(timeslice=kept_levels)
    )
    return sort_timeslices(result, ts)


def expand_timeslice(
    x: DataArray, ts: Optional[DataArray] = None, operation: str = "distribute"
) -> DataArray:
    """Convert a timesliced array to a finer level.

    The operation can be either 'distribute', or 'broadcast'
    - distribute: distribute values over the new timeslice level(s) according to
        timeslice lengths, such that the sum of the output over all timeslices is equal
        to the sum of the input
    - broadcast: broadcast values across over the new timeslice level(s)

    Args:
        x: Timesliced array to expand.
        ts: Dataarray with timeslice lengths. If None, defaults to the global timeslice.
        operation: Operation to perform ("distribute" or "broadcast").
            Defaults to "distribute".

    """
    if ts is None:
        ts = TIMESLICE

    # Raise error if x is not timesliced
    if "timeslice" not in x.dims:
        raise ValueError("DataArray must have a 'timeslice' dimension.")

    # Get level names
    ts_levels = ts.timeslice.to_index().names
    x_levels = x.timeslice.to_index().names

    # Raise error if x_level is not a subset of ts_levels
    if not set(x_levels).issubset(ts_levels):
        raise ValueError(
            f"Timeslice levels of x ({x_levels}) must be a subset of ts ({ts_levels})."
        )

    # Return x unchanged if already at the desired level
    finest_level = get_level(ts)
    current_level = get_level(x)
    if current_level == finest_level:
        return x

    # Prepare mask
    mask = ts.unstack(dim="timeslice")
    if operation == "broadcast":
        mask = mask.where(np.isnan(mask), 1)
    elif operation == "distribute":
        mask = mask / mask.sum(ts_levels[ts_levels.index(current_level) + 1 :])
    else:
        raise ValueError(
            f"Unknown operation: {operation}. Must be 'distribute' or 'broadcast'."
        )

    # Perform the operation
    result = (
        (x.unstack(dim="timeslice") * mask)
        .stack(timeslice=ts_levels)
        .dropna("timeslice")
    )
    return sort_timeslices(result, ts)


def drop_timeslice(data: DataArray) -> DataArray:
    """Drop the timeslice variable from a DataArray.

    If the array doesn't contain the timeslice variable, return the input unchanged.
    """
    if "timeslice" not in data.dims:
        return data

    return data.drop_vars(data.timeslice.indexes)


def get_level(data: DataArray) -> str:
    """Get the timeslice level of a DataArray."""
    if "timeslice" not in data.dims:
        raise ValueError("DataArray does not have a 'timeslice' dimension.")
    return data.timeslice.to_index().names[-1]


def sort_timeslices(data: DataArray, ts: Optional[DataArray] = None) -> DataArray:
    """Sorts the timeslices of a DataArray according to a reference timeslice."""
    if ts is None:
        ts = TIMESLICE

    # If data is at the finest timeslice level, sort timeslices according to ts
    if get_level(data) == get_level(ts):
        return data.sel(timeslice=ts.timeslice)
    # Otherwise, sort timeslices in alphabetical order
    return data.sortby("timeslice")


def timeslice_max(x: DataArray, ts: Optional[DataArray] = None) -> DataArray:
    """Find the max value over the timeslice dimension, normalized for timeslice length.

    This first annualizes the value in each timeslice by dividing by the fraction of the
    year that the timeslice occupies, then takes the maximum value
    """
    if ts is None:
        ts = TIMESLICE

    timeslice_level = get_level(x)
    timeslice_fractions = compress_timeslice(
        ts, ts=ts, level=timeslice_level, operation="sum"
    ) / broadcast_timeslice(ts.sum(), ts=ts, level=timeslice_level)
    return (x / timeslice_fractions).max("timeslice")
