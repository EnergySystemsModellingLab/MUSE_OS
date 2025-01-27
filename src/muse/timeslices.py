"""Timeslice utility functions."""

from __future__ import annotations

__all__ = [
    "broadcast_timeslice",
    "compress_timeslice",
    "distribute_timeslice",
    "drop_timeslice",
    "expand_timeslice",
    "get_level",
    "read_timeslices",
    "setup_module",
    "sort_timeslices",
    "timeslice_max",
]

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from xarray import DataArray

TIMESLICE: DataArray = None  # type: ignore


def read_timeslices(
    settings: Mapping | str,
    level_names: Sequence[str] = ("month", "day", "hour"),
) -> DataArray:
    from functools import reduce
    from logging import getLogger

    from toml import loads

    # Read timeslice settings
    if isinstance(settings, str):
        settings = loads(settings)
    settings = dict(**settings.get("timeslices", settings))

    # Legacy: warn user about deprecation of "aggregates" feature (#550)
    if "aggregates" in settings:
        msg = (
            "Timeslice aggregation has been deprecated since v1.3.0. Please see the "
            "release notes for that version for more information."
        )
        getLogger(__name__).warning(msg)
        settings.pop("aggregates")

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


def setup_module(settings: str | Mapping):
    """Sets up module singletons."""
    global TIMESLICE
    TIMESLICE = read_timeslices(settings)


def broadcast_timeslice(
    data: DataArray, ts: DataArray | None = None, level: str | None = None
) -> DataArray:
    """Convert a non-timesliced array to a timesliced array by broadcasting.

    If data is already timesliced in the appropriate scheme, it will be returned
    unchanged.

    Args:
        data: Array to broadcast.
        ts: Dataarray with timeslice weights. If None, defaults to the global timeslice.
        level: Level to broadcast to. If None, use the finest level of ts.

    """
    from xarray import Coordinates

    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = compress_timeslice(ts, ts=ts, level=level, operation="sum")

    # If data already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in data.dims:
        if data.timeslice.reset_coords(drop=True).equals(
            ts.timeslice.reset_coords(drop=True)
        ):
            return data
        raise ValueError(
            "Data is already timesliced, but does not match the reference."
        )

    mindex_coords = Coordinates.from_pandas_multiindex(ts.timeslice, "timeslice")
    broadcasted = data.expand_dims(timeslice=ts["timeslice"]).assign_coords(
        mindex_coords
    )
    return broadcasted


def distribute_timeslice(
    data: DataArray, ts: DataArray | None = None, level=None
) -> DataArray:
    """Convert a non-timesliced array to a timesliced array by distribution.

    Takes non-timesliced data and distributes it over the timeslice dimension according
    to the timeslice weights in `ts`. The sum of the output over all timeslices will be
    equal to the input. If data is already timesliced in the appropriate scheme, it will
    be returned unchanged.

    Args:
        data: Array to distribute.
        ts: Dataarray with timeslice weights. If None, defaults to the global timeslice.
        level: Level to distribute to. If None, use the finest level of ts.

    """
    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = compress_timeslice(ts, ts=ts, level=level, operation="sum")

    # If data already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in data.dims:
        if data.timeslice.reset_coords(drop=True).equals(
            ts.timeslice.reset_coords(drop=True)
        ):
            return data
        raise ValueError(
            "Data is already timesliced, but does not match the reference."
        )

    broadcasted = broadcast_timeslice(data, ts=ts)
    timeslice_sum = ts.sum("timeslice").clip(1e-6)  # prevents zero division
    timeslice_fractions = ts / broadcast_timeslice(timeslice_sum, ts=ts)
    return broadcasted * timeslice_fractions


def compress_timeslice(
    data: DataArray,
    ts: DataArray | None = None,
    level: str | None = None,
    operation: str = "sum",
) -> DataArray:
    """Convert a fully timesliced array to a coarser level.

    The operation can be either 'sum', or 'mean':
    - sum: sum values at each compressed timeslice level
    - mean: take a weighted average of values at each compressed timeslice level,
        according to the timeslice weights in ts

    Args:
        data: Timesliced array to compress. Must have the same timeslicing as ts.
        ts: Dataarray with timeslice weights. If None, defaults to the global timeslice.
        level: Level to compress to. If None, don't compress.
        operation: Operation to perform ("sum" or "mean"). Defaults to "sum".

    """
    if ts is None:
        ts = TIMESLICE

    # Raise error if data is not timesliced appropriately
    if "timeslice" not in data.dims:
        raise ValueError("Data must have a 'timeslice' dimension.")
    if not data.timeslice.reset_coords(drop=True).equals(
        ts.timeslice.reset_coords(drop=True)
    ):
        raise ValueError("Data has incompatible timeslicing with reference.")

    # If level is not specified, don't compress
    if level is None:
        return data

    # level must be a valid timeslice level
    x_levels = data.timeslice.to_index().names
    if level not in x_levels:
        raise ValueError(f"Unknown level: {level}. Must be one of {x_levels}.")

    # Return data unchanged if already at the desired level
    if get_level(data) == level:
        return data

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
        (data.unstack(dim="timeslice") * mask)
        .sum(compressed_levels)
        .stack(timeslice=kept_levels)
    )
    return sort_timeslices(result, ts)


def expand_timeslice(
    data: DataArray, ts: DataArray | None = None, operation: str = "distribute"
) -> DataArray:
    """Convert a timesliced array to a finer level.

    The operation can be either 'distribute', or 'broadcast'
    - distribute: distribute values over the new timeslice level(s) according to
        timeslice weights in `ts`, such that the sum of the output over all timeslices
        is equal to the sum of the input
    - broadcast: broadcast values across over the new timeslice level(s)

    Args:
        data: Timesliced array to expand.
        ts: Dataarray with timeslice weights. If None, defaults to the global timeslice.
        operation: Operation to perform ("distribute" or "broadcast").
            Defaults to "distribute".

    """
    if ts is None:
        ts = TIMESLICE

    # Raise error if data is not timesliced
    if "timeslice" not in data.dims:
        raise ValueError("Data must have a 'timeslice' dimension.")

    # Get level names
    ts_levels = ts.timeslice.to_index().names
    x_levels = data.timeslice.to_index().names

    # Raise error if x_levels is not a subset of ts_levels
    if not set(x_levels).issubset(ts_levels):
        raise ValueError(
            "Data has incompatible timeslicing with reference. "
            f"Timeslice levels of data ({x_levels}) must be a subset of ts "
            f"({ts_levels})."
        )

    # Return data unchanged if already at the desired level
    finest_level = get_level(ts)
    current_level = get_level(data)
    if current_level == finest_level:
        return data

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
        (data.unstack(dim="timeslice") * mask)
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
        raise ValueError("Data does not have a 'timeslice' dimension.")
    return data.timeslice.to_index().names[-1]


def sort_timeslices(data: DataArray, ts: DataArray | None = None) -> DataArray:
    """Sorts the timeslices of a DataArray according to a reference timeslice.

    This will only sort timeslices to match the reference if the data is at the same
    timeslice level as the reference. Otherwise, it will sort timeslices in alphabetical
    order.

    Args:
        data: Timesliced DataArray to sort.
        ts: Dataarray with reference timeslices in the appropriate order
    """
    if ts is None:
        ts = TIMESLICE

    # If data is at the same timeslice level as ts, sort timeslices according to ts
    if get_level(data) == get_level(ts):
        return data.sel(timeslice=ts.timeslice)
    # Otherwise, sort timeslices in alphabetical order
    return data.sortby("timeslice")


def timeslice_max(data: DataArray, ts: DataArray | None = None) -> DataArray:
    """Find the max value over the timeslice dimension, normalized for timeslice length.

    This first annualizes the value in each timeslice by dividing by the fraction of the
    year that the timeslice occupies, then takes the maximum value

    Args:
        data: Timesliced DataArray to find the max of.
        ts: Dataarray with relative timeslice lengths. If None, defaults to the global
        timeslice.
    """
    if ts is None:
        ts = TIMESLICE

    timeslice_level = get_level(data)
    timeslice_sum = ts.sum("timeslice").clip(1e-6)  # prevents zero division
    timeslice_fractions = compress_timeslice(
        ts, ts=ts, level=timeslice_level, operation="sum"
    ) / broadcast_timeslice(timeslice_sum, ts=ts, level=timeslice_level)
    return (data / timeslice_fractions).max("timeslice")
