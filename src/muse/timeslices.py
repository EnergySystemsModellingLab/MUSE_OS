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

    nln = min(len(levels[0]), len(level_names))
    level_names = (
        list(level_names[:nln]) + [str(i) for i in range(len(levels[0]))][nln:]
    )
    indices = pd.MultiIndex.from_tuples(levels, names=level_names)

    if any(
        reduce(set.union, indices.levels[:i], set()).intersection(indices.levels[i])
        for i in range(1, indices.nlevels)
    ):
        raise ValueError("Names from different levels should not overlap.")

    return DataArray(ts, coords={"timeslice": indices}, dims="timeslice")

    # # Create DataFrame
    # df = pd.DataFrame(ts, columns=["value"])
    # df["level"] = levels
    # df[level_names] = pd.DataFrame(df["level"].tolist(), index=df.index)
    # df = df.drop("level", axis=1).set_index(level_names)
    # return df


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
        ts = compress_timeslice(ts, level=level, operation="sum")

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
        ts = compress_timeslice(ts, level=level, operation="sum")

    # If x already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in x.dims:
        if x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
            return x
        raise ValueError("x has incompatible timeslicing.")

    broadcasted = broadcast_timeslice(x, ts)
    timeslice_fractions = ts / broadcast_timeslice(ts.sum(), ts)
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

    # Raise error if x is not timesliced
    if "timeslice" not in x.dims:
        raise ValueError("DataArray must have a 'timeslice' dimension.")

    # If level is not specified, don't compress
    if level is None:
        return x

    # x must have the same timeslicing as ts
    if not x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
        raise ValueError("x has incompatible timeslicing.")

    # level must be a valid timeslice level
    x_levels = x.timeslice.to_index().names
    if level not in x_levels:
        raise ValueError(f"Unknown level: {level}. Must be one of {x_levels}.")
    current_level, coarser_levels = x_levels[-1], x_levels[:-1]

    # Return x unchanged if already at the desired level
    if current_level == level:
        return x

    # Perform the operation over one timeslice level
    if operation == "sum":
        x = (
            x.unstack(dim="timeslice")
            .sum(current_level)
            .stack(timeslice=coarser_levels)
        )
        # return x.unstack(dim="timeslice").sum(["hour"]).stack(timeslice=["month", "day"])
    elif operation == "mean":
        # TODO: This should be a weighted mean according to timeslice length
        x = (
            x.unstack(dim="timeslice")
            .mean(current_level)
            .stack(timeslice=coarser_levels)
        )
    else:
        raise ValueError(f"Unknown operation: {operation}. Must be 'sum' or 'mean'.")

    # Recurse
    return compress_timeslice(x, ts=ts, level=level, operation=operation)


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
    return (
        (x.unstack(dim="timeslice") * mask)
        .stack(timeslice=ts_levels)
        .dropna("timeslice")
        .sel(timeslice=ts.timeslice)
    )


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
