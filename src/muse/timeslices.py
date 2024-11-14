"""Timeslice utility functions."""

__all__ = [
    "read_timeslices",
    "broadcast_timeslice",
    "distribute_timeslice",
    "drop_timeslice",
    "setup_module",
]

from collections.abc import Mapping, Sequence
from typing import Union

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
    x: DataArray, ts: DataArray | None = None, level: str | None = None
):
    """Convert a non-timesliced array to a timesliced array by broadcasting."""
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
    extensive = x.expand_dims(timeslice=ts["timeslice"]).assign_coords(mindex_coords)
    return extensive


def distribute_timeslice(x: DataArray, ts: DataArray | None = None, level=None):
    """Convert a non-timesliced array to a timesliced array by distribution."""
    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = compress_timeslice(ts, level=level, operation="sum")

    # If x already has timeslices, check that it matches the reference timeslice.
    if "timeslice" in x.dims:
        if x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
            return x
        raise ValueError("x has incompatible timeslicing.")

    extensive = broadcast_timeslice(x, ts)
    return extensive * (ts / broadcast_timeslice(ts.sum(), ts))


def compress_timeslice(
    x: DataArray,
    ts: DataArray | None = None,
    level: str | None = None,
    operation: str = "sum",
):
    """Convert a timesliced array to a lower level by performing the given operation.

    The operation can be either 'sum', or 'mean'
    """
    if ts is None:
        ts = TIMESLICE

    # If level is not specified, don't compress
    if level is None:
        return x

    # Get level names from x
    level_names = x.timeslice.to_index().names
    if level not in level_names:
        raise ValueError(f"Unknown level: {level}. Must be one of {level_names}.")
    current_level, coarser_levels = level_names[-1], level_names[:-1]

    # Return if already at the desired level
    if current_level == level:
        return x

    # Perform the operation over one timeslice level
    if operation == "sum":
        x = (
            x.unstack(dim="timeslice")
            .sum(current_level)
            .stack(timeslice=coarser_levels)
        )
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
    x: DataArray, ts: DataArray | None = None, operation: str = "distribute"
):
    """Convert a timesliced array to the global scheme by expanding.

    The operation can be either 'distribute', or 'broadcast'
    - distribute: distribute the values according to timeslice length
    - broadcast: broadcast the values across the new timeslice level
    """
    if ts is None:
        ts = TIMESLICE

    # Get level names from ts
    level_names = ts.timeslice.to_index().names
    finest_level = level_names[-1]

    # Return if already at the finest level
    current_level = x.timeslice.to_index().names[-1]
    if current_level == finest_level:
        return x
    else:
        pass

    # Perform the operation over one timeslice level
    finer_level = level_names[level_names.index(current_level) + 1]
    if operation == "broadcast":
        return x  # TODO
    elif operation == "distribute":
        return x  # TODO
    else:
        raise ValueError(
            f"Unknown operation: {operation}. Must be 'distribute' or 'broadcast'."
        )

    # Recurse
    return expand_timeslice(x, ts=ts, operation=operation)


def drop_timeslice(data: DataArray) -> DataArray:
    """Drop the timeslice variable from a DataArray.

    If the array doesn't contain the timeslice variable, return the input unchanged.
    """
    if "timeslice" not in data.dims:
        return data

    return data.drop_vars(data.timeslice.indexes)
