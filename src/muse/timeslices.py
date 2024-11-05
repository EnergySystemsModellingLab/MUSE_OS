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
"""Array with the finest timeslice."""
TRANSFORMS: dict[str, DataArray] = None  # type: ignore


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

    # Create DataFrame
    df = pd.DataFrame(ts, columns=["value"])
    df["level"] = levels
    df[level_names] = pd.DataFrame(df["level"].tolist(), index=df.index)
    df = df.drop("level", axis=1).set_index(level_names)
    return df


def setup_module(settings: Union[str, Mapping]):
    """Sets up module singletons."""
    global TIMESLICE
    global TRANSFORMS

    df = read_timeslices(settings)

    # Global timeslicing scheme
    TIMESLICE = DataArray(
        df.values.flatten(), coords={"timeslice": df.index}, dims="timeslice"
    )

    # Timeslices aggregated to each level
    TRANSFORMS = {}
    levels = df.index.names
    for i, level in enumerate(levels):
        group = levels[: i + 1]
        df_grouped = df.groupby(group, sort=False).sum()
        if isinstance(df_grouped.index, pd.MultiIndex):
            coords = {"timeslice": df_grouped.index}
        else:
            coords = {"timeslice": df_grouped.index.tolist()}
        TRANSFORMS[level] = DataArray(
            df_grouped.values.flatten(),
            coords=coords,
            dims="timeslice",
        )


def broadcast_timeslice(x, ts=None, level=None):
    from xarray import Coordinates

    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = TRANSFORMS[level]

    # If x already has timeslices, check that it is matches the reference timeslice.
    if "timeslice" in x.dims:
        if x.timeslice.reset_coords(drop=True).equals(ts.timeslice):
            return x
        raise ValueError("x has incompatible timeslicing.")

    mindex_coords = Coordinates.from_pandas_multiindex(ts.timeslice, "timeslice")
    extensive = x.expand_dims(timeslice=ts["timeslice"]).assign_coords(mindex_coords)
    return extensive


def distribute_timeslice(x, ts=None, level=None):
    if ts is None:
        ts = TIMESLICE

    if level is not None:
        ts = TRANSFORMS[level]

    extensive = broadcast_timeslice(x, ts, level)
    return extensive * (ts / broadcast_timeslice(ts.sum(), level=level))


def drop_timeslice(data: DataArray) -> DataArray:
    """Drop the timeslice variable from a DataArray.

    If the array doesn't contain the timeslice variable, return the input unchanged.
    """
    if "timeslice" not in data.dims:
        return data

    return data.drop_vars(data.timeslice.indexes)
