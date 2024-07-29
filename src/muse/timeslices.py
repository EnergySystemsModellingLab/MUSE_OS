"""Timeslice utility functions."""

__all__ = [
    "reference_timeslice",
    "aggregate_transforms",
    "convert_timeslice",
    "timeslice_projector",
    "setup_module",
    "represent_hours",
]

from collections.abc import Mapping, Sequence
from enum import Enum, unique
from typing import Optional, Union

import xarray as xr
from numpy import ndarray
from pandas import MultiIndex
from xarray import DataArray, Dataset

from muse.readers import kebab_to_camel

TIMESLICE: DataArray = None  # type: ignore
"""Array with the finest timeslice."""
TRANSFORMS: dict[tuple, ndarray] = None  # type: ignore
"""Transforms from each aggregate to the finest timeslice."""

DEFAULT_TIMESLICE_DESCRIPTION = """
    [timeslices]
    winter.weekday.night = 396
    winter.weekday.morning = 396
    winter.weekday.afternoon = 264
    winter.weekday.early-peak = 66
    winter.weekday.late-peak = 66
    winter.weekday.evening = 396
    winter.weekend.night = 156
    winter.weekend.morning = 156
    winter.weekend.afternoon = 156
    winter.weekend.evening = 156
    spring-autumn.weekday.night = 792
    spring-autumn.weekday.morning = 792
    spring-autumn.weekday.afternoon = 528
    spring-autumn.weekday.early-peak = 132
    spring-autumn.weekday.late-peak = 132
    spring-autumn.weekday.evening = 792
    spring-autumn.weekend.night = 300
    spring-autumn.weekend.morning = 300
    spring-autumn.weekend.afternoon = 300
    spring-autumn.weekend.evening = 300
    summer.weekday.night = 396
    summer.weekday.morning  = 396
    summer.weekday.afternoon = 264
    summer.weekday.early-peak = 66
    summer.weekday.late-peak = 66
    summer.weekday.evening = 396
    summer.weekend.night = 150
    summer.weekend.morning = 150
    summer.weekend.afternoon = 150
    summer.weekend.evening = 150
    level_names = ["month", "day", "hour"]

    [timeslices.aggregates]
    all-day = [
        "night", "morning", "afternoon", "early-peak", "late-peak", "evening", "night"
    ]
    all-week = ["weekday", "weekend"]
    all-year = ["winter", "summer", "spring-autumn"]
    """


def reference_timeslice(
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
        >>> from muse.timeslices import reference_timeslice
        >>> reference_timeslice(
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


def aggregate_transforms(
    settings: Optional[Union[Mapping, str]] = None,
    timeslice: Optional[DataArray] = None,
) -> dict[tuple, ndarray]:
    '''Creates dictionary of transforms for aggregate levels.

    The transforms are used to create the projectors towards the finest timeslice.

    Arguments:
        timeslice: a ``DataArray`` with the timeslice dimension.
        settings: A dictionary mapping the name of an aggregate with the values it
            aggregates, or a string that toml will parse as such. If not given, only the
            unit transforms are returned.

    Return:
        A dictionary of transforms for each possible slice to it's corresponding finest
        timeslices.

    Example:
        >>> toml = """
        ...     [timeslices]
        ...     spring.weekday = 5
        ...     spring.weekend = 2
        ...     autumn.weekday = 5
        ...     autumn.weekend = 2
        ...     winter.weekday = 5
        ...     winter.weekend = 2
        ...     summer.weekday = 5
        ...     summer.weekend = 2
        ...
        ...     [timeslices.aggregates]
        ...     spautumn = ["spring", "autumn"]
        ...     week = ["weekday", "weekend"]
        ... """
        >>> from muse.timeslices import reference_timeslice, aggregate_transforms
        >>> ref = reference_timeslice(toml)
        >>> transforms = aggregate_transforms(toml, ref)
        >>> transforms[("spring", "weekend")]
        array([0, 1, 0, 0, 0, 0, 0, 0])
        >>> transforms[("spautumn", "weekday")]
        array([1, 0, 1, 0, 0, 0, 0, 0])
        >>> transforms[("autumn", "week")].T
        array([0, 0, 1, 1, 0, 0, 0, 0])
        >>> transforms[("spautumn", "week")].T
        array([1, 1, 1, 1, 0, 0, 0, 0])
    '''
    from itertools import product

    from numpy import identity, sum
    from toml import loads

    if timeslice is None:
        timeslice = TIMESLICE
    if settings is None:
        settings = {}
    elif isinstance(settings, str):
        settings = loads(settings)

    # get timeslice dimension
    Id = identity(len(timeslice), dtype=int)
    indices = timeslice.get_index("timeslice")
    unitvecs: dict[tuple, ndarray] = {index: Id[i] for (i, index) in enumerate(indices)}
    if "timeslices" in settings or "aggregates" in settings:
        settings = settings.get("timeslices", settings).get("aggregates", {})
    assert isinstance(settings, Mapping)

    assert set(settings).intersection(unitvecs) == set()
    levels = [list(level) for level in indices.levels]
    for name, equivalent in settings.items():
        matching_levels = [
            set(level).issuperset(equivalent) for level in indices.levels
        ]
        if sum(matching_levels) == 0:
            raise ValueError(f"Could not find matching level for {equivalent}")
        elif sum(matching_levels) > 1:
            raise ValueError(f"Found more than one matching level for {equivalent}")
        level = matching_levels.index(True)
        levels[level].append(name)

    result: dict[tuple, ndarray] = {}
    for index in set(product(*levels)).difference(unitvecs):
        if not any(level in settings for level in index):
            continue
        agglevels = set(product(*(settings.get(level, [level]) for level in index)))
        result[index] = sum(
            [unitvecs[agg] for agg in unitvecs if agg in agglevels], axis=0
        )
    result.update(unitvecs)
    return result


def setup_module(settings: Union[str, Mapping]):
    """Sets up module singletons."""
    global TIMESLICE
    global TRANSFORMS
    TIMESLICE = reference_timeslice(settings)
    TRANSFORMS = aggregate_transforms(settings, TIMESLICE)


def timeslice_projector(
    x: Union[DataArray, MultiIndex],
    finest: Optional[DataArray] = None,
    transforms: Optional[dict[tuple, ndarray]] = None,
) -> DataArray:
    '''Project time-slice to standardized finest time-slices.

    Returns a matrix from the input timeslice ``x`` to the ``finest`` timeslice, using
    the input ``transforms``. The latter are a set of transforms that map indices from
    one timeslice to indices in another.

    Example:
        Lets define the following timeslices and aggregates:

        >>> toml = """
        ...     ["timeslices"]
        ...     winter.weekday.day = 5
        ...     winter.weekday.night = 5
        ...     winter.weekend.day = 2
        ...     winter.weekend.night = 2
        ...     winter.weekend.dusk = 1
        ...     summer.weekday.day = 5
        ...     summer.weekday.night = 5
        ...     summer.weekend.day = 2
        ...     summer.weekend.night = 2
        ...     summer.weekend.dusk = 1
        ...     level_names = ["semester", "week", "day"]
        ...     aggregates.allday = ["day", "night"]
        ... """
        >>> from muse.timeslices import (
        ...     reference_timeslice,  aggregate_transforms
        ... )
        >>> ref = reference_timeslice(toml)
        >>> transforms = aggregate_transforms(toml, ref)
        >>> from pandas import MultiIndex
        >>> input_ts = DataArray(
        ...     [1, 2, 3],
        ...     coords={
        ...         "timeslice": MultiIndex.from_tuples(
        ...             [
        ...                 ("winter", "weekday", "allday"),
        ...                 ("winter", "weekend", "dusk"),
        ...                 ("summer", "weekend", "night"),
        ...             ],
        ...             names=ref.get_index("timeslice").names,
        ...         ),
        ...     },
        ...     dims="timeslice"
        ... )
        >>> input_ts  # doctest: +SKIP
        <xarray.DataArray (timeslice: 3)> Size: 12B
        array([1, 2, 3])
        Coordinates:
          * timeslice  (timeslice) object 24B MultiIndex
          * semester   (timeslice) object 24B 'winter' 'winter' 'summer'
          * week       (timeslice) object 24B 'weekday' 'weekend' 'weekend'
          * day        (timeslice) object 24B 'allday' 'dusk' 'night'

        The input timeslice does not have to be complete. In any case, we can now
        compute a transform, i.e. a matrix that will take this timeslice and transform
        it to the equivalent times in the finest timeslice:

        >>> from muse.timeslices import timeslice_projector
        >>> timeslice_projector(input_ts, ref, transforms)  # doctest: +SKIP
        <xarray.DataArray 'projector' (finest_timeslice: 10, timeslice: 3)> Size: 120B
        array([[1, 0, 0],
               [1, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 1, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 1],
               [0, 0, 0]])
        Coordinates:
          * finest_timeslice  (finest_timeslice) object 80B MultiIndex
          * finest_semester   (finest_timeslice) object 80B 'winter' ... 'summer'
          * finest_week       (finest_timeslice) object 80B 'weekday' ... 'weekend'
          * finest_day        (finest_timeslice) object 80B 'day' 'night' ... 'dusk'
          * timeslice         (timeslice) object 24B MultiIndex
          * semester          (timeslice) object 24B 'winter' 'winter' 'summer'
          * week              (timeslice) object 24B 'weekday' 'weekend' 'weekend'
          * day               (timeslice) object 24B 'allday' 'dusk' 'night'

        It is possible to give as input an array which does not have a timeslice of its
        own:

        >>> nots = DataArray([5.0, 1.0, 2.0], dims="a", coords={'a': [1, 2, 3]})
        >>> timeslice_projector(nots, ref, transforms).T  # doctest: +SKIP
        <xarray.DataArray (timeslice: 1, finest_timeslice: 10)> Size: 40B
        array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        Coordinates:
          * finest_timeslice  (finest_timeslice) object 80B MultiIndex
          * finest_semester   (finest_timeslice) object 80B 'winter' ... 'summer'
          * finest_week       (finest_timeslice) object 80B 'weekday' ... 'weekend'
          * finest_day        (finest_timeslice) object 80B 'day' 'night' ... 'dusk'
        Dimensions without coordinates: timeslice
    '''
    from numpy import concatenate, ones_like
    from xarray import DataArray

    if finest is None:
        global TIMESLICE
        finest = TIMESLICE
    if transforms is None:
        global TRANSFORMS
        transforms = TRANSFORMS

    index = finest.get_index("timeslice")
    index = index.set_names(f"finest_{u}" for u in index.names)

    if isinstance(x, MultiIndex):
        timeslices = x
    elif "timeslice" in x.dims:
        timeslices = x.get_index("timeslice")
    else:
        return DataArray(
            ones_like(finest, dtype=int)[:, None],
            coords={"finest_timeslice": index},
            dims=("finest_timeslice", "timeslice"),
        )

    return DataArray(
        concatenate([transforms[index][:, None] for index in timeslices], axis=1),
        coords={"finest_timeslice": index, "timeslice": timeslices},
        dims=("finest_timeslice", "timeslice"),
        name="projector",
    )


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


def convert_timeslice(
    x: Union[DataArray, Dataset],
    ts: Union[DataArray, Dataset, MultiIndex],
    quantity: Union[QuantityType, str] = QuantityType.EXTENSIVE,
    finest: Optional[DataArray] = None,
    transforms: Optional[dict[tuple, ndarray]] = None,
) -> Union[DataArray, Dataset]:
    '''Adjusts the timeslice of x to match that of ts.

    The conversion can be done in on of two ways, depending on whether the
    quantity is extensive or intensive. See `QuantityType`.

    Example:
        Lets define three timeslices from finest, to fine, to rough:

        >>> toml = """
        ...     ["timeslices"]
        ...     winter.weekday.day = 5
        ...     winter.weekday.night = 5
        ...     winter.weekend.day = 2
        ...     winter.weekend.night = 2
        ...     summer.weekday.day = 5
        ...     summer.weekday.night = 5
        ...     summer.weekend.day = 2
        ...     summer.weekend.night = 2
        ...     level_names = ["semester", "week", "day"]
        ...     aggregates.allday = ["day", "night"]
        ...     aggregates.allweek = ["weekend", "weekday"]
        ...     aggregates.allyear = ["winter", "summer"]
        ... """
        >>> from muse.timeslices import setup_module
        >>> from muse.readers import read_timeslices
        >>> setup_module(toml)
        >>> finest_ts = read_timeslices()
        >>> fine_ts = read_timeslices(dict(week=["allweek"]))
        >>> rough_ts = read_timeslices(dict(semester=["allyear"], day=["allday"]))

        Lets also define to other data-arrays to demonstrate how we can play with
        dimensions:

        >>> from numpy import array
        >>> x = DataArray(
        ...     [5, 2, 3],
        ...     coords={'a': array([1, 2, 3], dtype="int64")},
        ...     dims='a'
        ... )
        >>> y = DataArray([1, 1, 2], coords={'b': ["d", "e", "f"]}, dims='b')

        We can now easily convert arrays with different dimensions. First, lets check
        conversion from an array with no timeslices:

        >>> from xarray import ones_like
        >>> from muse.timeslices import convert_timeslice, QuantityType
        >>> z = convert_timeslice(x, finest_ts, QuantityType.EXTENSIVE)
        >>> z.round(6)
        <xarray.DataArray (timeslice: 8, a: 3)> Size: 192B
        array([[0.892857, 0.357143, 0.535714],
               [0.892857, 0.357143, 0.535714],
               [0.357143, 0.142857, 0.214286],
               [0.357143, 0.142857, 0.214286],
               [0.892857, 0.357143, 0.535714],
               [0.892857, 0.357143, 0.535714],
               [0.357143, 0.142857, 0.214286],
               [0.357143, 0.142857, 0.214286]])
        Coordinates:
          * timeslice  (timeslice) object 64B MultiIndex
          * semester   (timeslice) object 64B 'winter' 'winter' ... 'summer' 'summer'
          * week       (timeslice) object 64B 'weekday' 'weekday' ... 'weekend'
          * day        (timeslice) object 64B 'day' 'night' 'day' ... 'day' 'night'
          * a          (a) int64 24B 1 2 3
        >>> z.sum("timeslice")
        <xarray.DataArray (a: 3)> Size: 24B
        array([5., 2., 3.])
        Coordinates:
          * a        (a) int64 24B 1 2 3

        As expected, the sum over timeslices recovers the original array.

        In the case of an intensive quantity without a timeslice dimension, the
        operation does not do anything:

        >>> convert_timeslice([1, 2], rough_ts, QuantityType.INTENSIVE)
        [1, 2]

        More interesting is the conversion between different timeslices:

        >>> from xarray import zeros_like
        >>> zfine = x + y + zeros_like(fine_ts.timeslice, dtype=int)
        >>> zrough = convert_timeslice(zfine, rough_ts)
        >>> zrough.round(6)
        <xarray.DataArray (timeslice: 2, a: 3, b: 3)> Size: 144B
        array([[[17.142857, 17.142857, 20.      ],
                [ 8.571429,  8.571429, 11.428571],
                [11.428571, 11.428571, 14.285714]],
        <BLANKLINE>
               [[ 6.857143,  6.857143,  8.      ],
                [ 3.428571,  3.428571,  4.571429],
                [ 4.571429,  4.571429,  5.714286]]])
        Coordinates:
          * timeslice  (timeslice) object 16B MultiIndex
          * semester   (timeslice) object 16B 'allyear' 'allyear'
          * week       (timeslice) object 16B 'weekday' 'weekend'
          * day        (timeslice) object 16B 'allday' 'allday'
          * a          (a) int64 24B 1 2 3
          * b          (b) <U1 12B 'd' 'e' 'f'

        We can check that nothing has been added to z (the quantity is ``EXTENSIVE`` by
        default):

        >>> from numpy import all
        >>> all(zfine.sum("timeslice").round(6) == zrough.sum("timeslice").round(6))
        <xarray.DataArray ()> Size: 1B
        array(True)

        Or that the ratio of weekdays to weekends makes sense:
        >>> weekdays = (
        ...    zrough
        ...    .unstack("timeslice")
        ...    .sel(week="weekday")
        ...    .stack(timeslice=["semester", "day"])
        ...    .squeeze()
        ... )
        >>> weekend = (
        ...    zrough
        ...    .unstack("timeslice")
        ...    .sel(week="weekend")
        ...    .stack(timeslice=["semester", "day"])
        ...    .squeeze()
        ... )
        >>> bool(all((weekend * 5).round(6) == (weekdays * 2).round(6)))
        True
    '''
    if finest is None:
        global TIMESLICE
        finest = TIMESLICE
    if transforms is None:
        global TRANSFORMS
        transforms = TRANSFORMS
    if hasattr(ts, "timeslice"):
        ts = ts.timeslice
    has_ts = "timeslice" in getattr(x, "dims", ())
    same_ts = has_ts and len(ts) == len(x.timeslice) and x.timeslice.equals(ts)
    if same_ts or ((not has_ts) and quantity == QuantityType.INTENSIVE):
        return x
    quantity = QuantityType(quantity)
    proj0 = timeslice_projector(x, finest=finest, transforms=transforms)
    proj1 = timeslice_projector(ts, finest=finest, transforms=transforms)
    if quantity is QuantityType.EXTENSIVE:
        finest = finest.rename(timeslice="finest_timeslice")
        index = finest.get_index("finest_timeslice")
        index = index.set_names(f"finest_{u}" for u in index.names)
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, "finest_timeslice")
        finest = finest.drop_vars(list(finest.coords)).assign_coords(mindex_coords)
        proj0 = proj0 * finest
        proj0 = proj0 / proj0.sum("finest_timeslice")
    elif quantity is QuantityType.INTENSIVE:
        proj1 = proj1 / proj1.sum("finest_timeslice")

    new_names = {"timeslice": "final_ts"} | {
        c: f"{c}_ts" for c in proj1.timeslice.coords if c != "timeslice"
    }
    P = (proj1.rename(**new_names) * proj0).sum("finest_timeslice")

    final_names = {"final_ts": "timeslice"} | {
        c: c.replace("_ts", "") for c in P.final_ts.coords if c != "final_ts"
    }
    return (P * x).sum("timeslice").rename(**final_names)


def new_to_old_timeslice(ts: DataArray, ag_level="Month") -> dict:
    """Transforms timeslices defined as DataArray to a pandas dataframe.

    This function is used in the LegacySector class to adapt the new MCA timeslices to
    the format required by the old sectors.
    """
    length = len(ts.month.values)
    converted_ts = {
        "Month": [kebab_to_camel(w) for w in ts.month.values],
        "Day": [kebab_to_camel(w) for w in ts.day.values],
        "Hour": [kebab_to_camel(w) for w in ts.hour.values],
        "RepresentHours": list(ts.represent_hours.values.astype(float)),
        "SN": list(range(1, length + 1)),
        "AgLevel": [ag_level] * length,
    }
    return converted_ts


def represent_hours(
    timeslices: DataArray, nhours: Union[int, float] = 8765.82
) -> DataArray:
    """Number of hours per timeslice.

    Arguments:
        timeslices: The timeslice for which to compute the number of hours
        nhours: The total number of hours represented in the timeslice. Defaults to the
            average number of hours in year.
    """
    return convert_timeslice(DataArray([nhours]), timeslices).squeeze()


def drop_timeslice(data: DataArray) -> DataArray:
    """Drop the timeslice variable from a DataArray.

    If the array doesn't contain the timeslice variable, return the input unchanged.
    """
    if "timeslice" not in data.dims:
        return data

    return data.drop_vars(data.timeslice.indexes)


setup_module(DEFAULT_TIMESLICE_DESCRIPTION)
