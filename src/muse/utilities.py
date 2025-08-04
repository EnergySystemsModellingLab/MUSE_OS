"""Collection of functions and stand-alone algorithms."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    Callable,
    NamedTuple,
    cast,
)

import numpy as np
import xarray as xr


def multiindex_to_coords(data: xr.Dataset | xr.DataArray, dimension: str = "asset"):
    """Flattens multi-index dimension into multi-coord dimension."""
    from pandas import MultiIndex

    assert dimension in data.dims
    assert isinstance(data.indexes[dimension], MultiIndex)
    names = data.indexes[dimension].names
    coords = {n: data[n].values for n in names}
    result = data.drop_vars([dimension, *names])
    for name, coord in coords.items():
        result[name] = dimension, coord
    if isinstance(result, xr.Dataset):
        result = result.set_coords(names)
    return result


def coords_to_multiindex(
    data: xr.Dataset | xr.DataArray, dimension: str = "asset"
) -> xr.Dataset | xr.DataArray:
    """Creates a multi-index from flattened multiple coords."""
    from pandas import MultiIndex

    assert dimension in data.dims
    assert dimension not in data.indexes
    names = [u for u in data.coords if data[u].dims == (dimension,)]
    index = MultiIndex.from_arrays([data[u].values for u in names], names=names)
    mindex_coords = xr.Coordinates.from_pandas_multiindex(index, dimension)
    return data.drop_vars(names).assign_coords(mindex_coords)


def reduce_assets(
    assets: xr.DataArray | xr.Dataset | Sequence[xr.Dataset | xr.DataArray],
    coords: str | Sequence[str] | Iterable[str] | None = None,
    dim: str = "asset",
    operation: Callable | None = None,
) -> xr.DataArray | xr.Dataset:
    r"""Combine assets along given asset dimension.

    This method simplifies combining assets across multiple agents, or combining assets
    across a given dimension. By default, it will sum together assets from the same
    region which have the same technology and the same installation date. In other
    words, assets are identified by the technology, installation year and region. The
    reduction happens over other possible coordinates, e.g. the owning agent.

    More specifically, assets are often indexed using what xarray calls a **dimension
    without coordinates**. In practice, there are still coordinates associated with
    assets, e.g. *technology* and *installed* (installation year or version), but the
    value associated with these coordinates are not unique.  There may be more than one
    asset with the same technology or installation year.

    For instance, with assets per agent defined as :math:`A^{i, r}_o`, with :math:`i` an
    agent index, :math:`r` a region, :math:`o` is the coordinates identified in
    ``coords``, and :math:`T` the transformation identified by ``operation``, then this
    function computes:

    .. math::

        R_{r, o} = T[\{A^{i, r}_o; \forall i\}]

    If :math:`T` is the sum operation, then:

    .. math::

        R_{r, o} = \sum_i  A^{i, r}_o


    Example:
        Lets construct assets that do have duplicates assets. First we construct the
        dimensions, using fake data:

        >>> data = xr.Dataset()
        >>> data['year'] = 'year', [2010, 2015, 2017]
        >>> data['installed'] = 'asset', [1990, 1991, 1991, 1990]
        >>> data['technology'] = 'asset', ['a', 'b', 'b', 'c']
        >>> data['region'] = 'asset', ['x', 'x', 'x', 'y']
        >>> data = data.set_coords(('installed', 'technology', 'region'))

        We can check there are duplicate assets in this coordinate system:

        >>> processes = set(
        ...     zip(data.installed.values, data.technology.values, data.region.values)
        ... )
        >>> len(processes) < len(data.asset)
        True

        Now we can easily create a fake two dimensional quantity per process and
        per year:

        >>> data['capacity'] = ('year', 'asset'), np.arange(3 * 4).reshape(3, 4)

        The point of `reduce_assets` is to aggregate assets that refer to the
        same process:

        >>> reduce_assets(data.capacity)  # doctest: +SKIP
        <xarray.DataArray 'capacity' (year: 3, asset: 3)> Size: 36B
        array([[ 0,  3,  3],
               [ 4,  7, 11],
               [ 8, 11, 19]])
        Coordinates:
          * year        (year) int32 12B 2010 2015 2017
            installed   (asset) int32 12B 1990 1990 1991
            technology  (asset) <U1 12B 'a' 'c' 'b'
            region      (asset) <U1 12B 'x' 'y' 'x'
        Dimensions without coordinates: asset

        We can also specify explicitly which coordinates in the 'asset'
        dimension should be reduced, and how:

        >>> reduce_assets(
        ...     data.capacity,
        ...     coords=('technology', 'installed'),
        ...     operation = lambda x: x.mean(dim='asset')
        ... )  # doctest: +SKIP
        <xarray.DataArray 'capacity' (year: 3, asset: 3)> Size: 72B
        array([[ 0. ,  1.5,  3. ],
               [ 4. ,  5.5,  7. ],
               [ 8. ,  9.5, 11. ]])
        Coordinates:
          * year        (year) int32 12B 2010 2015 2017
            technology  (asset) <U1 12B 'a' 'b' 'c'
            installed   (asset) int32 12B 1990 1991 1990
        Dimensions without coordinates: asset
    """
    from copy import deepcopy

    assets = deepcopy(assets)

    if operation is None:

        def operation(x):
            return x.sum(dim)

    assert operation is not None

    # Concatenate assets if a sequence is given
    if not isinstance(assets, (xr.Dataset, xr.DataArray)):
        assets = xr.concat(assets, dim=dim)
    assert isinstance(assets, (xr.Dataset, xr.DataArray))

    # If there are no assets, nothing needs to be done
    if assets[dim].size == 0:
        return assets

    # Coordinates to reduce over (e.g. technology, installed)
    if coords is None:
        coords = [cast(str, k) for k, v in assets.coords.items() if v.dims == (dim,)]
    elif isinstance(coords, str):
        coords = (coords,)
    coords = [k for k in coords if k in assets.coords and assets[k].dims == (dim,)]

    # Create a new dimension to group by
    dtypes = [(d, assets[d].dtype) for d in coords]
    grouper = np.array(
        list(zip(*(cast(Iterator, assets[d].values) for d in coords))), dtype=dtypes
    )
    assert "grouper" not in assets.coords
    assets["grouper"] = "asset", grouper

    # Perform the operation
    result = operation(assets.groupby("grouper")).rename(grouper=dim)
    for i, d in enumerate(coords):
        result[d] = dim, [u[i] for u in result[dim].values]
    return result.drop_vars("asset")


def broadcast_over_assets(
    data: xr.Dataset | xr.DataArray,
    template: xr.DataArray | xr.Dataset,
    installed_as_year: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Broadcasts an array to the shape of a template containing asset-level data.

    The dimensions of many arrays (such as technology datasets) are fully explicit, in
    that each concept (e.g. 'technology', 'region', 'year') is a separate dimension.
    However, other datasets (e.g capacity), are presented on a per-asset basis,
    containing a single 'asset' dimension with with coordinates such as 'region',
    'installed' (year of installation), and 'technology'. This latter representation is
    sparse if not all combinations of 'region', 'installed' and 'technology' are
    present.

    This function broadcasts the first representation to the shape and coordinates
    of the second, selecting the appropriate values for each asset (see example below).

    Note: this is not necessarily limited to technology datasets. For
    example, it could also be used on a dataset of commodity prices to select prices
    relevant to each asset (e.g. if assets exist in multiple regions).

    Arguments:
        data: The dataset/data-array to broadcast
        template: The dataset/data-array to use as a template
        installed_as_year: True means that the "year" dimension in 'data`
            corresponds to the year that the asset was installed. This will commonly
            be the case for most technology parameters (e.g. var_par/fix_par are
            specified the year that an asset is installed, and fixed for the lifetime of
            the asset). In this case, `data` must have a year coordinate for every
            possible "installed" year in the template.

            Conversely, if the values in `data` apply to the year of activity, rather
            than the year of installation, `installed_as_year` should be False.
            An example would be commodity prices, which can change over the lifetime
            of an asset. In this case, if "year" is present as a dimension in `data`,
            it will be maintained as a separate dimension in the output.

    Example:
        Define the data array:
        >>> import xarray as xr
        >>> technologies = xr.DataArray(
        ...     data=[[1, 2, 3], [4, 5, 6]],
        ...     dims=['technology', 'region'],
        ...     coords={'technology': ['gasboiler', 'heatpump'],
        ...             'region': ['R1', 'R2', 'R3']},
        ... )

        This array contains a value for every combination of technology and region (e.g.
        this could refer to the efficiency of each technology in each region). For
        simplicity, we are not including a "year" dimension in this example.

        Define the assets template:
        >>> assets = xr.DataArray(
        ...     data=[10, 50],
        ...     dims=["asset"],
        ...     coords={
        ...         "region": (["asset"], ["R1", "R2"]),
        ...         "technology": (["asset"], ["gasboiler", "heatpump"])},
        ... )

        We have two assets: a gas boiler in region R1 and a heat pump in region R2. In
        this case the values don't matter, but could correspond to the installed
        capacity of each asset, for example.

        We want to select the values from the technology array that correspond to each
        asset in the template. To do this, we perform `broadcast_over_assets` on
        `technologies` using `assets` as a template:
        >>> broadcast_over_assets(technologies, assets, installed_as_year=False)
        <xarray.DataArray (asset: 2)> Size: 16B
        array([1, 5])
        Coordinates:
            technology  (asset) <U9 72B 'gasboiler' 'heatpump'
            region      (asset) <U2 16B 'R1' 'R2'
        Dimensions without coordinates: asset

        The output array has an "asset" dimension which matches the template. Each value
        in the output is the value in the original technology array that matches the
        technology & region of each asset.
    """
    # TODO: this will return `data` unchanged if the template has no "asset"
    # dimension, but strictly speaking we shouldn't allow this.
    # assert "asset" in template.dims

    # Name of asset coordinates (e.g. "technology", "region", "installed")
    names = [u for u in template.coords if template[u].dims == ("asset",)]
    assert "year" not in names

    # If installed_as_year is True, we need to rename the installed dimension to "year"
    # TODO: this should be stricter, and enforce that the template has "installed" data,
    # and that the technologies dataset has a "year" dimension.
    # if installed_as_year:
    if installed_as_year and "installed" in names and "year" in data.dims:
        # assert "installed" in names
        data = data.rename(year="installed")

    # The first selection reduces the size of the data without affecting the
    # dimensions.
    first_sel = {n: data[n].isin(template[n]) for n in names if n in data.dims}
    techs = data.sel(first_sel)

    # Reshape the array to match the template
    second_sel = {n: template[n] for n in template.coords if n in techs.dims}
    return techs.sel(second_sel)


def clean_assets(assets: xr.Dataset, year: int):
    """Cleans up and prepares asset for current iteration.

    - removes data from before the specified year
    - removes assets for which there is no capacity now or in the future
    """
    assets = assets.sel(year=slice(year, None))
    assets = assets.where(assets.capacity.any(dim="year"), drop=True)
    return assets


def tupled_dimension(array: np.ndarray, axis: int):
    """Transforms one axis into a tuples."""
    if array.shape[axis] == 1:
        shape = tuple(j for i, j in enumerate(array.shape) if i != axis)
        return array.reshape(*shape)

    rolled = np.moveaxis(array, axis, -1)
    shape = rolled.shape
    flattened = rolled.reshape(-1, shape[-1])
    result = np.zeros(shape=flattened.shape[:-1], dtype=object)
    for i in range(0, flattened.shape[0]):
        result[i] = tuple(flattened[i, :])
    return result.reshape(*shape[:-1])


def lexical_comparison(
    objectives: xr.Dataset,
    binsize: xr.Dataset,
    order: Sequence[Hashable] | None = None,
    bin_last: bool = True,
) -> xr.DataArray:
    """Lexical comparison over the objectives.

    Lexical comparison operates by binning the objectives into bins of width
    `binsize`. Once binned, dimensions other than `asset` and `technology` are
    reduced by taking the max, e.g. the largest constraint. Finally, the
    objectives are ranked lexographically, in the order given by the parameters.

    Arguments:
        objectives: xr.Dataset containing the objectives to rank
        binsize: bin size, minimization direction
            (+ -> minimize, - -> maximize), and (optionally) order of
            lexicographical comparison. The order is the one given
            `binsize.data_vars` if the argument `order` is None.
        order: Optional array indicating the order in which to rank the tuples.
        bin_last: Whether the last metric should be binned, or whether it
            should be left as a the type it already is (e.g. no flooring and
            no turning to integer.)

    Result:
        An array of tuples which can subsequently be compared lexicographically.
    """
    if order is None:
        order = [u for u in binsize.data_vars]

    assert set(order) == set(binsize.data_vars)
    assert set(order).issuperset(objectives)

    result = objectives[order]
    for name in order if bin_last else order[:-1]:
        result[name] = np.floor(result[name] / binsize[name]).astype(int)
    if not bin_last:
        result[order[-1]] = result[order[-1]] / binsize[order[-1]]
    return result.to_array(dim="variable").reduce(tupled_dimension, dim="variable")


def merge_assets(
    capa_a: xr.DataArray,
    capa_b: xr.DataArray,
    dimension: str = "asset",
) -> xr.DataArray:
    """Merge two capacity arrays."""
    # Interpolate capacity arrays to a common time framework
    years = sorted(set(capa_a.year.values).union(capa_b.year.values))
    if len(capa_a.year) == 1:
        capa_a_interp = capa_a
        capa_b_interp = interpolate_capacity(capa_b, year=years)
    elif len(capa_b.year) == 1:
        capa_a_interp = interpolate_capacity(capa_a, year=years)
        capa_b_interp = capa_b
    else:
        capa_a_interp = interpolate_capacity(capa_a, year=years)
        capa_b_interp = interpolate_capacity(capa_b, year=years)

    # Concatenate the two capacity arrays
    result = xr.concat((capa_a_interp, capa_b_interp), dim=dimension)

    #
    forgroup = result.pipe(coords_to_multiindex, dimension=dimension)
    if isinstance(forgroup, xr.DataArray):
        forgroup = forgroup.to_dataset()
    if len(forgroup[dimension]) != len(set(forgroup[dimension].values)):
        result = (
            forgroup.groupby(dimension)
            .sum(dimension)
            .clip(min=0)
            .pipe(multiindex_to_coords, dimension=dimension)
        )
    return result


def avoid_repetitions(data: xr.DataArray, dim: str = "year") -> xr.DataArray:
    """List of years such that there is no repetition in the data.

    It removes the central year of any three consecutive years where all data is
    the same. This means the original data can be reobtained via a linear
    interpolation or a forward fill.
    See :py:func:`muse.utilities.interpolate_capacity`.

    The first and last year are always preserved.
    """
    roll = data.rolling({dim: 3}, center=True).construct("window")
    years = ~(roll == roll.isel(window=0)).all([u for u in roll.dims if u != dim])
    return data.year[years]


def interpolate_capacity(
    data: xr.DataArray, year: int | Sequence[int] | xr.DataArray
) -> xr.DataArray:
    """Interpolates capacity data to the given years.

    Capacity between years is interpolated linearly. Capacity outside the range of the
    data is set to zero.

    Arguments:
        data: DataArray containing the capacity data
        year: Year or years to interpolate to
    """
    return data.interp(
        year=year,
        method="linear",
        kwargs={"fill_value": 0.0},
    )


def nametuple_to_dict(nametup: Mapping | NamedTuple) -> Mapping:
    """Transforms a nametuple of type GenericDict into an OrderDict."""
    from collections import OrderedDict
    from dataclasses import asdict, is_dataclass

    if is_dataclass(nametup):
        out = asdict(nametup, OrderedDict)  # type: ignore
    elif hasattr(nametup, "_asdict"):
        out = nametup._asdict()  # type: ignore
    else:
        out = nametup.copy()  # type: ignore
    for key, value in out.items():
        if is_dataclass(value) or hasattr(value, "_asdict"):
            out[key] = nametuple_to_dict(value)
    return out


def future_propagation(
    data: xr.DataArray,
    future: xr.DataArray,
    threshold: float = 1e-12,
) -> xr.DataArray:
    """Propagates values into the future.

    Example:
        ``Data`` should be an array with at least one dimension, "year":

        >>> coords = dict(year=list(range(2020, 2040, 5)), fuel=["gas", "coal"])
        >>> data = xr.DataArray(
        ...     [list(range(4)), list(range(-5, -1))],
        ...     coords=coords,
        ...     dims=("fuel", "year")
        ... )

        ``future`` is an array with  exactly one year in its ``year`` coordinate, or
        that coordinate must correspond to a scalar. That one year should also be
        present in ``data``.

        >>> future = xr.DataArray(
        ...     [1.2, -3.95], coords=dict(fuel=coords['fuel'], year=2025), dims="fuel",
        ... )

        This function propagates into ``data`` values from ``future``, but only if those
        values differed for the current year beyond a given threshold:

        >>> from muse.utilities import future_propagation
        >>> future_propagation(data, future, threshold=0.1)  # doctest: +SKIP
        <xarray.DataArray (fuel: 2, year: 4)> Size: 64B
        array([[ 0. ,  1.2,  1.2,  1.2],
               [-5. , -4. , -3. , -2. ]])
        Coordinates:
          * year     (year) int32 16B 2020 2025 2030 2035
          * fuel     (fuel) <U4 32B 'gas' 'coal'

        Above, the data for coal is not sufficiently different given the threshold.
        hence, the future values for coal remain as they where.

        The dimensions of ``future`` do not have to match exactly those of ``data``.
        Standard broadcasting is used if they do not match:

        >>> future_propagation(
        ...    data, future.sel(fuel="gas", drop=True), threshold=0.1
        ... )  # doctest: +SKIP
        <xarray.DataArray (fuel: 2, year: 4)> Size: 64B
        array([[ 0. ,  1.2,  1.2,  1.2],
               [-5. ,  1.2,  1.2,  1.2]])
        Coordinates:
          * year     (year) int32 16B 2020 2025 2030 2035
          * fuel     (fuel) <U4 32B 'gas' 'coal'
        >>> future_propagation(
        ...     data, future.sel(fuel="coal", drop=True), threshold=0.1
        ... )  # doctest: +SKIP
        <xarray.DataArray (fuel: 2, year: 4)> Size: 64B
        array([[ 0.  , -3.95, -3.95, -3.95],
               [-5.  , -4.  , -3.  , -2.  ]])
        Coordinates:
          * year     (year) int32 16B 2020 2025 2030 2035
          * fuel     (fuel) <U4 32B 'gas' 'coal'
    """
    if "year" not in data.dims or "year" not in future.coords:
        raise ValueError("Expected dimension 'year' in `data` and `future`.")
    if future["year"].ndim != 0 and len(future["year"]) != 1:
        raise ValueError('``future["year"] should be of length 1 or a scalar.')
    if not future["year"].isin(data["year"]).all():
        raise ValueError(f'{future["year"]} not found in data["year"].')

    year = future["year"].item()
    if "year" in future.dims:
        future = future.sel(year=year, drop=True)
    return data.where(
        np.logical_or(
            data.year < year, np.abs(data.sel(year=year) - future) < threshold
        ),
        future,
    )


def agent_concatenation(
    data: Mapping[Hashable, xr.DataArray | xr.Dataset],
    dim: str = "asset",
    name: str = "agent",
    fill_value: Any = 0,
) -> xr.DataArray | xr.Dataset:
    """Concatenates input map along given dimension.

    Example:
        Lets create sets of random assets to work with. We set the seed so that this
        test can be reproduced exactly.

        >>> from muse.examples import random_agent_assets
        >>> rng = np.random.default_rng(1234)
        >>> assets = {i: random_agent_assets(rng) for i in range(5)}

        The concatenation will create a new dataset (or datarray) combining all the
        inputs along the dimension "asset". The origin of each datum is retained in a
        new coordinate "agent" with dimension "asset".

        >>> from muse.utilities import agent_concatenation
        >>> aggregate = agent_concatenation(assets)
        >>> aggregate # doctest: +SKIP
        <xarray.Dataset> Size: 4kB
        Dimensions:     (asset: 19, year: 12)
        Coordinates:
            agent       (asset) int32 76B 0 0 0 0 0 1 1 1 2 2 2 2 3 3 3 4 4 4 4
          * year        (year) int64 96B 2033 2035 2036 2037 ... 2046 2047 2048 2049
            installed   (asset) int64 152B 2030 2025 2030 2010 ... 2025 2030 2010 2025
            technology  (asset) <U9 684B 'oven' 'stove' 'oven' ... 'oven' 'thermomix'
            region      (asset) <U9 684B 'Brexitham' 'Brexitham' ... 'Brexitham'
        Dimensions without coordinates: asset
        Data variables:
            capacity    (asset, year) float64 2kB 26.0 26.0 26.0 56.0 ... 62.0 62.0 62.0

        Note that the `dtype` of the capacity has changed from integers to floating
        points. This is due to how ``xarray`` performs the operation.

        We can check that all the data from each agent is indeed present in the
        aggregate.

        >>> for agent, inventory in assets.items():
        ...    assert (aggregate.sel(asset=aggregate.agent == agent) == inventory).all()

        However, it should be noted that the data is not always strictly equivalent:
        dimensions outside of "assets" (most notably "year") will include all points
        from all agents. Missing values for the "year" dimension are forward filled (and
        backfilled with zeros). Others are left with "NaN".
    """
    from itertools import repeat

    data = {k: v.copy() for k, v in data.items()}
    for key, datum in data.items():
        if name in datum.coords:
            raise ValueError(f"Coordinate {name} already exists")
        if len(data) == 1 and isinstance(datum, xr.DataArray):
            data[key] = datum.assign_coords(
                {
                    name: (
                        dim,
                        list(repeat(key, datum.sizes[dim])),
                    )
                }
            )
        else:
            datum[name] = key
    result = xr.concat(data.values(), dim=dim)
    if isinstance(result, xr.Dataset):
        result = result.set_coords("agent")
    if "year" in result.dims:
        result = result.ffill("year")
    if fill_value is not np.nan:
        result = result.fillna(fill_value)
    return result


def check_dimensions(
    data: xr.DataArray | xr.Dataset,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
):
    """Ensure that an array has the required dimensions.

    This will check that all required dimensions are present, and that no other
    dimensions are present, apart from those listed as optional.

    Args:
        data: DataArray or Dataset to check dimensions of
        required: List of dimension names that must be present
        optional: List of dimension names that may be present
    """
    present = set(data.dims)
    missing = set(required) - present
    if missing:
        raise ValueError(f"Missing required dimensions: {missing}")
    extra = present - set(required) - set(optional)
    if extra:
        raise ValueError(f"Extra dimensions: {extra}")


def interpolate_technodata(
    data: xr.Dataset,
    time_framework: list[int],
    interpolation_mode: str = "linear",
) -> xr.Dataset:
    """Interpolates technologies data to a given time framework.

    The approach depends on the format of the data:
    - If the original data contains data for more than one year, then data will be
    interpolated with flat back/forward extension to cover the time period.
    - If the original data does not have a "year" dimension, or only has data for a
    single year, then this data will not be modified or duplicated, but a dummy
    "year" dimension covering the time framework will be added.

    In both cases, data for any year in the time framework can be selected
    using data.sel(year=year) on the returned dataset.

    Args:
        data: Dataset to interpolate
        time_framework: List of years to interpolate to
        interpolation_mode: Interpolation mode to use. Must be one of:
            "linear", "nearest", "zero", "slinear", "quadratic", "cubic"

    Returns:
        Dataset with the data interpolated to the time framework, sorted by year.
    """
    # If the data has only one year then no interpolation is needed,  we just need to
    # add a year dimension with the time framework
    if "year" not in data.dims:
        data = data.copy()
        data["year"] = ("year", time_framework)
        return data

    if len(data.year) == 1:
        data = data.isel(year=0, drop=True)
        data["year"] = ("year", time_framework)
        return data

    # Sort the data by year, just in case it hasn't already been sorted
    data = data.sortby("year")

    # Flat forward extrapolation
    maxyear = max(time_framework)
    if data.year.max() < maxyear:
        years = [*data.year.values.tolist(), maxyear]
        data = data.reindex(year=years, method="ffill")

    # Flat backward extrapolation
    minyear = min(time_framework)
    if data.year.min() > minyear:
        years = [minyear, *data.year.values.tolist()]
        data = data.reindex(year=years, method="bfill")

    # Interpolation to fill gaps
    years = sorted(set(time_framework).union(data.year.values.tolist()))
    data = data.interp(year=years, method=interpolation_mode)
    return data


def camel_to_snake(name: str) -> str:
    """Transforms CamelCase to snake_case."""
    from re import sub

    pattern = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    result = sub("([a-z0-9])([A-Z])", r"\1_\2", pattern).lower()
    result = result.replace("co2", "CO2")
    result = result.replace("ch4", "CH4")
    result = result.replace("n2_o", "N2O")
    result = result.replace("f-gases", "F-gases")
    return result
