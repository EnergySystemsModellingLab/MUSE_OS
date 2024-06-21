"""Collection of functions and stand-alone algorithms."""

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import numpy as np
import xarray as xr


def multiindex_to_coords(
    data: Union[xr.Dataset, xr.DataArray], dimension: str = "asset"
):
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
    data: Union[xr.Dataset, xr.DataArray], dimension: str = "asset"
) -> Union[xr.Dataset, xr.DataArray]:
    """Creates a multi-index from flattened multiple coords."""
    from pandas import MultiIndex

    assert dimension in data.dims
    assert dimension not in data.indexes
    names = [u for u in data.coords if data[u].dims == (dimension,)]
    index = MultiIndex.from_arrays([data[u].values for u in names], names=names)
    mindex_coords = xr.Coordinates.from_pandas_multiindex(index, dimension)
    return data.drop_vars(names).assign_coords(mindex_coords)


def reduce_assets(
    assets: Union[xr.DataArray, xr.Dataset, Sequence[Union[xr.Dataset, xr.DataArray]]],
    coords: Optional[Union[str, Sequence[str], Iterable[str]]] = None,
    dim: str = "asset",
    operation: Optional[Callable] = None,
) -> Union[xr.DataArray, xr.Dataset]:
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
    from copy import copy

    if operation is None:

        def operation(x):
            return x.sum(dim)

    assert operation is not None

    if not isinstance(assets, (xr.Dataset, xr.DataArray)):
        assets = xr.concat(assets, dim=dim)
    assert isinstance(assets, (xr.Dataset, xr.DataArray))
    if assets[dim].size == 0:
        return assets
    if coords is None:
        coords = [cast(str, k) for k, v in assets.coords.items() if v.dims == (dim,)]
    elif isinstance(coords, str):
        coords = (coords,)
    coords = [k for k in coords if k in assets.coords and assets[k].dims == (dim,)]
    assets = copy(assets)
    dtypes = [(d, assets[d].dtype) for d in coords]
    grouper = np.array(
        list(zip(*(cast(Iterator, assets[d].values) for d in coords))), dtype=dtypes
    )
    assert "grouper" not in assets.coords
    assets["grouper"] = "asset", grouper
    result = operation(assets.groupby("grouper")).rename(grouper=dim)
    for i, d in enumerate(coords):
        result[d] = dim, [u[i] for u in result[dim].values]
    return result.drop_vars("asset")


def broadcast_techs(
    technologies: Union[xr.Dataset, xr.DataArray],
    template: Union[xr.DataArray, xr.Dataset],
    dimension: str = "asset",
    interpolation: str = "linear",
    installed_as_year: bool = True,
    **kwargs,
) -> Union[xr.Dataset, xr.DataArray]:
    """Broadcasts technologies to the shape of template in given dimension.

    The dimensions of the technologies are fully explicit, in that each concept
    'technology', 'region', 'year' (for year of issue) is a separate dimension.
    However, the dataset or data arrays representing other quantities, such as
    capacity, are often flattened out with coordinates 'region', 'installed',
    and 'technology' represented in a single 'asset' dimension. This latter
    representation is sparse if not all combinations of 'region', 'installed',
    and 'technology' are present, whereas the former representation makes it
    easier to select a subset of the same.

    This function broadcast the first representation to the shape and coordinates
    of the second.

    Arguments:
        technologies: The dataset to broadcast
        template: the dataset or data-array to use as a template
        dimension: the name of the dimensiom from `template` over which to
            broadcast
        interpolation: interpolation method used across `year`
        installed_as_year: if the coordinate `installed` exists, then it is
            applied to the `year` dimension of the technologies dataset
        kwargs: further arguments are used initial filters over the
            `technologies` dataset.
    """
    # this assert will trigger if 'year' is changed to 'installed' in
    # technologies, because then this function should be modified.
    assert "installed" not in technologies.dims
    names = [u for u in template.coords if template[u].dims == (dimension,)]
    # the first selection reduces the size of technologies without affecting the
    # dimensions.
    first_sel = {
        n: technologies[n].isin(template[n])
        for n in names
        if n in technologies.dims and n != "year"
    }
    first_sel.update({k: v for k, v in kwargs.items() if k != "year"})
    techs = technologies.sel(first_sel)

    if "year" in technologies.dims:
        year = None
        if installed_as_year and "installed" in names:
            year = template["installed"]
        elif (not installed_as_year) and "year" in template.dims:
            year = template["year"]
        if year is not None and len(year) > 0:
            techs = techs.interp(
                year=sorted(set(cast(Iterable, year.values))), method=interpolation
            )
        if installed_as_year and "installed" in names:
            techs = techs.rename(year="installed")

    second_sel = {n: template[n] for n in template.coords if n in techs.dims}

    return techs.sel(second_sel)


def clean_assets(assets: xr.Dataset, years: Union[int, Sequence[int]]):
    """Cleans up and prepares asset for current iteration.

    - adds current and forecast year by backfilling missing entries
    - removes assets for which there is no capacity now or in the future
    """
    if isinstance(years, Sequence):
        current = min(*years)
        years = sorted(set(assets.year[assets.year >= current].values).union(years))
    else:
        x = set(assets.year[assets.year >= years].values)
        x.add(years)
        years = sorted(x)
    result = assets.reindex(year=years, method="backfill").fillna(0)
    not_asset = [u for u in result.dims if u != "asset"]
    return result.sel(asset=result.capacity.any(not_asset))


def filter_input(
    dataset: Union[xr.Dataset, xr.DataArray],
    year: Optional[Union[int, Iterable[int]]] = None,
    interpolation: str = "linear",
    **kwargs,
) -> Union[xr.Dataset, xr.DataArray]:
    """Filter inputs, taking care to interpolate years."""
    if year is None:
        setyear: set[int] = set()
    else:
        try:
            setyear = {int(year)}  # type: ignore
        except TypeError:
            setyear = set(int(u) for u in year)  # type: ignore
    withyear = (
        "year" in dataset.dims
        and year is not None
        and setyear.issubset(dataset.year.values)
    )
    if withyear:
        kwargs["year"] = year
        year = None
    dataset = dataset.sel(**kwargs)
    if withyear and "year" not in dataset.dims and "year" in dataset.coords:
        dataset = dataset.drop_vars("year")

    if "year" in dataset.dims and year is not None:
        dataset = dataset.interp(year=year, method=interpolation)
        if "year" not in dataset.dims and "year" in dataset.coords:
            dataset = dataset.drop_vars("year")
        elif "year" in dataset.dims:
            dataset = dataset.ffill("year")
    return dataset


def filter_with_template(
    data: Union[xr.Dataset, xr.DataArray],
    template: Union[xr.DataArray, xr.Dataset],
    asset_dimension: str = "asset",
    **kwargs,
):
    """Filters data to match template.

    If the `asset_dimension` is present in `template.dims`, then the call is
    forwarded to `broadcast_techs`. Otherwise, the set of dimensions and indices
    in common between `template` and `data` are determined, and the resulting
    call is forwarded to `filter_input`.

    Arguments:
        data: Data to transform
        template: Data from which to figure coordinates and dimensions
        asset_dimension: Name of the dimension which if present indicates the
            format is that of an *asset* (see `broadcast_techs`)
        kwargs: passed on to `broadcast_techs` or `filter_input`

    Returns:
        `data` transformed to match the form of `template`
    """
    if asset_dimension in template.dims:
        return broadcast_techs(data, template, dimension=asset_dimension, **kwargs)

    match_indices = set(data.dims).intersection(template.dims) - set(kwargs)
    match = {d: template[d].isin(data[d]).values for d in match_indices if d != "year"}
    if "year" in match_indices:
        match["year"] = template.year.values
    return filter_input(data, **match, **kwargs)  # type: ignore


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
    order: Optional[Sequence[Hashable]] = None,
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
    interpolation: str = "linear",
    dimension: str = "asset",
) -> xr.DataArray:
    """Merge two capacity arrays."""
    years = sorted(set(capa_a.year.values).union(capa_b.year.values))

    if len(capa_a.year) == 1:
        result = xr.concat(
            (
                capa_a,
                capa_b.interp(year=years, method=interpolation).fillna(0),
            ),
            dim=dimension,
        ).fillna(0)
    elif len(capa_b.year) == 1:
        result = xr.concat(
            (
                capa_a.interp(year=years, method=interpolation).fillna(0),
                capa_b,
            ),
            dim=dimension,
        ).fillna(0)
    else:
        result = xr.concat(
            (
                capa_a.interp(year=years, method=interpolation).fillna(0),
                capa_b.interp(year=years, method=interpolation).fillna(0),
            ),
            dim=dimension,
        )
    forgroup = result.pipe(coords_to_multiindex, dimension=dimension)
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

    The first and last year are always preserved.
    """
    roll = data.rolling({dim: 3}, center=True).construct("window")
    years = ~(roll == roll.isel(window=0)).all([u for u in roll.dims if u != dim])
    return data.year[years]


def nametuple_to_dict(nametup: Union[Mapping, NamedTuple]) -> Mapping:
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
    dim: str = "year",
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
    if dim not in data.dims or dim not in future.coords:
        raise ValueError("Expected dimension 'year' in `data` and `future`.")
    if future[dim].ndim != 0 and len(future[dim]) != 1:
        raise ValueError(f'``future["{dim}"] should be of length 1 or a scalar.')
    if not future[dim].isin(data[dim]).all():
        raise ValueError(f'{future[dim]} not found in data["{dim}"].')

    if future[dim].ndim != 0:
        future = future.loc[{dim: 0}]
    year = future[dim].values
    return data.where(
        np.logical_or(
            data.year < year, np.abs(data.loc[{dim: year}] - future) < threshold
        ),
        future,
    )


def agent_concatenation(
    data: Mapping[Hashable, Union[xr.DataArray, xr.Dataset]],
    dim: str = "asset",
    name: str = "agent",
    fill_value: Any = 0,
) -> Union[xr.DataArray, xr.Dataset]:
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


def aggregate_technology_model(
    data: Union[xr.DataArray, xr.Dataset],
    dim: str = "asset",
    drop: Union[str, Sequence[str]] = "installed",
) -> Union[xr.DataArray, xr.Dataset]:
    """Aggregate together assets with the same installation year.

    The assets of a given agent, region, and technology but different installation year
    are grouped together and summed over.

    Example:
        We first create a random set of agent assets and aggregate them.
        Some of these agents own assets from the same technology but potentially with
        different installation year. This function will aggregate together all assets
        of a given agent with same technology.

        >>> from muse.examples import random_agent_assets
        >>> from muse.utilities import agent_concatenation, aggregate_technology_model
        >>> rng = np.random.default_rng(1234)
        >>> agent_assets = {i: random_agent_assets(rng) for i in range(5)}
        >>> assets = agent_concatenation(agent_assets)
        >>> reduced = aggregate_technology_model(assets)

        We can check that the tuples (agent, technology) are unique (each agent works in
        a single region):

        >>> ids = list(zip(reduced.agent.values, reduced.technology.values))
        >>> assert len(set(ids)) == len(ids)

        And we can check they correspond to the right summation:

        >>> for agent, technology in set(ids):
        ...     techsel = assets.technology == technology
        ...     agsel = assets.agent == agent
        ...     expected = assets.sel(asset=techsel & agsel).sum("asset")
        ...     techsel = reduced.technology == technology
        ...     agsel = reduced.agent == agent
        ...     actual = reduced.sel(asset=techsel & agsel)
        ...     assert len(actual.asset) == 1
        ...     assert (actual == expected).all()
    """
    if isinstance(drop, str):
        drop = (drop,)
    return reduce_assets(
        data,
        [cast(str, u) for u in data.coords if u not in drop and data[u].dims == (dim,)],
    )
