"""Collection of functions and stand-alone algorithms."""
from typing import (
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Text,
    Union,
    cast,
)

from numpy import ndarray
from xarray import DataArray, Dataset


def multiindex_to_coords(data: Union[Dataset, DataArray], dimension: Text = "asset"):
    """Flattens multi-index dimension into multi-coord dimension."""
    from pandas import MultiIndex

    assert dimension in data.dims
    assert isinstance(data.indexes[dimension], MultiIndex)
    names = data.indexes[dimension].names
    coords = {n: data[n].values for n in names}
    result = data.drop_vars(dimension)
    for name, coord in coords.items():
        result[name] = dimension, coord
    if isinstance(result, Dataset):
        result = result.set_coords(names)
    return result


def coords_to_multiindex(
    data: Union[Dataset, DataArray], dimension: Text = "asset"
) -> Union[Dataset, DataArray]:
    """Creates a multi-index from flattened multiple coords."""
    from pandas import MultiIndex

    assert dimension in data.dims
    assert dimension not in data.indexes
    names = [u for u in data.coords if data[u].dims == (dimension,)]
    index = MultiIndex.from_arrays([data[u] for u in names], names=names)
    result = data.drop_vars(names)
    result[dimension] = index
    return result


def reduce_assets(
    assets: Union[DataArray, Sequence[DataArray]],
    coords: Optional[Union[Text, Sequence[Text]]] = None,
    dim: Text = "asset",
    operation: Optional[Callable] = None,
) -> DataArray:
    r"""Combine assets along given asset dimension.

    This method simplifies combining assets accross multiple agents, or combining assets
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

        >>> data = Dataset()
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

        >>> from numpy import arange
        >>> data['capacity'] = ('year', 'asset'), arange(3 * 4).reshape(3, 4)

        The point of `reduce_assets` is to aggregate assets that refer to the
        same process:

        >>> reduce_assets(data.capacity)
        <xarray.DataArray 'capacity' (year: 3, asset: 3)>
        array([[ 0,  3,  3],
               [ 4,  7, 11],
               [ 8, 11, 19]])
        Coordinates:
          * year        (year) ... 2010 2015 2017
            installed   (asset) ... 1990 1990 1991
            technology  (asset) <U1 'a' 'c' 'b'
            region      (asset) <U1 'x' 'y' 'x'
        Dimensions without coordinates: asset

        We can also specify explicitly which coordinates in the 'asset'
        dimension should be reduced, and how:

        >>> reduce_assets(
        ...     data.capacity,
        ...     coords=('technology', 'installed'),
        ...     operation = lambda x: x.mean(dim='asset')
        ... )
        <xarray.DataArray 'capacity' (year: 3, asset: 3)>
        array([[ 0. ,  1.5,  3. ],
               [ 4. ,  5.5,  7. ],
               [ 8. ,  9.5, 11. ]])
        Coordinates:
          * year        (year) ... 2010 2015 2017
            technology  (asset) <U1 'a' 'b' 'c'
            installed   (asset) ... 1990 1991 1990
        Dimensions without coordinates: asset
    """
    from copy import copy
    from numpy import array
    from xarray import concat

    if operation is None:

        def operation(x):
            return x.sum(dim)

    assert operation is not None

    if not isinstance(assets, DataArray):
        assets = concat(assets, dim=dim)
    assert isinstance(assets, DataArray)
    if coords is None:
        coords = [cast(Text, k) for k, v in assets.coords.items() if v.dims == (dim,)]
    elif isinstance(coords, Text):
        coords = (coords,)
    coords = [k for k in coords if k in assets.coords and assets[k].dims == (dim,)]
    assets = copy(assets)
    dtypes = [(d, assets[d].dtype) for d in coords]
    grouper = array(list(zip(*(assets[d].values for d in coords))), dtype=dtypes)
    assert "grouper" not in assets.coords
    assets["grouper"] = "asset", grouper
    result = operation(assets.groupby("grouper")).rename(grouper=dim)
    for i, d in enumerate(coords):
        result[d] = dim, [u[i] for u in result[dim].values]
    return result.drop_vars("asset")


def broadcast_techs(
    technologies: Union[Dataset, DataArray],
    template: Union[DataArray, Dataset],
    dimension: Text = "asset",
    interpolation: Text = "linear",
    installed_as_year: bool = True,
    **kwargs,
) -> Union[Dataset, DataArray]:
    """Broadcasts technologies to the shape of template in given dimension.

    The dimensions of the technologies are fully explicit, in that each concept
    'technology', 'region', 'year' (for year of issue) is a separate dimension.
    However, the dataset or data arrays representing other quantities, such as
    capacity, are often flattened out with coordinates 'region', 'installed',
    and 'technology' represented in a single 'asset' dimension. This latter
    representation is sparse if not all combinations of 'region', 'installed',
    and 'technology' are present, whereas the former represention makes it
    easier to select a subset of the same.

    This function broadcast the first represention to the shape and coordinates
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


def clean_assets(assets: Dataset, years: Union[int, Sequence[int]]):
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
    dataset: Union[Dataset, DataArray],
    year: Optional[Union[int, Iterable[int]]] = None,
    interpolation: Text = "linear",
    **kwargs,
) -> Union[Dataset, DataArray]:
    """Filter inputs, taking care to interpolate years."""
    from typing import Set

    if year is None:
        setyear: Set[int] = set()
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
    data: Union[Dataset, DataArray],
    template: Union[DataArray, Dataset],
    asset_dimension: Text = "asset",
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

    Returns
        `data` transformed to match the form of `template`
    """
    if asset_dimension in template.dims:
        return broadcast_techs(data, template, dimension=asset_dimension, **kwargs)

    match_indices = set(data.dims).intersection(template.dims) - set(kwargs)
    match = {d: template[d].isin(data[d]).values for d in match_indices if d != "year"}
    if "year" in match_indices:
        match["year"] = template.year.values
    return filter_input(data, **match, **kwargs)


def tupled_dimension(array: ndarray, axis: int):
    """Transforms one axis into a tuples."""
    from numpy import moveaxis, zeros

    if array.shape[axis] == 1:
        shape = tuple(j for i, j in enumerate(array.shape) if i != axis)
        return array.reshape(*shape)

    rolled = moveaxis(array, axis, -1)
    shape = rolled.shape
    flattened = rolled.reshape(-1, shape[-1])
    result = zeros(shape=flattened.shape[:-1], dtype=object)
    for i in range(0, flattened.shape[0]):
        result[i] = tuple(flattened[i, :])
    return result.reshape(*shape[:-1])


def lexical_comparison(
    objectives: Dataset,
    binsize: Dataset,
    order: Optional[Sequence[Text]] = None,
    bin_last: bool = True,
) -> DataArray:
    """Lexical comparison over the objectives.

    Lexical comparison operates by binning the objectives into bins of width
    `binsize`. Once binned, dimensions other than `asset` and `technology` are
    reduced by taking the max, e.g. the largest constraint. Finally, the
    objectives are ranked lexographically, in the order given by the parameters.

    Arguments:
        objectives: Dataset containing the objectives to rank
        binsize: bin size, minimization direction
            (+ -> minimize, - -> maximize), and (optionally) order of
            lexicographical comparison. The order is the one given
            `binsize.data_vars` if the argument `order` is None.
        order: Optional array indicating the order in which to rank the tuples.
        bin_last: Whether the last metric should be binned, or whether it
            should be left as a the type it already is (e.g. no flooring and
            no turning to integer.)

    Result:
        An array of tuples which can subsquently be compared lexicographically.
    """
    from numpy import floor

    if order is None:
        order = list(binsize.data_vars)

    assert set(order) == set(binsize.data_vars)
    assert set(order).issuperset(objectives)

    result = objectives[order]
    for name in order if bin_last else order[:-1]:
        result[name] = floor(result[name] / binsize[name]).astype(int)
    if not bin_last:
        result[order[-1]] = result[order[-1]] / binsize[order[-1]]
    return result.to_array(dim="variable").reduce(tupled_dimension, dim="variable")


def merge_assets(
    capa_a: DataArray,
    capa_b: DataArray,
    interpolation: Optional[Text] = "linear",
    dimension: Text = "asset",
) -> DataArray:
    """Merge two capacity arrays."""
    from xarray import concat

    years = sorted(set(capa_a.year.values).union(capa_b.year.values))

    levels = (coord for coord in capa_a.coords if capa_a[coord].dims == (dimension,))
    result = concat(
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
            .pipe(multiindex_to_coords, dimension=dimension)
            .rename({"asset_level_%i" % i: coord for i, coord in enumerate(levels)})
        )
    return result


def avoid_repetitions(data: DataArray, dim: Text = "year") -> DataArray:
    """list of years such that there is no repetition in the data.

    It removes the central year of any three consecutive years where all data is
    the same. This means the original data can be reobtained via a linear
    interpolation or a forward fill.

    The first and last year are always preserved.
    """
    roll = data.rolling(**{dim: 3, "center": True}).construct("window")
    years = ~(roll == roll.isel(window=0)).all([u for u in roll.dims if u != dim])
    return data.year[years]


def nametuple_to_dict(nametup: Union[Mapping, NamedTuple]) -> Mapping:
    """Transforms a nametuple of type GenericDict into an OrderDict."""
    from dataclasses import is_dataclass, asdict
    from collections import OrderedDict

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
    data: DataArray, future: DataArray, threshhold: float = 1e-12, dim: Text = "year"
) -> DataArray:
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
        values differed for the current year beyond a given threshhold:

        >>> from muse.utilities import future_propagation
        >>> future_propagation(data, future, threshhold=0.1)
        <xarray.DataArray (fuel: 2, year: 4)>
        array([[ 0. ,  1.2,  1.2,  1.2],
               [-5. , -4. , -3. , -2. ]])
        Coordinates:
          * year     (year) ... 2020 2025 2030 2035
          * fuel     (fuel) <U4 'gas' 'coal'

        Above, the data for coal is not sufficiently different given the threshhold.
        hence, the future values for coal remain as they where.

        The dimensions of ``future`` do not have to match exactly those of ``data``.
        Standard broadcasting is used if they do not match:

        >>> future_propagation(data, future.sel(fuel="gas", drop=True), threshhold=0.1)
        <xarray.DataArray (fuel: 2, year: 4)>
        array([[ 0. ,  1.2,  1.2,  1.2],
               [-5. ,  1.2,  1.2,  1.2]])
        Coordinates:
          * year     (year) ... 2020 2025 2030 2035
          * fuel     (fuel) <U4 'gas' 'coal'
        >>> future_propagation(data, future.sel(fuel="coal", drop=True), threshhold=0.1)
        <xarray.DataArray (fuel: 2, year: 4)>
        array([[ 0.  , -3.95, -3.95, -3.95],
               [-5.  , -4.  , -3.  , -2.  ]])
        Coordinates:
          * year     (year) ... 2020 2025 2030 2035
          * fuel     (fuel) <U4 'gas' 'coal'
    """
    from numpy import abs, logical_or

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
        logical_or(data.year < year, abs(data.loc[{dim: year}] - future) < threshhold),
        future,
    )
