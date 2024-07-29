"""Sinks where output quantities can be stored.

Sinks take as argument a DataArray and store it somewhere. Additionally they
take a dictionary as argument. This dictionary will always contains the items
('quantity', 'sector', 'year') referring to the name of the quantity, the name
of the calling sector, the current year. They may contain additional parameters
which depend on the actual sink, such as 'filename'.

Optionally, a description of the storage (filename, etc) can be returned.

The signature of a sink is:

.. code-block:: python

    @register_output_sink(name="netcfd")
    def to_netcfd(quantity: DataArray, config: Mapping) -> Optional[Text]:
        pass
"""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator

OUTPUT_SINK_SIGNATURE = Callable[
    [Union[xr.DataArray, pd.DataFrame], int, KwArg(Any)], Optional[str]
]
"""Signature of functions used to save quantities."""

OUTPUT_SINKS: MutableMapping[str, Union[OUTPUT_SINK_SIGNATURE, Callable]] = {}
"""Stores a quantity somewhere."""


def factory(parameters: Mapping, sector_name: str = "default") -> Callable:
    from functools import partial
    from inspect import isclass
    from pathlib import Path

    from muse.outputs.sinks import OUTPUT_SINKS

    config = dict(**parameters)
    config.pop("quantity", None)

    def normalize(
        params: Optional[Mapping], filename: Optional[str] = None
    ) -> MutableMapping:
        if isinstance(params, Mapping):
            params = dict(**params)
        elif isinstance(params, str):
            params = dict(name=params)
        else:
            suffix = config.get("suffix", Path(filename).suffix if filename else "csv")
            if not suffix:
                suffix = "csv"
            params = dict(name=suffix)
        if "aggregate" in params:
            params["name"] = "aggregate"
            params["final_sink"] = dict(
                sink=normalize(params.pop("aggregate"), filename)
            )
        return params

    params = normalize(config.pop("sink", None), config.get("filename", None))
    sink_name = params.pop("name")

    if len(set(params).intersection(config)) != 0:
        raise ValueError("duplicate settings in output section")
    params.update(config)
    params["sector"] = sector_name.lower()
    if sink_name[0] == ".":
        sink_name = sink_name[1:]
    sink = OUTPUT_SINKS[sink_name]
    if isclass(sink):
        return sink(**params)  # type: ignore
    else:
        return partial(sink, **params)


@registrator(registry=OUTPUT_SINKS, loglevel=None)
def register_output_sink(function: OUTPUT_SINK_SIGNATURE = None) -> Callable:
    """Registers a function to save quantities."""
    assert function is not None
    return function


def sink_to_file(suffix: str):
    """Simplifies sinks to files.

    The decorator takes care of figuring out the path to the file, as well as trims the
    configuration dictionary to include only parameters for the sink itself. The
    decorated function returns the path to the output file.
    """
    from functools import wraps
    from logging import getLogger
    from pathlib import Path

    from muse.defaults import DEFAULT_OUTPUT_DIRECTORY

    def decorator(function: Callable[[Union[pd.DataFrame, xr.DataArray], str], None]):
        @wraps(function)
        def decorated(
            quantity: Union[pd.DataFrame, xr.DataArray], year: int, **config
        ) -> Path:
            params = config.copy()
            filestring = str(
                params.pop(
                    "filename", "{default_output_dir}/{Sector}{year}{Quantity}{suffix}"
                )
            )
            overwrite = params.pop("overwrite", False)
            sector = params.pop("sector", "")
            lsuffix = params.pop("suffix", suffix)
            if lsuffix is not None and len(lsuffix) > 0 and lsuffix[0] != ".":
                lsuffix = "." + lsuffix
            # assumes directory if filename has no suffix and does not exist
            if getattr(quantity, "name", None) is None:
                name = "quantity"
            else:
                name = str(quantity.name)

            filename = Path(
                filestring.format(
                    cwd=str(Path().absolute()),
                    quantity=name,
                    Quantity=name.title(),
                    sector=sector,
                    Sector=sector.title(),
                    year=year,
                    suffix=lsuffix,
                    default_output_dir=str(DEFAULT_OUTPUT_DIRECTORY),
                )
            )
            if filename.exists():
                if overwrite:
                    filename.unlink()
                else:
                    msg = (
                        f"File {filename} already exists and overwrite argument has "
                        "not been given."
                    )
                    getLogger(function.__module__).critical(msg)
                    raise OSError(msg)

            filename.parent.mkdir(parents=True, exist_ok=True)
            function(quantity, filename, **params)  # type: ignore
            return filename

        return decorated

    return decorator


def standardize_quantity(
    function: Callable[[Union[pd.DataFrame, xr.DataArray], str], None],
):
    """Helps standardize how the quantities are specified.

    This decorator adds three keyword arguments to an input function:

    - set_index: A dictionary or any argument accepted by
      :py:meth:`pandas.DataFrame.set_index`. Ignored if not specified. If specified, a
      call to :py:meth:`pandas.DataFrame.reset_index` is made first.
    - sort_index: A dictionary or any argument accepted by
      :py:meth:`pandas.DataFrame.sort_index`. Ignored if not specified.
    - keep_columns: a string or a list of strings with the names of the columns to keep.
      Ignored if not specified.

    The three functions are applied in the order given, assuming an input is specified.
    """
    from functools import wraps

    class NotSpecified:
        pass

    @wraps(function)
    def decorated(
        quantity: Union[pd.DataFrame, xr.DataArray],
        *args,
        set_index: Union[Any, NotSpecified] = NotSpecified,
        sort_index: Union[Any, NotSpecified, bool] = NotSpecified,
        keep_columns: Union[str, Sequence[str], NotSpecified] = NotSpecified,
        group_by: Union[Any, NotSpecified] = NotSpecified,
        **config,
    ) -> None:
        any_calls = (
            set_index is not NotSpecified
            or set_index is not NotSpecified
            or keep_columns is not None
        )
        if any_calls and hasattr(quantity, "to_dataframe"):
            data: pd.DataFrame = quantity.to_dataframe()
        else:
            data = cast(pd.DataFrame, quantity)
        if isinstance(set_index, Mapping):
            data = data.reset_index().set_index(**set_index)
        elif set_index is not NotSpecified:
            data = data.reset_index().set_index(set_index)
        if isinstance(sort_index, Mapping):
            data = data.sort_index(**sort_index)
        elif sort_index is True:
            data = data.sort_index()
        elif sort_index is not NotSpecified and sort_index is not False:
            data = data.sort_index(sort_index)
        if keep_columns is not NotSpecified:
            data = data[keep_columns]
        if isinstance(group_by, Mapping):
            data = data.groupby(group_by).sum()
        elif group_by is not NotSpecified:
            data = data.groupby(group_by).sum()

        return function(data, *args, **config)

    return decorated


@register_output_sink(name="csv")
@sink_to_file(".csv")
@standardize_quantity
def to_csv(
    quantity: Union[pd.DataFrame, xr.DataArray], filename: str, **params
) -> None:
    """Saves data array to csv format, using pandas.to_csv.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `pandas.to_csv`
    """
    params.update({"float_format": "%.11f"})
    if "index" not in params:
        params["index"] = False

    if isinstance(quantity, xr.DataArray):
        quantity = quantity.to_dataframe()

    par_list = [i for i in params.keys()]
    if len(par_list) > 0:
        if "columns" in par_list:
            quantity = quantity.reset_index()
    quantity.to_csv(filename, **params)


@register_output_sink(name=("netcdf", "nc"))
@sink_to_file(".nc")
def to_netcdf(
    quantity: Union[xr.DataArray, pd.DataFrame], filename: str, **params
) -> None:
    """Saves data array to csv format, using xarray.to_netcdf.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `xarray.to_netcdf`
    """
    name = quantity.name if getattr(quantity, "name", None) is not None else "quantity"
    assert name is not None
    if isinstance(quantity, pd.DataFrame):
        dataset = xr.Dataset.from_dataframe(quantity)
    else:
        dataset = xr.Dataset({name: quantity})
    dataset.to_netcdf(filename, **params)


@register_output_sink(name=("excel", "xlsx"))
@sink_to_file(".xlsx")
@standardize_quantity
def to_excel(
    quantity: Union[pd.DataFrame, xr.DataArray], filename: str, **params
) -> None:
    """Saves data array to csv format, using pandas.to_excel.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `pandas.to_excel`
    """
    from logging import getLogger

    if isinstance(quantity, xr.DataArray):
        quantity = quantity.to_dataframe()
    try:
        quantity.to_excel(filename, **params)
    except ModuleNotFoundError as e:
        msg = f"Cannot save to excel format: missing python package ({e})"
        getLogger(__name__).critical(msg)
        raise


@register_output_sink(name="aggregate")
class YearlyAggregate:
    """Incrementally aggregates data from year to year."""

    def __init__(
        self,
        final_sink: Optional[MutableMapping[str, Any]] = None,
        sector: str = "",
        axis="year",
        **kwargs,
    ):
        if final_sink is None:
            final_sink = dict(**kwargs)
        else:
            final_sink["sink"].update(**kwargs)
        if "overwrite" not in final_sink and not (
            isinstance(final_sink.get("sink", None), Mapping)
            and "overwrite" in final_sink["sink"]
        ):
            final_sink["overwrite"] = True
        self.sink = factory(final_sink, sector_name=sector)
        self.aggregate: Optional[pd.Dataframe] = None
        self.axis = axis

    def __call__(self, data: Union[pd.DataFrame, xr.DataArray], year: int):
        if isinstance(data, xr.DataArray):
            dataframe = data.to_dataframe()
        else:
            dataframe = data
        if self.axis in dataframe.columns:
            dataframe = dataframe[dataframe[self.axis] == year]
        elif self.axis in getattr(dataframe.index, "names", []):
            dataframe = dataframe.xs(year, level=self.axis).assign(year=year)
        if self.aggregate is None:
            self.aggregate = dataframe
        else:
            self.aggregate = pd.concat((self.aggregate, dataframe), sort=True)
        assert self.aggregate is not None
        if getattr(data, "name", None) is not None:
            self.aggregate.name = data.name
        return self.sink(self.aggregate, year=year)


class FiniteResourceException(Exception):
    """Raised when a finite resource is exceeded."""


@register_output_sink
def finite_resource_logger(
    data: Union[pd.DataFrame, xr.DataArray], year: int, early_exit=False, **kwargs
):
    from logging import getLogger

    if data is None:
        return

    over_limit = (~data).any([u for u in data.dims if u != "commodity"])
    over_limit = over_limit.sel(commodity=over_limit)
    if over_limit.size != 0:
        msg = "The following commodities have exceeded their limits: " + ", ".join(
            [str(u) for u in over_limit.commodity.values]
        )
        getLogger(__name__).critical(msg)
        if early_exit:
            raise FiniteResourceException(msg)
