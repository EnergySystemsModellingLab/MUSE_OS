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

from typing import Any, Callable, Mapping, MutableMapping, Optional, Text, Union

import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator

OUTPUT_SINK_SIGNATURE = Callable[[xr.DataArray, int, KwArg()], Optional[Text]]
"""Signature of functions used to save quantities."""

OUTPUT_SINKS: Mapping[Text, Union[OUTPUT_SINK_SIGNATURE, Callable]] = {}
"""Stores a quantity somewhere."""


def factory(parameters: Mapping, sector_name: Text = "default") -> Callable:
    from pathlib import Path
    from inspect import isclass
    from functools import partial
    from muse.outputs.sinks import OUTPUT_SINKS

    config = dict(**parameters)
    config.pop("quantity", None)

    def normalize(
        params: Optional[Mapping], filename: Optional[Text] = None
    ) -> MutableMapping:
        if isinstance(params, Mapping):
            params = dict(**params)
        elif isinstance(params, Text):
            params = dict(name=params)
        else:
            params = dict(
                name=config.get("suffix", Path(filename).suffix if filename else "csv")
            )
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


def sink_to_file(suffix: Text):
    """Simplifies sinks to files.

    The decorator takes care of figuring out the path to the file, as well as trims the
    configuration dictionary to include only parameters for the sink itself. The
    decorated function returns the path to the output file.
    """
    from functools import wraps
    from logging import getLogger
    from pathlib import Path
    from muse.defaults import DEFAULT_OUTPUT_DIRECTORY

    def decorator(function: Callable[[xr.DataArray, Text], None]):
        @wraps(function)
        def decorated(quantity: xr.DataArray, year: int, **config) -> Path:
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
                    raise IOError(msg)

            filename.parent.mkdir(parents=True, exist_ok=True)
            function(quantity, filename, **params)  # type: ignore
            return filename

        return decorated

    return decorator


@register_output_sink(name="csv")
@sink_to_file(".csv")
def to_csv(quantity: xr.DataArray, filename: Text, **params) -> None:
    """Saves data array to csv format, using pandas.to_csv.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `pandas.to_csv`
    """
    params.update({"float_format": "%.11f"})
    quantity.to_dataframe().to_csv(filename, **params)


@register_output_sink(name=("netcdf", "nc"))
@sink_to_file(".nc")
def to_netcdf(quantity: xr.DataArray, filename: Text, **params) -> None:
    """Saves data array to csv format, using xarray.to_netcdf.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `xarray.to_netcdf`
    """
    name = quantity.name if quantity.name is not None else "quantity"
    xr.Dataset({name: quantity}).to_netcdf(filename, **params)


@register_output_sink(name=("excel", "xlsx"))
@sink_to_file(".xlsx")
def to_excel(quantity: xr.DataArray, filename: Text, **params) -> None:
    """Saves data array to csv format, using pandas.to_excel.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `pandas.to_excel`
    """
    from logging import getLogger

    try:
        quantity.to_dataframe().to_excel(filename, **params)
    except ModuleNotFoundError as e:
        msg = "Cannot save to excel format: missing python package (%s)" % e
        getLogger(__name__).critical(msg)
        raise


@register_output_sink(name="aggregate")
class YearlyAggregate:
    """Incrementally aggregates data from year to year."""

    def __init__(self, final_sink: Mapping[Text, Any], sector: Text = "", axis="year"):
        final_sink["sink"]["overwrite"] = True
        self.sink = factory(final_sink, sector_name=sector)
        self.aggregate: Optional[xr.DataArray] = None
        self.axis = axis

    def __call__(self, data: xr.DataArray, year: int):
        if self.aggregate is None:
            self.aggregate = data
        else:
            self.aggregate = xr.concat((self.aggregate, data), self.axis)
        return self.sink(self.aggregate, year=year)
