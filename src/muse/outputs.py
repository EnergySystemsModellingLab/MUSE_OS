"""Output quantities and sinks.

Functions that compute quantities for post-simulation analysis should all follow
the same signature:

.. code-block:: python

    @register_output_quantity
    def quantity(
        capacity: DataArray,
        market: Dataset,
        technologies: Dataset
    ) -> DataArray:
        pass

They take as input the current capacity profile, aggregated across a sectoar,
a dataset containing market-related quantities, and a dataset characterizing the
technologies in the market. It returns a single DataArray object.

The function should never modify it's arguments.

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
from typing import Callable, List, Mapping, Optional, Text, Union

from xarray import DataArray, Dataset

from muse.registration import registrator

OUTPUT_QUANTITY_SIGNATURE = Callable[[DataArray, Dataset, Dataset], DataArray]
"""Signature of functions computing quantities for later analysis."""

OUTPUT_QUANTITIES: Mapping[Text, OUTPUT_QUANTITY_SIGNATURE] = {}
"""Quantity for post-simulation analysis."""

OUTPUT_SINKS: Mapping[Text, Callable] = {}
"""Stores a quantity somewhere."""


OUTPUT_SINK_SIGNATURE = Callable[[DataArray, Mapping], Optional[Text]]
"""Signature of functions used to save quantities."""

OUTPUTS_PARAMETERS = Union[Text, Mapping]
"""Acceptable Datastructures for outputs parameters"""


@registrator(registry=OUTPUT_QUANTITIES)
def register_output_quantity(function: OUTPUT_QUANTITY_SIGNATURE = None) -> Callable:
    """Registers a function to compute an output quantity."""
    from functools import wraps

    assert function is not None

    @wraps(function)
    def decorated(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, DataArray):
            result.name = function.__name__
        return result

    return decorated


@registrator(registry=OUTPUT_SINKS, loglevel=None)
def register_output_sink(function: OUTPUT_SINK_SIGNATURE = None) -> Callable:
    """Registers a function to save quantities."""
    from functools import wraps
    from logging import getLogger

    logger = getLogger(function.__module__)

    assert function is not None

    @wraps(function)
    def decorated(quantity: DataArray, config: Mapping) -> Optional[Text]:
        assert function is not None
        result = function(quantity, config)
        if result is not None:
            msg = "Saving %s to %s" % (config["quantity"], result)
            logger.info(msg)
        return result

    return function


def factory(
    *parameters: OUTPUTS_PARAMETERS,
) -> Callable[[DataArray, Dataset, Dataset], None]:
    """Creates outputs functions for post-mortem analysis.

    Each parameter is a dictionary containing the following:

    - quantity (mandatory): name of the quantity to output. Mandatory.
    - sink (optional): name of the storage procedure, e.g. the file format
      or database format. When it cannot be guessed from `filename`, it defaults to
      "csv".
    - filename (optional): path to a directory or a file where to store the quantity. In
      the latter case, if sink is not given, it will be determined from the file
      extension. The filename can incorporate markers. By default, it is
      "{default_output_dir}/{sector}{year}{quantity}{suffix}".
    - any other parameter relevant to the sink, e.g. `pandas.to_csv` keyword
      arguments.

    For simplicity, it is also possible to given lone strings as input.
    They default to `{'quantity': string}` (and the sink will default to
    "csv").
    """
    if isinstance(parameters, Text):
        params: List = [{"quantity": parameters}]
    elif isinstance(parameters, Mapping):
        params = [parameters]
    else:
        params = [  # type: ignore
            {"quantity": o} if isinstance(o, Text) else o for o in parameters
        ]

    def save_multiple_outputs(
        capacity: DataArray,
        market: Dataset,
        technologies: Dataset,
        sector: Text = "default",
    ):

        for outputs in params:
            config = outputs.copy()
            config["sector"] = sector
            config["year"] = int(market.year.min())
            save_output(config, capacity, market, technologies)

    return save_multiple_outputs


@register_output_quantity
def capacity(capacity: DataArray, market: Dataset, technologies: Dataset) -> DataArray:
    """Current capacity."""
    return capacity


def market_quantity(
    quantity: DataArray,
    sum_over: Optional[Union[Text, List[Text]]] = None,
    drop: Optional[Union[Text, List[Text]]] = None,
) -> DataArray:
    from muse.utilities import multiindex_to_coords

    if sum_over:
        quantity = quantity.sum(sum_over)
    if "timeslice" in quantity.dims:
        quantity = multiindex_to_coords(quantity, "timeslice")
    if drop:
        quantity = quantity.drop_vars(drop)
    return quantity


@register_output_quantity
def consumption(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current consumption."""
    return market_quantity(market.consumption, sum_over=sum_over, drop=drop)


@register_output_quantity
def supply(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current supply."""
    return market_quantity(market.supply, sum_over=sum_over, drop=drop)


@register_output_quantity
def costs(
    capacity: DataArray,
    market: Dataset,
    technologies: Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
) -> DataArray:
    """Current supply."""
    from muse.commodities import is_pollutant

    return market_quantity(
        market.costs.sel(commodity=~is_pollutant(market.comm_usage)),
        sum_over=sum_over,
        drop=drop,
    )


def save_output(
    config: Mapping, capacity: DataArray, market: Dataset, technologies: Dataset
) -> Optional[Text]:
    """Computes output quantity and saves it to the sink.

    Arguments:
        config: A configuration dictionary which must contain at least `quantity`.
            It should also contain either `filename` or all three of `sink`, `sector`,
            and `year`. In the latter case, it can also optionally be given
            `dictionary`. If `sink` is not given, then it is guessed from the
            extension of the `filename`. If neither `sink` nor `filename` are given,
            then `sink` defaults to `csv`. Finally, the dictionary can be given
            any argument relevant to the sink.
        capacity: Asset capacity aggregated over the whole sector
        market: Market quantities
        technologies: Characteristics of the technologies

    Returns: Optionally, text describing where the data was saved, e.g. filename.
    """
    from pathlib import Path

    config = dict(**config)
    quantity_params = config.pop("quantity")
    if isinstance(quantity_params, Mapping):
        quantity_params = dict(**quantity_params)
        quantity = quantity_params.pop("name")
    else:
        quantity = quantity_params
        quantity_params = {}

    sink_params = config.pop("sink", None)
    if isinstance(sink_params, Mapping):
        sink_params = dict(**sink_params)
        sink = sink_params.pop("name")
    elif isinstance(sink_params, Text):
        sink = sink_params
        sink_params = {}
    else:
        filename = config.get("filename", None)
        sink = config.get("suffix", Path(filename).suffix if filename else "csv")
        sink_params = {}

    if len(set(sink_params).intersection(config)) != 0:
        raise ValueError("duplicate settings in output section")
    sink_params.update(config)
    if sink[0] == ".":
        sink = sink[1:]
    data = OUTPUT_QUANTITIES[quantity](  # type: ignore
        capacity, market, technologies, **quantity_params
    )
    return OUTPUT_SINKS[sink](data, **sink_params)


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

    def decorator(function: Callable[[DataArray, Text], None]):
        @wraps(function)
        def decorated(quantity: DataArray, **config) -> Path:
            params = config.copy()
            filestring = str(
                params.pop(
                    "filename", "{default_output_dir}/{Sector}{year}{Quantity}{suffix}"
                )
            )
            overwrite = params.pop("overwrite", False)
            year = params.pop("year", None)
            sector = params.pop("sector", "")
            lsuffix = params.pop("suffix", suffix)
            if lsuffix is not None and len(lsuffix) > 0 and lsuffix[0] != ".":
                lsuffix = "." + lsuffix
            # assumes directory if filename has no suffix and does not exist
            name = getattr(quantity, "name", function.__name__)
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
def to_csv(quantity: DataArray, filename: Text, **params) -> None:
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
def to_netcdf(quantity: DataArray, filename: Text, **params) -> None:
    """Saves data array to csv format, using xarray.to_netcdf.

    Arguments:
        quantity: The data to be saved
        filename: File to which the data should be saved
        params: A configuration dictionary accepting any argument to `xarray.to_netcdf`
    """
    name = quantity.name if quantity.name is not None else "quantity"
    Dataset({name: quantity}).to_netcdf(filename, **params)


@register_output_sink(name=("excel", "xlsx"))
@sink_to_file(".xlsx")
def to_excel(quantity: DataArray, filename: Text, **params) -> None:
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
