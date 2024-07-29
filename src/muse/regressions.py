"""Functions and functors to compute macro-drivers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, ClassVar

from xarray import DataArray, Dataset

__all__ = [
    "factory",
    "Exponential",
    "ExponentialAdj",
    "Logistic",
    "Loglog",
    "LogisticSigmoid",
    "Linear",
    "endogenous_demand",
    "register_regression",
]


REGRESSION_FUNCTOR_CREATOR = {}
"""Dictionary of factory functions for creating regression functors."""

REGRESSION_FUNCTOR_NAMES = {}
"""Dictionary of alternative names for a given functor."""


class Regression(Callable):
    """Regression functors predicting demand from macro quantities.

    All regression functors are derived from this object.

    Examples:
    Creating a regression function can be done via it's constructor, or
    through a input csv file. This file is a

        >>> from muse.regressions import Exponential
        >>> from muse.defaults import DATA_DIRECTORY
        >>> path_to_regression_params = DATA_DIRECTORY / "regressionparamaters.csv"
        >>> if path_to_regression_params.exists():
        ...     expo = Exponential.factory(path_to_regression_params)

    The regression function itself takes either two `xarray.DataArray` or a
    `xarray.Dataset` as input. In any case, it is given the gpd and
    population. These can be read from standard MUSE csv files:

        >>> from muse.readers import read_macro_drivers
        >>> from muse.defaults import DATA_DIRECTORY
        >>> path_to_macrodrivers = DATA_DIRECTORY / "Macrodrivers.csv"
        >>> if path_to_macrodrivers.exists():
        ...     macrodrivers = read_macro_drivers(path_to_macrodrivers)
        ...     demand = expo(macrodrivers, year=2010, forecast=5)
    """

    __mappings__: ClassVar = {}
    """ Maps from input names to coefficient names

    Maps the coefficients names in the class to their names in the input data
    tables. This class attribute must be overridden.
    """
    __regression__ = ""
    """ Name of the regression function.

    This class attribute must be overridden.
    """

    def __init__(self, interpolation: str = "linear", base_year: int = 2010, **kwargs):
        super().__init__()
        self.interpolation = interpolation
        """Interpolation method when interpolating years"""
        self.base_year = base_year
        """Reference year for the start of the simulation."""

        attrs = {k: v for k, v in kwargs.items() if k in self.__mappings__}
        filters = {k: v for k, v in kwargs.items() if k not in self.__mappings__}
        self.coeffs = Dataset(attrs).sel(filters)
        """Coefficients of the regression function."""

    @abstractmethod
    def __call__(
        self,
        gdp_or_dataset: DataArray | Dataset,
        population: DataArray | None,
        year: int | Sequence[int] | None = None,
        forecast: int = 5,
        **kwargs,
    ) -> DataArray:
        pass

    def sel(self, **filters) -> Regression:
        """Regression over part of the data only."""
        return self.__class__(
            interpolation=self.interpolation,
            base_year=self.base_year,
            **(self.coeffs.sel(filters).data_vars),
        )

    @staticmethod
    def _to_dataset(
        first: DataArray | Dataset, population: DataArray | None
    ) -> Dataset:
        data = first if isinstance(first, Dataset) else Dataset({"gdp": first})
        if population is not None:
            data["population"] = population
        return data

    @staticmethod
    def _split_kwargs(data: Dataset, **kwargs) -> tuple[Mapping, Mapping]:
        filters = {k: v for k, v in kwargs.items() if k in data.dims}
        attrs = {k: v for k, v in kwargs.items() if k not in data.dims}
        return filters, attrs

    @classmethod
    def factory(
        cls,
        regression_data: str | Path | Dataset,
        interpolation: str = "linear",
        base_year: int = 2010,
        **filters,
    ) -> Regression:
        """Creates a regression function from standard muse input."""
        from muse.readers import read_regression_parameters

        assert cls.__mappings__
        assert cls.__regression__ != ""

        if isinstance(regression_data, (str, Path)):
            regression_data = read_regression_parameters(regression_data)

        # Get the parameters of interest with a 'simple' name
        coeffs = Dataset({k: regression_data[v] for k, v in cls.__mappings__.items()})
        filters.update(coeffs.data_vars)
        return cls(interpolation=interpolation, base_year=base_year, **filters)


def factory(
    regression_parameters: str | Path | Dataset,
    sector: str | Sequence[str] | None = None,
) -> Regression:
    """Creates regression functor from standard MUSE data for given sector."""
    from muse.readers import read_regression_parameters

    if isinstance(regression_parameters, (str, Path)):
        regression_parameters = read_regression_parameters(regression_parameters)

    if sector is not None:
        regression_parameters = regression_parameters.sel(sector=sector)

    if regression_parameters.function_type.size > 1:
        functions = [
            REGRESSION_FUNCTOR_CREATOR[value](group)
            for value, group in regression_parameters.groupby("function_type")
        ]

        def regressions(*args, **kwargs):
            from xarray import align

            result = functions[0](*args, **kwargs)
            for function in functions[1:]:
                left, right = align(result, function(*args, **kwargs), join="outer")
                result = left.fillna(0) + right.fillna(0)
            return result

        return regressions

    if regression_parameters.function_type.dims == ():
        functype = str(regression_parameters.function_type.values)
    else:
        functype = str(regression_parameters.function_type[0].values)
    regfactory = REGRESSION_FUNCTOR_CREATOR[functype]
    return regfactory(regression_parameters)


def _snake_case(name: str) -> str:
    from re import sub

    s1 = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _kebab_case(name: str) -> str:
    from re import sub

    s1 = sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


def register_regression(
    Functor: Regression = None, name: str | None = None
) -> Regression:
    """Registers a functor with MUSE regressions.

    Regression functors are registered with MUSE so that the functors can be
    called easily on created.

    functor name that the functor is registered with defaults to the snake_case
    version of the functor name.  However, it can also be specified explicitly
    as a *keyword* argument. In any case, it must be unique amongst all
    registered regression functor.
    """
    from logging import getLogger
    from pathlib import Path

    from muse.registration import name_variations

    # allows specifyng the registered name as a keyword argument
    if Functor is None:
        return lambda x: register_regression(x, name=name)

    logger = getLogger(__name__)

    def factory(file_or_dataset, *args, **kwargs):
        if isinstance(file_or_dataset, (Path, str)):
            opt = _kebab_case(name if name is not None else Functor.__name__)
            msg = f"Creating regression functor {opt} from data in {file_or_dataset}"
        else:
            opt = _kebab_case(name if name is not None else Functor.__name__)
            msg = f"Creating regression functor {opt} from dataset"
        logger.info(msg)
        function = getattr(Functor, "factory", Functor)
        return function(file_or_dataset, *args, **kwargs)

    names = {_snake_case(a) for a in {Functor.__name__, name} if a is not None}
    REGRESSION_FUNCTOR_NAMES[Functor.__name__.lower()] = []
    for n in name_variations(*names):
        if n in REGRESSION_FUNCTOR_CREATOR:
            msg = f"A regression with the name {n} already exists"
            raise RuntimeError(msg)
        REGRESSION_FUNCTOR_CREATOR[n] = factory
        REGRESSION_FUNCTOR_NAMES[Functor.__name__.lower()].append(n)

    return Functor


def regression_functor(
    mappings: Mapping[str, str], name: str | None = None
) -> Regression:
    """Creates a macro-driver functor from a function.

    The functions are transformed into classes inheriting from Regression.

    Arguments:
        mappings: a dictionary mapping from the functions expected coefficients
            (e.g. a, b, c) to the name in the input csv data tables (.e.g.
            constant, GDPexp, GDPscale).
        name: name by which the function is referred to in the input data table.
    """
    from logging import getLogger

    def decorator(func):
        from functools import wraps

        if func.__name__[0] != func.__name__[0].upper():
            raise Exception(
                "The function will be turned into a class. "
                "It's name should be capitalized."
            )

        name_ = (func.__name__ if name is None else name).lower()
        classname = func.__name__

        logger = getLogger(__name__)
        log = f"Calling {_kebab_case(classname)} regression function"

        # the main function will transform the input so 'func' can deal with it
        @wraps(func)
        def __call__(
            self,
            gdp_or_dataset: DataArray | Dataset,
            population: DataArray | None = None,
            year: int | Sequence[int] | None = None,
            forecast: int | Sequence[int] | None = None,
            **kwargs,
        ):
            from numpy import ndarray

            logger.debug(log)
            data = self._to_dataset(gdp_or_dataset, population)
            filters, attrs = self._split_kwargs(data, **kwargs)

            years = year
            if forecast is not None and year is None and "year" in data.dims:
                years = data.year

            if isinstance(forecast, (Sequence, ndarray)):
                forecast = DataArray(
                    forecast, coords={"forecast": forecast}, dims="forecast"
                )

            if forecast is not None and years is not None:
                years = years

            data = data.sel(filters)
            if years is not None:
                data = data.interp(
                    year=years,
                    method=self.interpolation,
                    kwargs={"fill_value": "extrapolate"},
                )

            attrs.update(**data.data_vars)
            # for the duration of the call, change coeffs to allow for
            # selections
            old_coeffs = self.coeffs
            try:
                filters = {k: v for k, v in kwargs.items() if k in self.coeffs.dims}
                self.coeffs = self.coeffs.sel(filters)
                return func(self, year=year, forecast=forecast, **attrs)
            finally:
                self.coeffs = old_coeffs

        msg = """
        This function accepts extra keyword arguments to filter over the
        dimensions of the input data-array.

        Furthermore, the gdp and population can be passed as a single argument

        if the first argument (not including self) is a dataset, then it is
        expected it should hold both the gdp and population. If population is
        also given, it will override population in the dataset argument.
        """
        if __call__.__doc__ is None:
            __call__.__doc__ = "\n\n" + msg
        else:
            __call__.__doc__ += f"Regression function: {name_}\n\n{msg}"

        doc = f"""Regression function: {name_}

        This functor is a regression function registered with MUSE as '{name_}'.
        """

        Self = type(
            classname,
            (Regression,),
            {
                "__regression__": name_,
                "__mappings__": mappings,
                "__call__": __call__,
                "__module__": func.__module__,
                "__doc__": doc,
            },
        )
        return Self

    return decorator


@register_regression
@regression_functor({"a": "constant", "b": "GDPexp"})
def Exponential(
    self, gdp: DataArray, population: DataArray, *args, year: int = 0, **kwargs
) -> DataArray:
    from numpy import exp

    factor = 1e6 * self.coeffs.a * population
    return factor * exp(self.coeffs.b * population / gdp)


@register_regression
@regression_functor({"a": "constant", "b": "GDPexp", "w": "timeEff"})
def ExponentialAdj(
    self,
    gdp: DataArray,
    population: DataArray,
    *args,
    year: int | Sequence[int] | None = None,
    forecast: int = 5,
    n: int = 6,
    **kwargs,
) -> DataArray:
    from numpy import exp, power

    if year is None:
        year = self.base_year

    factor = 1e6 * self.coeffs.a * population
    unadjusted = factor * exp(self.coeffs.b * population / gdp)
    p = power(year + forecast - self.base_year, n)
    return unadjusted * (1 + self.coeffs.w * p) / (1 + p)


@register_regression
@regression_functor({"a": "constant", "b": "GDPscale", "c": "GDPexp", "w": "timeEff"})
def Logistic(
    self, gdp: DataArray, population: DataArray, forecast: int = 5, n: int = 4, **kwargs
) -> DataArray:
    """(1 + t * f^n) / (1 + f^n) * a * pop / (1 + b * e^(gpd * c / pop)).

    With f the number of forecast years.
    """
    from numpy import exp, power

    a, b, c, w = self.coeffs.a, self.coeffs.b, self.coeffs.c, self.coeffs.w
    p = power(forecast, n)
    factor = 1e6 * a * population * (1 + w * p) / (1 + p)
    return factor / (1 + b * exp(gdp * c / population))


@register_regression(name="log-log")
@regression_functor({"a": "constant", "b": "GDPexp"})
def Loglog(self, gdp: DataArray, population: DataArray, *args, **kwargs) -> DataArray:
    """1e6 * e^a * population * (gpd/population)^b."""
    from numpy import exp, power

    factor = 1e6 * exp(self.coeffs.a) * population
    return factor * power(gdp / population, self.coeffs.b)


@register_regression
@regression_functor(
    {"a": "constant", "b0": "GDPscaleLess", "b1": "GDPscaleGreater", "c": "GDPexp"}
)
def LogisticSigmoid(
    self,
    gdp: DataArray,
    population: DataArray,
    *args,
    year: int | Sequence[int] | None = None,
    **kwargs,
) -> DataArray:
    """0.001 * (constant * pop + gdp * c / sqrt(1 + (gdp * scale / pop)^2)."""
    from numpy import power

    constant = self.coeffs.a
    c = self.coeffs.c
    if year is None:
        year = self.base_year
    if isinstance(year, int):
        scale = self.coeffs.b0 if year < 2015 else self.coeffs.b1

    elif year is not None and "year" in gdp.dims:
        # fmt: disable
        years = (
            year
            if isinstance(year, DataArray)
            else DataArray(year, coords={"year": year}, dims="year")
        )
        # fmt: enable
        scale = self.coeffs.b0.where(years < 2015, self.coeffs.b1)
    else:
        scale = 1

    p = power(1 + power(gdp * scale / population, 2), 0.5)
    return 0.001 * (constant * population + gdp * c / p)


@register_regression
class Linear(Regression):
    """a * population + b * (gdp - gdp[2010]/population[2010] * population)."""

    __mappings__: ClassVar[dict[str, str]] = {
        "a": "constant",
        "b0": "GDPscaleLess",
        "b1": "GDPscaleGreater",
    }

    __regression__ = "linear"
    __scaleyear__ = 2015

    def __call__(
        self,
        gdp_or_dataset: DataArray | Dataset,
        population: DataArray | None = None,
        year: int | Sequence[int] | None = None,
        forecast: int = 5,
        **kwargs,
    ) -> DataArray:
        from logging import getLogger

        getLogger(__name__).debug("Calling linear regression function")

        data = self._to_dataset(gdp_or_dataset, population)
        filters = self._split_kwargs(data, **kwargs)[0]

        coeffs = self.coeffs.sel(**filters)

        data = data.sel(**filters)

        if isinstance(year, int):
            condition = year + forecast < self.__scaleyear__
            scale = coeffs.b0 if condition else coeffs.b1
        elif year is not None and "year" in data.dims:
            years = (
                year
                if isinstance(year, DataArray)
                else DataArray(year, coords={"year": year}, dims="year")
            )
            sel = years + forecast < self.__scaleyear__
            scale = coeffs.b0.where(sel, coeffs.b1)
        else:
            scale = coeffs.b0
        data_baseyear = data.sel(year=self.base_year)
        gdpcap_offset = data_baseyear.gdp / data_baseyear.population

        if year is not None and "year" in data.dims:
            data = data.interp(year=year, method=self.interpolation)
        return coeffs.a * data.population + scale * (
            data.gdp - gdpcap_offset * data.population
        )


def endogenous_demand(
    regression_parameters: str | Path | Dataset,
    drivers: str | Path | Dataset,
    sector: str | Sequence | None = None,
    **kwargs,
) -> Dataset:
    """Endogenous demand based on macro drivers and regression parameters."""
    from muse.readers import read_macro_drivers

    regression = factory(regression_parameters, sector=sector)
    if isinstance(drivers, (str, Path)):
        drivers = read_macro_drivers(drivers)
    return regression(drivers, **kwargs)
