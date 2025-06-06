"""Common test fixtures for MUSE tests."""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from logging import CRITICAL, getLogger
from os import walk
from pathlib import Path
from typing import Callable, Optional, Union
from unittest.mock import patch
from warnings import filterwarnings, simplefilter

import numpy as np
import pandas as pd
from numpy.random import choice, default_rng, rand, randint
from pytest import fixture
from xarray import DataArray, Dataset

from muse.__main__ import patched_broadcast_compat_data
from muse.agents import Agent
from muse.commodities import CommodityUsage
from muse.timeslices import TIMESLICE, read_timeslices, setup_module

# Constants
RANDOM_SEED = 123
DEFAULT_FACTOR = 100.0
DEFAULT_TECH_TYPES = ["solid", "liquid", "solid", "liquid"]
DEFAULT_FUELS = ["person", "person", "oil", "person"]

DEFAULT_TIMESLICES = """
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
"""


@fixture(autouse=True)
def logger():
    """Configure logger for tests."""
    logger = getLogger("muse")
    logger.setLevel(CRITICAL)
    return logger


@fixture(autouse=True)
def patch_broadcast_compat_data():
    """Patch broadcast compatibility data."""
    with patch(
        "xarray.core.variable._broadcast_compat_data", patched_broadcast_compat_data
    ):
        yield


@fixture(autouse=True)
def random():
    """Set random seed for all tests to make them reproducible."""
    rng = default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    return rng


def compare_df(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    msg: Optional[str] = None,
) -> None:
    """Compare two dataframes approximately.

    Args:
        expected: Expected dataframe
        actual: Actual dataframe to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Whether to consider NaN values equal
        msg: Optional message to display on failure

    Raises:
        AssertionError: If dataframes don't match within tolerances
    """
    from pytest import approx

    assert set(expected.columns) == set(actual.columns), "Columns don't match"
    assert expected.shape == actual.shape, "Shapes don't match"
    assert set(expected.index) == set(actual.index), "Indices don't match"

    floats = [u for (u, d) in zip(actual.columns, actual.dtypes) if d == "float"]
    nonfloats = [u for (u, d) in zip(actual.columns, actual.dtypes) if d != "float"]

    assert all(expected[nonfloats] == actual.loc[expected.index, nonfloats]), (
        "Non-float columns don't match"
    )

    for col in floats:
        actual_col = actual.loc[expected.index, col].values
        expected_col = expected[col].values
        try:
            assert actual_col == approx(
                expected_col, rel=rtol, abs=atol, nan_ok=equal_nan
            )
        except AssertionError:
            # if the columns are not equal, we check if the sorted ones are equal as
            # sometimes the order of the rows is different because of different sorting
            # algorithms
            try:
                assert np.sort(actual_col) == approx(
                    np.sort(expected_col), rel=rtol, abs=atol, nan_ok=equal_nan
                )
            except AssertionError:
                print(f"file: {msg}, column: {col}")
                raise


@fixture
def compare_dirs() -> Callable:
    """Factory for directory comparison function."""

    def compare_dirs(
        actual_dir: Union[str, Path], expected_dir: Union[str, Path], **kwargs
    ) -> None:
        """Compare all CSV files in two directories.

        Args:
            actual_dir: Path to directory with actual files
            expected_dir: Path to directory with expected files
            **kwargs: Additional arguments passed to compare_df

        Raises:
            AssertionError: If directories don't match or test is not set up correctly
        """
        compared_something = False
        for dirpath, _, filenames in walk(expected_dir):
            subdir = Path(actual_dir) / Path(dirpath).relative_to(expected_dir)
            for filename in filenames:
                compared_something = True
                expected_filename = Path(dirpath) / filename
                expected = pd.read_csv(expected_filename)
                actual_filename = Path(subdir) / filename
                assert actual_filename.exists(), f"Missing file: {actual_filename}"
                assert actual_filename.is_file(), f"Not a file: {actual_filename}"
                actual = pd.read_csv(actual_filename)
                try:
                    compare_df(expected, actual, msg=filename, **kwargs)
                except Exception:
                    msg = (
                        f"Expected {expected_filename}\n"
                        + expected_filename.read_text()
                        + f"\n\nActual {actual_filename}:\n"
                        + actual_filename.read_text()
                        + "\n"
                    )
                    print(msg)
                    raise
        assert compared_something, "The test is not setup correctly"

    return compare_dirs


@fixture
def default_timeslice_globals():
    """Set up default timeslice configuration."""
    setup_module(DEFAULT_TIMESLICES)
    return DEFAULT_TIMESLICES


@fixture
def timeslice(default_timeslice_globals) -> DataArray:
    """Get the default timeslice dataset."""
    if TIMESLICE is None:
        # If TIMESLICE is not set, create it from the default timeslices
        return read_timeslices(default_timeslice_globals)
    return TIMESLICE


@fixture
def coords() -> Mapping:
    """Return standard coordinates for test cases.

    Returns:
        Mapping with technology, region, year, commodity and comm_type coordinates
    """
    return {
        "technology": ["burger_flipper", "soda_shaker", "deep_frier", "salad_arranger"],
        "region": ["ASEAN", "USA"],
        "year": [2010, 2030],
        "commodity": [
            "pizza",
            "coke",
            "pepsi",
            "tamales",
            "CH4",
            "CO2",
            "calories",
            "addviews",
        ],
        "comm_type": [
            "energy",
            "energy",
            "energy",
            "energy",
            "environmental",
            "environmental",
            "service",
            "service",
        ],
    }


@fixture
def agent_args(coords: Mapping) -> Mapping:
    """Generate standard arguments for creating an agent.

    Args:
        coords: Standard coordinate mapping

    Returns:
        Mapping with region, share, enduses and maturity_threshold
    """
    return {
        "region": choice(coords["region"]),
        "share": "agent_share",
        "enduses": {
            "space_heating": randint(0, 1000),
            "water_heating": randint(0, 1000),
        },
        "maturity_threshold": rand(),
    }


def var_generator(
    result: Dataset, dims: list[str], factor: float = DEFAULT_FACTOR
) -> tuple:
    """Generate random variables for a dataset.

    Args:
        result: Dataset to generate variables for
        dims: Dimensions to generate variables over
        factor: Scaling factor for random values

    Returns:
        Tuple of (dims, random_values)
    """
    shape = tuple(len(result[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


@fixture
def technologies(coords: Mapping) -> Dataset:
    """Generate random technology characteristics.

    Args:
        coords: Standard coordinate mapping

    Returns:
        Dataset with technology characteristics
    """
    result = Dataset(coords=coords)

    result["comm_type"] = ("commodity", coords["comm_type"])
    result["tech_type"] = "technology", ["solid", "liquid", "solid", "liquid"]

    result = result.set_coords(("comm_type", "tech_type"))

    # Generate random variables
    result["agent_share"] = var_generator(result, ["technology", "region", "year"])
    result["agent_share"] /= np.sum(result.agent_share)
    result["agent_share_zero"] = result["agent_share"] * 0

    # first create a mask so each tech will have consistent inputs/outputs across years
    # and regions
    fuels = result.comm_type == "energy"
    result["fixed_inputs"] = var_generator(result, ["technology", "commodity"])
    result.fixed_inputs[:] = randint(0, 3, result.fixed_inputs.shape) == 0
    result.fixed_inputs.loc[{"commodity": ~fuels}] = 0
    result["flexible_inputs"] = result.fixed_inputs * (
        randint(0, 2, result.fixed_inputs.shape) == 0
    )

    result["fixed_outputs"] = var_generator(result, ["technology", "commodity"])
    result.fixed_outputs[:] = randint(0, 3, result.fixed_outputs.shape) == 0
    enduses = result.comm_type == "service"
    environmentals = result.comm_type == "environmental"
    result.fixed_outputs.loc[{"commodity": ~(enduses | environmentals)}] = 0

    # make sure at least one energy input and service output is set
    for tech in result.technology:
        fin = result.fixed_inputs
        if (fin.sel(technology=tech, commodity=fuels) < 1e-12).all():
            i = result.commodity[choice(np.nonzero(fuels.values)[0])]
            fin.loc[{"technology": tech, "commodity": i}] = 1

        fout = result.fixed_outputs
        if (fout.sel(technology=tech, commodity=enduses) < 1e-12).all():
            i = result.commodity[choice(np.nonzero(enduses.values)[0])]
            fout.loc[{"technology": tech, "commodity": i}] = 1

    # Expand along year and region dimensions
    ones = (result.year == result.year) * (result.region == result.region)
    result["fixed_inputs"] = result.fixed_inputs * ones
    result.fixed_inputs[:] *= rand(*result.fixed_inputs.shape)
    result["flexible_inputs"] = result.flexible_inputs * ones
    result.flexible_inputs[:] *= rand(*result.flexible_inputs.shape)
    result["fixed_outputs"] = result.fixed_outputs * ones
    result.fixed_outputs[:] *= rand(*result.fixed_outputs.shape)

    # Generate capacity and utilization parameters
    result["total_capacity_limit"] = var_generator(
        result, ["technology", "region", "year"]
    )
    result.total_capacity_limit.loc[{"year": 2030}] += result.total_capacity_limit.sel(
        year=2030
    )
    result["max_capacity_addition"] = var_generator(
        result, ["technology", "region", "year"]
    )
    result["max_capacity_growth"] = var_generator(
        result, ["technology", "region", "year"]
    )

    result["utilization_factor"] = var_generator(
        result, ["technology", "region", "year"], factor=0.05
    )
    result.utilization_factor.values += 0.95

    # Generate cost parameters
    result["fix_par"] = var_generator(
        result, ["technology", "region", "year"], factor=2.0
    )
    result["cap_par"] = var_generator(
        result, ["technology", "region", "year"], factor=30.0
    )
    result["var_par"] = var_generator(
        result, ["technology", "region", "year"], factor=1.0
    )
    result["fix_exp"] = var_generator(
        result, ["technology", "region", "year"], factor=1.0
    )
    result["cap_exp"] = var_generator(
        result, ["technology", "region", "year"], factor=1.0
    )
    result["var_exp"] = var_generator(
        result, ["technology", "region", "year"], factor=1.0
    )

    # Generate technical parameters
    result["technical_life"] = var_generator(
        result, ["technology", "region", "year"], factor=10
    )
    result["technical_life"] = result.technical_life.astype(int).clip(min=1)
    result["interest_rate"] = var_generator(
        result, ["technology", "region", "year"], factor=0.1
    )

    # Set commodity usage
    result["comm_usage"] = "commodity", CommodityUsage.from_technologies(result).values
    result = result.set_coords("comm_usage").drop_vars("comm_type")

    return result


@fixture
def agent_market(coords: Mapping, timeslice: DataArray) -> Dataset:
    """Generate market data for agent testing.

    Args:
        coords: Standard coordinate mapping
        timeslice: Timeslice dataset

    Returns:
        Dataset with market data for agents
    """
    result = Dataset(coords=timeslice.coords)
    result["commodity"] = "commodity", coords["commodity"]
    result["region"] = "region", coords["region"]
    result["technology"] = "technology", coords["technology"]
    result["year"] = "year", coords["year"]

    # Generate market variables
    result["capacity"] = var_generator(result, ["technology", "region", "year"])
    result["supply"] = var_generator(
        result, ["commodity", "region", "timeslice", "year"]
    )
    result["consumption"] = var_generator(
        result, ["commodity", "region", "timeslice", "year"]
    )
    result["prices"] = var_generator(
        result, ["commodity", "region", "year", "timeslice"]
    )

    return result


@fixture
def market(coords: Mapping, timeslice: DataArray) -> Dataset:
    """Generate market data for testing.

    Args:
        coords: Standard coordinate mapping
        timeslice: Timeslice dataset

    Returns:
        Dataset with market data
    """
    result = Dataset(coords=timeslice.coords)
    result["commodity"] = "commodity", coords["commodity"]
    result["region"] = "region", coords["region"]
    result["year"] = "year", coords["year"]

    # Generate market variables
    result["consumption"] = var_generator(
        result, ["commodity", "region", "year", "timeslice"]
    )
    result["supply"] = var_generator(
        result, ["commodity", "region", "year", "timeslice"]
    )
    result["prices"] = var_generator(
        result, ["commodity", "region", "year", "timeslice"]
    )

    return result


def create_agent(
    agent_args: Mapping,
    technologies: Dataset,
    stock: Dataset,
    agent_type: str = "retrofit",
) -> Agent:
    """Create an agent for testing.

    Args:
        agent_args: Arguments for agent creation
        technologies: Technology characteristics
        stock: Stock data
        agent_type: Type of agent to create ("retrofit" or "newcapa")

    Returns:
        Created agent instance
    """
    from muse.agents.factories import create_agent as factory_create_agent

    region = agent_args["region"]
    agent = factory_create_agent(
        agent_type=agent_type,
        technologies=technologies.sel(region=region),
        capacity=stock.where(stock.region == region, drop=True).assign_coords(
            region=region
        ),
        year=2010,
        **agent_args,
    )

    # because most of the input is random numbers, the agent's assets might
    # encompass every single technology.
    # This is not quite representative of the use case in the code, so in that
    # case, we add a bit of structure by removing some of the assets.
    technology_names = set(technologies.technology.values)
    if set(agent.assets.technology.values) == technology_names:
        techs = choice(
            list(technology_names), len(technology_names) // 2, replace=False
        )
        agent.assets = agent.assets.where(agent.assets.technology.isin(techs))

    return agent


@fixture
def newcapa_agent(agent_args: Mapping, technologies: Dataset, stock: Dataset) -> Agent:
    """Create a new capacity agent for testing.

    Args:
        agent_args: Arguments for agent creation
        technologies: Technology characteristics
        stock: Stock data

    Returns:
        New capacity agent instance
    """
    return create_agent(agent_args, technologies, stock.capacity, "newcapa")


@fixture
def retro_agent(agent_args: Mapping, technologies: Dataset, stock: Dataset) -> Agent:
    """Create a retrofit agent for testing.

    Args:
        agent_args: Arguments for agent creation
        technologies: Technology characteristics
        stock: Stock data

    Returns:
        Retrofit agent instance
    """
    return create_agent(agent_args, technologies, stock.capacity, "retrofit")


@fixture
def stock(coords: Mapping, technologies: Dataset) -> Dataset:
    """Generate stock data for testing.

    Args:
        coords: Standard coordinate mapping
        technologies: Technology characteristics

    Returns:
        Dataset with stock data
    """
    return _stock(coords, technologies)


def _stock(coords: Mapping, technologies: Dataset) -> Dataset:
    """Internal function to generate stock data.

    Args:
        coords: Standard coordinate mapping
        technologies: Technology characteristics

    Returns:
        Dataset with stock data
    """
    from numpy import cumprod, stack

    from muse.utilities import broadcast_over_assets

    n_assets = 10

    # Create asset coordinates
    asset_coords = {
        "technology": ("asset", choice(coords["technology"], n_assets, replace=True)),
        "region": ("asset", choice(coords["region"], n_assets, replace=True)),
        "installed": ("asset", choice(coords["year"], n_assets)),
    }
    assets = Dataset(coords=asset_coords)

    # Generate random capacity data
    capacity_limits = broadcast_over_assets(technologies.total_capacity_limit, assets)
    factors = cumprod(rand(n_assets, len(coords["year"])) / 4 + 0.75, axis=1).clip(
        max=1
    )
    capacity = stack(
        [0.75 * capacity_limits * factors[:, i] for i in range(factors.shape[1])],
        axis=1,
    )

    # Create final dataset
    result = assets.copy()
    result["year"] = "year", coords["year"]
    result["capacity"] = ("asset", "year"), capacity
    return result


@fixture
def demand_share(coords: Mapping, timeslice: DataArray) -> DataArray:
    """Generate demand share data for testing.

    Args:
        coords: Standard coordinate mapping
        timeslice: Timeslice dataset

    Returns:
        DataArray with demand share data
    """
    n_assets = 5
    axes = {
        "commodity": coords["commodity"],
        "timeslice": timeslice.timeslice,
        "technology": (["asset"], choice(coords["technology"], n_assets, replace=True)),
        "region": (["asset"], choice(coords["region"], n_assets, replace=True)),
    }
    shape = (len(axes["commodity"]), len(axes["timeslice"]), n_assets)

    return DataArray(
        rand(*shape),
        dims=["commodity", "timeslice", "asset"],
        coords=axes,
        name="demand_share",
    )


def create_fake_capacity(n: int, technologies: Dataset) -> DataArray:
    """Create fake capacity data for testing.

    Args:
        n: Number of assets to create
        technologies: Technology characteristics

    Returns:
        DataArray with fake capacity data
    """
    years = technologies.year
    techs = choice(technologies.technology.values, 5)
    regions = choice(technologies.region.values, 5)

    data = Dataset()
    data["year"] = "year", technologies.year.values
    data["installed"] = "asset", choice(years, n)
    data["technology"] = "asset", choice(techs, n)
    data["region"] = "asset", choice(regions, n)
    data = data.set_coords(("installed", "technology", "region"))
    data["capacity"] = ("year", "asset"), rand(len(years), n)
    return data.capacity


@fixture
def capacity(technologies: Dataset) -> DataArray:
    """Generate capacity data for testing.

    Args:
        technologies: Technology characteristics

    Returns:
        DataArray with capacity data
    """
    return create_fake_capacity(20, technologies)


@fixture
def settings(tmpdir) -> dict:
    """Generate settings for testing.

    Args:
        tmpdir: Temporary directory path

    Returns:
        Dictionary with test settings
    """
    import toml

    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import format_paths

    def drop_optionals(settings: dict) -> None:
        """Remove optional settings from dictionary."""
        for k, v in list(settings.items()):
            if v == "OPTIONAL":
                settings.pop(k)
            elif isinstance(v, Mapping):
                drop_optionals(v)

    settings = toml.load(DEFAULT_SETTINGS_PATH)
    drop_optionals(settings)
    out = format_paths(settings, cwd=tmpdir, path=tmpdir, muse_sectors=tmpdir)

    # Add required settings
    required = {
        "time_framework": [2010, 2015, 2020],
        "regions": ["MEX"],
        "equilibrium": False,
        "maximum_iterations": 3,
        "tolerance": 0.1,
        "interpolation_mode": "linear",
    }
    out.update(required)

    # Add required carbon budget settings
    carbon_budget_required = {
        "budget": [420000, 413000, 403000],
        "commodities": ["CO2f", "CO2r", "CH4", "N2O"],
    }
    out["carbon_budget_control"].update(carbon_budget_required)

    return out


@fixture(autouse=True)
def warnings_as_errors(request):
    """Configure warnings to be treated as errors during testing.

    Args:
        request: Pytest request object
    """
    # Disable fixture for specific tests
    if (
        request.module.__name__ == "test_outputs"
        and request.node.name == "test_save_with_fullpath_to_excel_with_sink"
    ):
        return

    # Configure warning filters
    simplefilter("error", FutureWarning)
    simplefilter("error", DeprecationWarning)
    simplefilter("error", PendingDeprecationWarning)

    # The following warning is safe to ignore (raised by adhoc solver with Python 3.9)
    # TODO: may be able to remove this once support for Python 3.9 is dropped
    if request.module.__name__ == "test_fullsim_regression":
        filterwarnings(
            "ignore",
            message="__array__ implementation doesn't accept a copy keyword",
            category=DeprecationWarning,
            module="xarray.core.variable",
        )


@fixture
def save_registries():
    """Save and restore registry state during tests."""

    @contextmanager
    def saveme(module_name: str, registry_name: str):
        """Save and restore a specific registry.

        Args:
            module_name: Name of module containing registry
            registry_name: Name of registry to save/restore
        """
        module = import_module(module_name)
        old = getattr(module, registry_name)
        setattr(module, registry_name, deepcopy(old))
        yield
        setattr(module, registry_name, deepcopy(old))

    # List of registries to save/restore
    iterators = [
        saveme("muse.sectors", "SECTORS_REGISTERED"),
        saveme("muse.objectives", "OBJECTIVES"),
        saveme("muse.carbon_budget", "CARBON_BUDGET_FITTERS"),
        saveme("muse.carbon_budget", "CARBON_BUDGET_METHODS"),
        saveme("muse.constraints", "CONSTRAINTS"),
        saveme("muse.decisions", "DECISIONS"),
        saveme("muse.decorators", "SETTINGS_CHECKS"),
        saveme("muse.demand_share", "DEMAND_SHARE"),
        saveme("muse.filters", "FILTERS"),
        saveme("muse.hooks", "INITIAL_ASSET_TRANSFORM"),
        saveme("muse.hooks", "FINAL_ASSET_TRANSFORM"),
        saveme("muse.investments", "INVESTMENTS"),
        saveme("muse.production", "PRODUCTION_METHODS"),
        saveme("muse.outputs.mca", "OUTPUT_QUANTITIES"),
        saveme("muse.outputs.sectors", "OUTPUT_QUANTITIES"),
        saveme("muse.outputs.sinks", "OUTPUT_SINKS"),
        saveme("muse.interactions", "INTERACTION_NET"),
        saveme("muse.interactions", "AGENT_INTERACTIONS"),
        saveme("muse.regressions", "REGRESSION_FUNCTOR_CREATOR"),
        saveme("muse.regressions", "REGRESSION_FUNCTOR_NAMES"),
        saveme("muse.readers.toml", "SETTINGS_CHECKS"),
    ]

    map(next, iterators)
    yield
    map(next, iterators)


@fixture
def rng(request):
    """Create a random number generator for testing.

    Args:
        request: Pytest request object

    Returns:
        Random number generator instance
    """
    from numpy.random import default_rng

    return default_rng(getattr(request.config.option, "randomly_seed", None))
