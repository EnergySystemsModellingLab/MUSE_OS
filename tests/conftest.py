from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Text

from pandas import DataFrame
from pytest import fixture, mark
from xarray import DataArray, Dataset

from muse.agents import Agent


@fixture(autouse=True)
def logger():
    from logging import getLogger, DEBUG

    logger = getLogger("muse")
    logger.setLevel(DEBUG)
    return logger


@fixture(scope="session")
def cases_directory() -> Optional[Path]:
    try:
        import muse_legacy
    except ImportError:
        return None

    return Path(muse_legacy.__file__).parent / "data" / "test" / "cases"


@fixture(scope="session")
def regression_directories(cases_directory) -> Mapping[Text, Path]:
    if cases_directory is None:
        return {}
    return {
        directory.name: cases_directory / directory
        for directory in cases_directory.iterdir()
        if directory.is_dir()
        and (directory / "input").is_dir()
        and (directory / "output").is_dir()
    }


@fixture(scope="session")
def residential_dir(regression_directories):
    return regression_directories["Residential"]


@fixture()
def sectors_dir(tmpdir):
    """Copies sectors directory to new dir.

    This gives some assurance the machinery for specifying sectors data actually works.
    """
    from muse.defaults import DEFAULT_SECTORS_DIRECTORY
    from shutil import copytree

    copytree(DEFAULT_SECTORS_DIRECTORY, tmpdir.join("sectors_data_dir"))
    return tmpdir.join("sectors_data_dir")


def compare_df(
    expected: DataFrame,
    actual: DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan=False,
    msg=None,
):
    """Compares two dataframes approximately.

    Uses `numpy.allclose` for columns which are floating points.
    """
    from pytest import approx

    assert set(expected.columns) == set(actual.columns)
    assert set(expected.index) == set(actual.index)
    assert expected.shape == actual.shape

    floats = [u for (u, d) in zip(actual.columns, actual.dtypes) if d == "float"]
    nonfloats = [u for (u, d) in zip(actual.columns, actual.dtypes) if d != "float"]
    assert all(expected[nonfloats] == actual.loc[expected.index, nonfloats])
    for col in floats:
        actual_col = actual.loc[expected.index, col].values
        expected_col = expected[col].values
        assert actual_col == approx(expected_col, rel=rtol, abs=atol, nan_ok=equal_nan)


@fixture
def compare_dirs() -> Callable:
    def compare_dirs(actual_dir, expected_dir, **kwargs):
        """Compares all the csv files in a directory."""
        from pandas import read_csv
        from os.path import join, relpath, exists, isfile
        from os import walk

        compared_something = False
        for (dirpath, _, filenames) in walk(expected_dir):
            subdir = join(actual_dir, relpath(dirpath, expected_dir))
            for filename in filenames:
                compared_something = True
                expected = read_csv(join(dirpath, filename))
                assert exists(join(subdir, filename))
                assert isfile(join(subdir, filename))
                actual = read_csv(join(subdir, filename))
                compare_df(expected, actual, msg=filename, **kwargs)
        assert compared_something, "The test is not setup correctly"

    return compare_dirs


def pytest_collection_modifyitems(config, items):
    try:
        __import__("SGIModelData")
    except ImportError:
        skip_sgi_data = mark.skip(reason="Test reqires private data")
        for item in items:
            if "sgidata" in item.keywords:
                item.add_marker(skip_sgi_data)
    try:
        __import__("muse_legacy")
    except ImportError:
        skip_legacy = mark.skip(reason="Test reqires legacy code")
        for item in items:
            if "legacy" in item.keywords:
                item.add_marker(skip_legacy)


@fixture(scope="session")
def loaded_residential_settings(residential_input_file):
    """Initialized MCA with the default settings and the residential sector."""
    from muse.readers import read_settings
    from muse.readers.toml import undo_damage, convert

    settings = undo_damage(read_settings(residential_input_file))
    residential = settings["sectors"]["residential"]
    residential["subsectors"] = dict(
        new_and_retro=dict(
            agents=residential.pop("agents"),
            existing_capacity=residential.pop("existing_capacity"),
        )
    )

    return convert(settings)


@fixture(scope="session")
def mca(loaded_residential_settings):
    """Initialized MCA with the default settings and the residential sector."""
    from muse.mca import MCA

    result = MCA.factory(loaded_residential_settings)

    return result


@fixture(scope="session")
def buildings(mca):
    """Residential sector as read from regression test case."""
    return next((sector for sector in mca.sectors if sector.name == "residential"))


@fixture
def save_timeslice_globals():
    from muse import timeslices

    old = timeslices.TIMESLICE, timeslices.TRANSFORMS
    yield
    timeslices.TIMESLICE, timeslices.TRANSFORMS = old


@fixture
def default_timeslice_globals(save_timeslice_globals):
    from muse import timeslices

    timeslices.setup_module(timeslices.DEFAULT_TIMESLICE_DESCRIPTION)


@fixture
def timeslice(default_timeslice_globals) -> Dataset:
    from muse.readers.toml import read_timeslices

    return read_timeslices(dict(hour=["all-day"]))


@fixture
def other_timeslice() -> Dataset:
    from pandas import MultiIndex

    months = ["winter", "spring-autumn", "summer"]
    days = ["all-week", "all-week", "all-week"]
    hour = ["all-day", "all-day", "all-day"]
    coordinates = MultiIndex.from_arrays(
        [months, days, hour], names=("month", "day", "hour")
    )
    result = Dataset(coords={"timeslice": coordinates})
    result["represent_hours"] = ("timeslice", [2920, 2920, 2920])
    result = result.set_coords("represent_hours")
    return result


@fixture
def coords() -> Mapping:
    """Technoeconomics coordinates."""
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
def agent_args(coords) -> Mapping:
    """some standard arguments defining an agent."""
    from numpy.random import randint, choice, rand

    return {
        "region": choice(coords["region"]),
        "share": "agent_share",
        "enduses": {
            "space_heating": randint(0, 1000),
            "water_heating": randint(0, 1000),
        },
        "maturity_threshhold": rand(),
    }


@fixture
def technologies(coords) -> Dataset:
    """randomly generated technology characteristics."""
    from numpy import sum, nonzero
    from numpy.random import rand, randint, choice
    from muse.commodities import CommodityUsage

    result = Dataset(coords=coords)

    result["comm_type"] = ("commodity", coords["comm_type"])
    result["tech_type"] = "technology", ["solid", "liquid", "solid", "liquid"]
    result["fuel"] = "technology", ["person", "person", "oil", "person"]

    result = result.set_coords(("comm_type", "fuel", "tech_type"))

    def var(*dims, factor=100.0):
        shape = tuple(len(result[u]) for u in dims)
        return dims, (rand(*shape) * factor).astype(type(factor))

    result["agent_share"] = var("technology", "region", "year")
    result["agent_share"] /= sum(result.agent_share)
    result["agent_share_zero"] = result["agent_share"] * 0

    # first create a mask so each teach will have the same enduse, fuel, or
    # environmentals across years and regions
    fuels = result.comm_type == "energy"
    result["fixed_inputs"] = var("technology", "commodity")
    result.fixed_inputs[:] = randint(0, 3, result.fixed_inputs.shape) == 0
    result.fixed_inputs.loc[{"commodity": ~fuels}] = 0
    result["flexible_inputs"] = result.fixed_inputs * (
        randint(0, 2, result.fixed_inputs.shape) == 0
    )

    result["fixed_outputs"] = var("technology", "commodity")
    result.fixed_outputs[:] = randint(0, 3, result.fixed_outputs.shape) == 0
    enduses = result.comm_type == "service"
    environmentals = result.comm_type == "environmental"
    result.fixed_outputs.loc[{"commodity": ~(enduses | environmentals)}] = 0

    # make sure at least one fuel or enduse is set
    for tech in result.technology:
        fin = result.fixed_inputs
        if (fin.sel(technology=tech, commodity=fuels) < 1e-12).all():
            i = result.commodity[choice(nonzero(fuels.values)[0])]
            fin.loc[{"technology": tech, "commodity": i}] = 1

        fout = result.fixed_outputs
        if (fout.sel(technology=tech, commodity=enduses) < 1e-12).all():
            i = result.commodity[choice(nonzero(enduses.values)[0])]
            fout.loc[{"technology": tech, "commodity": i}] = 1

    # expand along year and region, and fill with random numbers
    ones = (result.year == result.year) * (result.region == result.region)
    result["fixed_inputs"] = result.fixed_inputs * ones
    result.fixed_inputs[:] *= rand(*result.fixed_inputs.shape)
    result["flexible_inputs"] = result.flexible_inputs * ones
    result.flexible_inputs[:] *= rand(*result.flexible_inputs.shape)
    result["fixed_outputs"] = result.fixed_outputs * ones
    result.fixed_outputs[:] *= rand(*result.fixed_outputs.shape)

    result["total_capacity_limit"] = var("technology", "region", "year")
    result.total_capacity_limit.loc[{"year": 2030}] += result.total_capacity_limit.sel(
        year=2030
    )
    result["max_capacity_addition"] = var("technology", "region", "year")
    result["max_capacity_growth"] = var("technology", "region", "year")

    result["utilization_factor"] = var("technology", "region", "year", factor=0.05)
    result.utilization_factor.values += 0.95
    result["fix_par"] = var("technology", "region", "year", factor=2.0)
    result["cap_par"] = var("technology", "region", "year", factor=30.0)
    result["var_par"] = var("technology", "region", "year", factor=1.0)
    result["fix_exp"] = var("technology", "region", "year", factor=1.0)
    result["cap_exp"] = var("technology", "region", "year", factor=1.0)
    result["var_exp"] = var("technology", "region", "year", factor=1.0)

    result["technical_life"] = var("technology", "region", "year", factor=10)
    result["technical_life"] = result.technical_life.astype(int).clip(min=1)

    result["interest_rate"] = var("technology", "region", "year", factor=0.1)

    result["comm_usage"] = "commodity", CommodityUsage.from_technologies(result)
    result = result.set_coords("comm_usage").drop_vars("comm_type")

    return result


@fixture
def agent_market(coords, technologies, timeslice) -> Dataset:
    from numpy.random import rand

    result = timeslice.copy()
    result["commodity"] = "commodity", coords["commodity"]
    result["region"] = "region", coords["region"]
    result["technology"] = "technology", coords["technology"]
    result["year"] = "year", coords["year"]

    def var(*dims, factor=100.0):
        shape = tuple(len(result[u]) for u in dims)
        return dims, (rand(*shape) * factor).astype(type(factor))

    result["capacity"] = var("technology", "region", "year")
    result["supply"] = var("commodity", "region", "timeslice", "year")
    result["consumption"] = var("commodity", "region", "timeslice", "year")
    result["prices"] = var("commodity", "region", "year", "timeslice")

    return result


@fixture
def market(coords, technologies, timeslice) -> Dataset:
    from numpy.random import rand

    result = timeslice.copy()
    result["commodity"] = "commodity", coords["commodity"]
    result["region"] = "region", coords["region"]
    result["year"] = "year", coords["year"]

    def var(*dims, factor=100.0):
        shape = tuple(len(result[u]) for u in dims)
        return dims, (rand(*shape) * factor).astype(type(factor))

    result["consumption"] = var("commodity", "region", "year", "timeslice")
    result["supply"] = var("commodity", "region", "year", "timeslice")
    result["prices"] = var("commodity", "region", "year", "timeslice")

    return result


def create_agent(agent_args, technologies, stock, agent_type="retrofit") -> Agent:
    from numpy.random import choice
    from muse import create_agent

    agent = create_agent(
        agent_type=agent_type,
        technologies=technologies,
        capacity=stock,
        year=2010,
        **agent_args
    )

    # because most of the input is random numbers, the agent's assets might
    # encompass every single technology.
    # This is not quite representative of the use case in the code, so in that
    # case, we add a bit of structure by removing some of the assets.
    technology = set([u for u in technologies.technology.values])
    if set(agent.assets.technology.values) == technology:
        techs = choice(technology, len(technology) // 2, replace=False)
        agent.assets = agent.assets.sel(asset=techs)
    return agent


@fixture
def newcapa_agent(agent_args, technologies, stock) -> Agent:
    return create_agent(agent_args, technologies, stock.capacity, "newcapa")


@fixture
def retro_agent(agent_args, technologies, stock) -> Agent:
    return create_agent(agent_args, technologies, stock.capacity, "retrofit")


@fixture
def objective(retro_agent, coords) -> DataArray:
    from numpy.random import choice, rand

    asset = retro_agent.assets.technology.rename(technology="asset")
    techs = [i for i in coords["technology"] if choice((True, False))]
    data = rand(len(asset), len(techs))
    coords = {"asset": asset, "technology": techs}
    return DataArray(data, coords=coords, dims=("asset", "technology"))


@fixture
def stock(coords, technologies) -> Dataset:
    return _stock(coords, technologies)


@fixture
def stock_factory() -> Callable:
    return _stock


def _stock(
    coords,
    technologies,
    region: Optional[Sequence[Text]] = None,
    nassets: Optional[int] = None,
) -> Dataset:
    from xarray import Dataset
    from numpy import cumprod, stack
    from numpy.random import choice, randint, rand

    ymin, ymax = min(coords["year"]), max(coords["year"]) + 1

    if nassets is None:
        nmin = max(1, 0 if region is None else len(region))
        n = randint(nmin, max(nmin, 10))
    else:
        n = nassets
    tech_subset = choice(
        coords["technology"], randint(2, len(coords["technology"])), replace=False
    )
    asset = {
        (choice(tech_subset), choice(range(ymin, min(ymin + 3, ymax))))
        for u in range(2 * n)
    }
    technology = [u[0] for u in asset][:n]
    installed = [u[1] for u in asset][:n]

    factors = cumprod(
        rand(len(installed), len(coords["year"])) / 4 + 0.75, axis=1
    ).clip(max=1)
    capacity = 0.75 * technologies.total_capacity_limit.sel(
        technology=technology, year=2010, region="USA", drop=True
    )
    capacity = stack(
        [capacity * factors[:, i] for i in range(factors.shape[1])], axis=1
    )

    result = Dataset()
    result["technology"] = "asset", technology
    result["installed"] = "asset", installed
    if region is not None and len(region) > 0:
        result["region"] = "asset", choice(region, len(installed))
    result["year"] = "year", [ymin, max(max(installed), ymax)]
    result["capacity"] = ("asset", "year"), capacity
    result = result.set_coords(("technology", "installed"))
    if "region" in result.data_vars:
        result = result.set_coords("region")
    return result


@fixture
def assets(coords, technologies) -> Dataset:
    """Stock with repeat technologies."""
    from xarray import concat

    return concat(
        (
            _stock(coords, technologies),
            _stock(coords, technologies),
            _stock(coords, technologies),
        ),
        dim="technology",
    )


@fixture
def search_space(retro_agent, technologies):
    """Example search space, as would be computed by an agent."""
    from numpy.random import randint

    coords = {
        "asset": list(set(retro_agent.assets.technology.values)),
        "replacement": technologies.technology.values,
    }
    return DataArray(
        randint(0, 4, tuple(len(u) for u in coords.values())) == 0,
        coords=coords,
        dims=coords.keys(),
        name="search_space",
    )


@fixture
def demand_share(coords, timeslice):
    """Example demand share, as would be computed by an agent."""
    from numpy.random import rand

    axes = {
        "commodity": coords["commodity"],
        "asset": list(set(coords["technology"])),
        "timeslice": timeslice.timeslice,
    }
    shape = len(axes["commodity"]), len(axes["asset"]), len(axes["timeslice"])
    result = DataArray(rand(*shape), coords=axes, dims=axes.keys(), name="demand_share")
    result.coords["represent_hours"] = timeslice.represent_hours
    return result


def create_fake_capacity(n: int, technologies: Dataset) -> DataArray:
    from numpy.random import rand, choice
    from xarray import Dataset

    n = 20
    baseyear = int(technologies.year.min())
    techs = choice(technologies.technology.values, 5)
    regions = choice(technologies.region.values, 5)
    data = Dataset()
    data["year"] = "year", technologies.year
    data["installed"] = "asset", choice(range(baseyear, baseyear + 5), n)
    data["technology"] = "asset", choice(techs, len(data.installed))
    data["region"] = "asset", choice(regions, len(data.installed))
    data = data.set_coords(("installed", "technology", "region"))
    data["capacity"] = ("year", "asset"), rand(len(data.year), len(data.asset))
    return data.capacity


@fixture
def capacity(technologies: Dataset) -> DataArray:
    return create_fake_capacity(50, technologies)


@fixture
def settings(tmpdir) -> dict:
    """Creates a dummy settings dictionary out of the default settings."""
    import toml
    from muse.readers import DEFAULT_SETTINGS_PATH
    from muse.readers.toml import format_paths

    def drop_optionals(settings):
        from copy import copy

        for k, v in copy(settings).items():
            if v == "OPTIONAL":
                settings.pop(k)
            elif isinstance(v, Mapping):
                drop_optionals(v)

    settings = toml.load(DEFAULT_SETTINGS_PATH)
    drop_optionals(settings)
    out = format_paths(settings, cwd=tmpdir, path=tmpdir, muse_sectors=tmpdir)

    required = {
        "time_framework": [2010, 2015, 2020],
        "foresight": 10,
        "regions": ["MEX"],
        "interest_rate": 0.1,
        "equilibrium": False,
        "maximum_iterations": 3,
        "tolerance": 0.1,
        "interpolation_mode": "Active",
    }
    out.update(required)

    carbon_budget_required = {
        "budget": [420000, 413000, 403000],
        "commodities": ["CO2f", "CO2r", "CH4", "N2O"],
    }

    out["carbon_budget_control"].update(carbon_budget_required)

    return out


@fixture(scope="session")
def residential_input_file() -> Path:
    """Gets the example residential sector settings file."""
    import muse_legacy

    input_file = (
        Path(muse_legacy.__file__).parent
        / "data"
        / "test"
        / "cases"
        / "Residential"
        / "settings_residential.toml"
    )
    return input_file


@fixture(autouse=True)
def warnings_as_errors(request):
    from warnings import simplefilter

    # disable fixture for some tests using legacy sectors.
    if (
        request.module.__name__ == "test_legacy_sector"
        and request.node.name.startswith("test_legacy_sector_regression[")
    ):
        return

    simplefilter("error", FutureWarning)
    simplefilter("error", DeprecationWarning)
    simplefilter("error", PendingDeprecationWarning)


@fixture
def rng():
    from numpy.random import default_rng

    return default_rng()
