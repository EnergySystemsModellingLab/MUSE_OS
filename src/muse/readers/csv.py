"""Ensemble of functions to read MUSE data."""
__all__ = [
    "read_technodictionary",
    "read_io_technodata",
    "read_initial_capacity",
    "read_technologies",
    "read_csv_timeslices",
    "read_global_commodities",
    "read_timeslice_shares",
    "read_csv_agent_parameters",
    "read_macro_drivers",
    "read_initial_market",
    "read_attribute_table",
    "read_regression_parameters",
    "read_csv_outputs",
]

from pathlib import Path
from typing import List, Optional, Sequence, Text, Union, cast

from xarray import DataArray, Dataset

from muse.defaults import DEFAULT_SECTORS_DIRECTORY


def find_sectors_file(
    filename: Union[Text, Path],
    sector: Optional[Text] = None,
    sectors_directory: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
) -> Path:
    """Looks through a few standard place for sector files."""
    filename = Path(filename)

    if sector is not None:
        dirs: Sequence[Path] = (
            Path(sectors_directory) / sector.title(),
            Path(sectors_directory),
        )
    else:
        dirs = (Path(sectors_directory),)
    for directory in dirs:
        path = directory / filename
        if path.is_file():
            return path
    if sector is not None:
        msg = "Could not find sector %s file %s." % (sector.title(), filename)
    else:
        msg = "Could not find file %s." % filename
    raise IOError(msg)


def read_technodictionary(filename: Union[Text, Path]) -> Dataset:
    """Reads and formats technodata into a dataset.

    There are three axes: technologies, regions, and year.
    """
    from re import sub
    from pandas import MultiIndex, read_csv, to_numeric
    from muse.readers import camel_to_snake

    def to_agent_share(name):
        return sub(r"agent(\d)", r"agent_share_\1", name)

    csv = read_csv(filename, float_precision="high", low_memory=False)
    csv = csv.rename(columns=camel_to_snake)
    csv = csv.rename(columns=to_agent_share)
    csv = csv.rename(columns={"end_use": "enduse", "availabiliy year": "availability"})
    data = csv[csv.process_name != "Unit"]
    ts = MultiIndex.from_arrays(
        [data.process_name, data.region_name, [int(u) for u in data.time]],
        names=("technology", "region", "year"),
    )
    data.index = ts
    data.columns.name = "technodata"
    data.index.name = "technology"
    data = data.drop(["process_name", "region_name", "time"], axis=1)

    data = data.apply(lambda x: to_numeric(x, errors="ignore"), axis=0)

    result = Dataset.from_dataframe(data.sort_index())
    if "fuel" in result.variables:
        result["fuel"] = result.fuel.isel(region=0, year=0)
    if "type" in result.variables:
        result["tech_type"] = result.type.isel(region=0, year=0)
    if "enduse" in result.variables:
        result["enduse"] = result.enduse.isel(region=0, year=0)

    units = csv[csv.process_name == "Unit"].drop(
        ["process_name", "region_name", "time", "level"], axis=1
    )
    for variable, value in units.items():
        if all(u not in {"-", "Retro", "New"} for u in value.values):
            result[variable].attrs["units"] = value.values[0]

    # Sanity checks
    if "year" in result.dims:
        assert len(set(result.year.data)) == result.year.data.size
        result = result.sortby("year")

    if "year" in result.dims and len(result.year) == 1:
        result = result.isel(year=0, drop=True)
    return result


def read_io_technodata(filename: Union[Text, Path]) -> Dataset:
    """Reads process inputs or ouputs.

    There are four axes: (technology, region, year, commodity)
    """
    from pandas import read_csv, MultiIndex, to_numeric
    from xarray import concat
    from numpy import array
    from muse.readers import camel_to_snake

    csv = read_csv(filename, float_precision="high", low_memory=False)
    data = csv[csv.ProcessName != "Unit"]

    region = array(data.RegionName, dtype=str)
    process = data.ProcessName
    year = [int(u) for u in data.Time]

    data = data.drop(["ProcessName", "RegionName", "Time"], axis=1)

    ts = MultiIndex.from_arrays(
        [process, region, year], names=("technology", "region", "year")
    )
    data.index = ts
    data.columns.name = "commodity"
    data.index.name = "technology"

    data = data.rename(columns=camel_to_snake)
    data = data.apply(lambda x: to_numeric(x, errors="ignore"), axis=0)

    fixed_set = Dataset.from_dataframe(data[data.level == "fixed"]).drop_vars("level")
    flexible_set = Dataset.from_dataframe(data[data.level == "flexible"]).drop_vars(
        "level"
    )
    commodity = DataArray(
        list(fixed_set.data_vars.keys()), dims="commodity", name="commodity"
    )
    fixed = concat(fixed_set.data_vars.values(), dim=commodity)
    flexible = concat(flexible_set.data_vars.values(), dim=commodity)

    result = Dataset(data_vars={"fixed": fixed, "flexible": flexible})
    result["flexible"] = result.flexible.fillna(0)

    # add units for flexible and fixed
    units = csv[csv.ProcessName == "Unit"].drop(
        ["ProcessName", "RegionName", "Time", "Level"], axis=1
    )
    units.index.name = "units"
    units.columns.name = "commodity"
    units = DataArray(units).isel(units=0, drop=True)
    result["commodity_units"] = units
    return result


def read_initial_capacity(filename: Union[Text, Path]) -> DataArray:
    """Reads and formats data about initial capacity into a dataframe."""
    from re import match
    from pandas import read_csv, MultiIndex
    from xarray import concat

    data = read_csv(filename, float_precision="high", low_memory=False).drop(
        "Unit", axis=1
    )

    data.index = MultiIndex.from_arrays(
        [data.ProcessName, data.RegionName], names=("asset", "region")
    )
    data.index.name = "asset"
    years = list([u for u in data.columns if match("^[0-9]{4}$", u) is not None])
    data = data[years]
    xrdata = Dataset.from_dataframe(data)

    ydim = DataArray(list((int(u) for u in years)), dims="year", name="year")
    result = concat([xrdata.get(u) for u in years], dim=ydim)
    result = result.rename("initial capacity")

    baseyear = int(result.year.min())
    result["asset"] = (
        "asset",
        MultiIndex.from_arrays(
            (result.asset.values, [baseyear] * len(result.asset)),
            names=("tech", "base"),
        ),
    )
    result = result.sel(asset=result.any(("region", "year")))
    result["technology"] = result.tech
    result["installed"] = result.base
    return result.drop_vars("asset")


def read_technologies(
    technodata_path_or_sector: Optional[Union[Text, Path]] = None,
    comm_out_path: Optional[Union[Text, Path]] = None,
    comm_in_path: Optional[Union[Text, Path]] = None,
    commodities: Optional[Union[Text, Path, Dataset]] = None,
    sectors_directory: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
) -> Dataset:
    """Reads data characterising technologies from files.

    Arguments:
        technodata_path_or_sector: If `comm_out_path` and `comm_in_path` are not given,
            then this argument refers to the name of the sector. The three paths are
            then determined using standard locations and name. Specifically, thechnodata
            looks for a "technodataSECTORNAME.csv" file in the standard location for
            that sector. However, if  `comm_out_path` and `comm_in_path` are given, then
            this should be the path to the the technodata file.
        comm_out_path: If given, then refers to the path of the file specifying output
            commmodities. If not given, then defaults to
            "commOUTtechnodataSECTORNAME.csv" in the relevant sector directory.
        comm_in_path: If given, then refers to the path of the file specifying input
            commmodities. If not given, then defaults to
            "commINtechnodataSECTORNAME.csv" in the relevant sector directory.
        commodities: Optional. If commodities is given, it should point to a global
            commodities file, or a dataset akin to reading such a file with
            `read_global_commodities`. In either case, the information pertaining to
            commodities will be added to the technologies dataset.
        sectors_directory: Optional. If `paths_or_sector` is a string indicating the
            name of the sector, then this is a path to a directory where standard input
            files are contained.

    Returns:
        A dataset with all the characteristics of the technologies.
    """
    from pathlib import Path
    from logging import getLogger
    from muse.commodities import CommodityUsage

    if (not comm_out_path) and (not comm_in_path):
        sector = technodata_path_or_sector
        assert sector is None or isinstance(sector, Text)
        tpath = find_sectors_file(
            f"technodata{sector.title()}.csv", sector, sectors_directory  # type: ignore
        )
        opath = find_sectors_file(
            f"commOUTtechnodata{sector.title()}.csv",  # type: ignore
            sector,
            sectors_directory,
        )
        ipath = find_sectors_file(
            f"commINtechnodata{sector.title()}.csv",  # type: ignore
            sector,
            sectors_directory,
        )
    else:
        assert isinstance(technodata_path_or_sector, (Text, Path))
        assert comm_out_path is not None
        assert comm_in_path is not None
        tpath = Path(technodata_path_or_sector)
        opath = Path(comm_out_path)
        ipath = Path(comm_in_path)

    msg = f"""Reading technology information from:
    - technodata: {tpath}
    - outputs: {opath}
    - inputs: {ipath}
    """
    if isinstance(commodities, (Text, Path)):
        msg += f"- global commodities file {commodities}"
    logger = getLogger(__name__)
    logger.info(msg)

    result = read_technodictionary(tpath)
    if any(result[u].isnull().any() for u in result.data_vars):
        raise ValueError(f"Inconsistent data in {tpath} (e.g. inconsistent years)")
    outs = read_io_technodata(opath).rename(
        flexible="flexible_outputs", fixed="fixed_outputs"
    )
    ins = read_io_technodata(ipath).rename(
        flexible="flexible_inputs", fixed="fixed_inputs"
    )
    if "year" in result.dims:
        outs = outs.interp(year=result.year)
        ins = ins.interp(year=result.year)

    result = result.merge(outs).merge(ins)

    # try and add info about commodities
    if isinstance(commodities, (Text, Path)):
        try:
            commodities = read_global_commodities(commodities)
        except IOError:
            logger.warning("Could not load global commodities file.")
            commodities = None

    if isinstance(commodities, Dataset):
        if result.commodity.isin(commodities.commodity).all():
            result = result.merge(commodities.sel(commodity=result.commodity))
        else:
            logger.warn("Commodities missing in global commodities file.")

    result["comm_usage"] = "commodity", CommodityUsage.from_technologies(result)
    result = result.set_coords("comm_usage")
    if "comm_type" in result.data_vars or "comm_type" in result.coords:
        result = result.drop_vars("comm_type")

    return result


def read_csv_timeslices(path: Union[Text, Path], **kwargs) -> DataArray:
    """Reads timeslice information from input."""
    from pandas import read_csv, MultiIndex
    from logging import getLogger

    getLogger(__name__).info("Reading timeslices from %s" % path)
    data = read_csv(path, float_precision="high", **kwargs)

    def snake_case(string):
        from re import sub

        result = sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r"-\1", string)
        return result.lower().strip()

    months = [snake_case(u) for u in data.Month.dropna()]
    days = [snake_case(u) for u in data.Day.dropna()]
    hours = [snake_case(u) for u in data.Hour.dropna()]
    ts_index = MultiIndex.from_arrays(
        (months, days, hours), names=("month", "day", "hour")
    )
    result = DataArray(
        data.RepresentHours.dropna().astype(int),
        coords={"timeslice": ts_index},
        dims="timeslice",
        name="represent_hours",
    )
    result.coords["represent_hours"] = result
    return result.timeslice


def read_global_commodities(path: Union[Text, Path]) -> Dataset:
    """Reads commodities information from input."""
    from pandas import read_csv
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if path.is_dir():
        path = path / "MuseGlobalCommodities.csv"
    if not path.is_file():
        raise IOError(f"File {path} does not exist.")

    getLogger(__name__).info(f"Reading global commodities from {path}.")

    data = read_csv(path, float_precision="high", low_memory=False)
    data.index = [camel_to_snake(u) for u in data.CommodityName]
    data.CommodityType = [camel_to_snake(u) for u in data.CommodityType]
    data = data.drop("CommodityName", axis=1)
    data = data.rename(
        columns={
            "CommodityType": "comm_type",
            "Commodity": "comm_name",
            "CommodityEmissionFactor_CO2": "emmission_factor",
            "HeatRate": "heat_rate",
            "Unit": "unit",
        }
    )
    data.index.name = "commodity"
    return Dataset(data)


def read_timeslice_shares(
    path: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
    sector: Optional[Text] = None,
    timeslice: Union[Text, Path, DataArray] = "Timeslices{sector}.csv",
) -> Dataset:
    """Reads sliceshare information into a Dataset.

    Additionaly, this function will try and recover the timeslice multi- index from a
    import file "Timeslices{sector}.csv" in the same directory as the timeslice shares.
    Pass `None` if this behaviour is not required.
    """
    from re import match
    from pandas import read_csv, MultiIndex
    from xarray import DataArray
    from logging import getLogger

    path = Path(path)
    if sector is None:
        if path.is_dir():
            sector = path.name
        else:
            path, filename = path.parent, path.name
            re = match(r"TimesliceShare(.*)\.csv", filename)
            sector = path.name if re is None else re.group(1)
    if isinstance(timeslice, Text) and "{sector}" in timeslice:
        timeslice = timeslice.format(sector=sector)
    if isinstance(timeslice, (Text, Path)) and not Path(timeslice).is_file():
        timeslice = find_sectors_file(timeslice, sector, path)
    if isinstance(timeslice, (Text, Path)):
        timeslice = read_csv_timeslices(timeslice, low_memory=False)

    share_path = find_sectors_file("TimesliceShare%s.csv" % sector, sector, path)
    getLogger(__name__).info("Reading timeslice shares from %s" % share_path)
    data = read_csv(share_path, float_precision="high", low_memory=False)
    data.index = MultiIndex.from_arrays(
        (data.RegionName, data.SN), names=("region", "timeslice")
    )
    data.index.name = "rt"
    data = data.drop(["RegionName", "SN"], axis=1)
    data.columns.name = "commodity"

    result = DataArray(data).unstack("rt").to_dataset(name="shares")

    if timeslice is None:
        result = result.drop_vars("timeslice")
    elif isinstance(timeslice, DataArray) and hasattr(timeslice, "timeslice"):
        result["timeslice"] = timeslice.timeslice
        result[timeslice.name] = timeslice
    else:
        result["timeslice"] = timeslice
    return result.shares


def read_csv_agent_parameters(filename) -> List:
    """Reads standard MUSE agent-declaration csv-files.

    Returns a list of dictionaries, where each dictionary can be used to instantiate an
    agent in :py:func:`muse.agents.factories.factory`.
    """
    from pandas import read_csv
    from re import sub

    if (
        isinstance(filename, Text)
        and Path(filename).suffix != ".csv"
        and not Path(filename).is_file()
    ):
        filename = find_sectors_file(f"BuildingAgent{filename}.csv", filename)

    data = read_csv(filename, float_precision="high", low_memory=False)
    if "AgentNumber" in data.columns:
        data = data.drop(["AgentNumber"], axis=1)
    result = []

    # We remove rows with missing information, and next over the rest
    for _, row in data.iterrows():
        objectives = [row.Objective1, row.Objective2, row.Objective3]
        floats = [row.ObjData1, row.ObjData2, row.ObjData3]
        sorting = [row.Objsort1, row.Objsort2, row.Objsort3]
        decision_params = [
            u for u in zip(objectives, sorting, floats) if isinstance(u[0], Text)
        ]
        agent_type = {
            "new": "newcapa",
            "newcapa": "newcapa",
            "retrofit": "retrofit",
            "retro": "retrofit",
            "agent": "agent",
        }[getattr(row, "Type", "agent").lower()]
        data = {
            "name": row.Name,
            "region": row.RegionName,
            "objectives": [u[0] for u in decision_params],
            "search_rules": row.SearchRule,
            "decision": {"name": row.DecisionMethod, "parameters": decision_params},
            "agent_type": agent_type,
        }
        if hasattr(row, "Quantity"):
            data["quantity"] = row.Quantity
        if hasattr(row, "MaturityThreshold"):
            data["maturity_threshhold"] = row.MaturityThreshold
        if agent_type == "retrofit":
            data["share"] = sub(r"Agent(\d)", r"agent_share_\1", row.AgentShare)
        if agent_type == "retrofit" and data["decision"] == "lexo":
            data["decision"] = "retro_lexo"
        result.append(data)
    return result


def read_macro_drivers(path: Union[Text, Path]) -> Dataset:
    """Reads a standard MUSE csv file for macro drivers."""
    from pandas import read_csv
    from logging import getLogger

    path = Path(path)

    getLogger(__name__).info(f"Reading macro drivers from {path}")

    table = read_csv(path, float_precision="high", low_memory=False)
    table.index = table.RegionName
    table.index.name = "region"
    table.columns.name = "year"
    table = table.drop(["Unit", "RegionName"], axis=1)

    population = table[table.Variable == "Population"]
    population = population.drop("Variable", axis=1)
    gdp = table[table.Variable == "GDP|PPP"].drop("Variable", axis=1)

    result = Dataset({"gdp": gdp, "population": population})
    result["year"] = "year", result.year.astype(int)
    result["region"] = "region", result.region.astype(str)
    return result


def read_initial_market(
    projections: Union[DataArray, Path, Text],
    base_year_import: Optional[Union[Text, Path, DataArray]] = None,
    base_year_export: Optional[Union[Text, Path, DataArray]] = None,
    timeslices: Optional[DataArray] = None,
) -> Dataset:
    """Read projections, import and export csv files."""
    from xarray import zeros_like
    from logging import getLogger
    from muse.timeslices import convert_timeslice, QuantityType

    # Projections must always be present
    if isinstance(projections, (Text, Path)):
        getLogger(__name__).info(f"Reading projections from {projections}")
        projections = read_attribute_table(projections)
    if timeslices is not None:
        projections = convert_timeslice(projections, timeslices, QuantityType.INTENSIVE)

    # Base year export is optional. If it is not there, it's set to zero
    if isinstance(base_year_export, (Text, Path)):
        getLogger(__name__).info(f"Reading base year export from {base_year_export}")
        base_year_export = read_attribute_table(base_year_export)
    elif base_year_export is None:
        getLogger(__name__).info("Base year export not provided. Set to zero.")
        base_year_export = zeros_like(projections)

    # Base year import is optional. If it is not there, it's set to zero
    if isinstance(base_year_import, (Text, Path)):
        getLogger(__name__).info(f"Reading base year import from {base_year_import}")
        base_year_import = read_attribute_table(base_year_import)
    elif base_year_import is None:
        getLogger(__name__).info("Base year import not provided. Set to zero.")
        base_year_import = zeros_like(projections)

    if timeslices is not None:
        base_year_export = convert_timeslice(
            base_year_export, timeslices, QuantityType.EXTENSIVE
        )
        base_year_import = convert_timeslice(
            base_year_import, timeslices, QuantityType.EXTENSIVE
        )
    base_year_export.name = "exports"
    base_year_import.name = "imports"

    static_trade = base_year_import - base_year_export
    static_trade.name = "static_trade"

    result = Dataset(
        {
            projections.name: projections,
            base_year_export.name: base_year_export,
            base_year_import.name: base_year_import,
            static_trade.name: static_trade,
        }
    )

    result = result.rename(
        commodity_price="prices", units_commodity_price="units_prices"
    )
    result["prices"] = result["prices"].expand_dims({"timeslice": timeslices})

    return result


def read_attribute_table(path: Union[Text, Path]) -> DataArray:
    """Read a standard MUSE csv file for price projections."""
    from pandas import read_csv, MultiIndex
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise IOError(f"{path} does not exist.")

    getLogger(__name__).info(f"Reading prices from {path}")

    table = read_csv(path, float_precision="high", low_memory=False)
    units = table.loc[0].drop(["RegionName", "Attribute", "Time"])
    table = table.drop(0)

    table.columns.name = "commodity"
    table = table.rename(
        columns={"RegionName": "region", "Attribute": "attribute", "Time": "year"}
    )

    region, year = table.region, table.year.astype(int)
    table = table.drop(["region", "year"], axis=1)
    table.index = MultiIndex.from_arrays([region, year], names=["region", "year"])

    attribute = camel_to_snake(table.attribute.unique()[0])
    table = table.drop(["attribute"], axis=1)
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    result = DataArray(table, name=attribute).astype(float)
    result = result.unstack("dim_0").fillna(0)

    result.coords["units_" + attribute] = ("commodity", units)

    return result


def read_regression_parameters(path: Union[Text, Path]) -> Dataset:
    """Reads the regression parameters from a standard MUSE csv file."""
    from pandas import read_csv, MultiIndex
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise IOError(f"{path} does not exist or is not a file.")
    getLogger(__name__).info(f"Reading regression parameters from {path}.")
    table = read_csv(path, float_precision="high", low_memory=False)

    # Normalize clumn names
    table.columns.name = "commodity"
    table = table.rename(
        columns={
            "RegionName": "region",
            "SectorName": "sector",
            "FunctionType": "function_type",
        }
    )

    # Create a multiindex based on three of the columns
    sector, region, function_type = (
        table.sector.apply(lambda x: x.lower()),
        table.region,
        table.function_type,
    )
    table = table.drop(["sector", "region", "function_type"], axis=1)
    table.index = MultiIndex.from_arrays([sector, region], names=["sector", "region"])
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    # Create a dataset, separating each type of coeeficient as a separate DataArray
    coeffs = Dataset(
        {
            k: DataArray(table[table.coeff == k].drop("coeff", axis=1))
            for k in table.coeff.unique()
        }
    )

    # Unstack the multi-index into separate dimensions
    coeffs = coeffs.unstack("dim_0").fillna(0)

    # We pair each sector with its function type
    function_type = list(zip(*set(zip(sector, function_type))))
    function_type = DataArray(
        list(function_type[1]),
        dims=["sector"],
        coords={"sector": list(function_type[0])},
    )
    coeffs["function_type"] = function_type

    return coeffs


def read_csv_outputs(
    paths: Union[Text, Path, Sequence[Union[Text, Path]]],
    columns: Text = "commodity",
    indices: Sequence[Text] = ("RegionName", "ProcessName", "Timeslice"),
    drop: Sequence[Text] = ("Unnamed: 0",),
) -> Dataset:
    """Read standard MUSE output files for consumption or supply."""
    from re import match
    from pandas import MultiIndex, read_csv
    from muse.readers import camel_to_snake

    def expand_paths(path):
        from glob import glob

        if isinstance(paths, Text):
            return [Path(p) for p in glob(path)]
        return Path(path)

    if isinstance(paths, Text):
        allfiles = expand_paths(paths)
    else:
        allfiles = [expand_paths(p) for p in cast(Sequence, paths)]
    if len(allfiles) == 0:
        raise IOError(f"No files found with paths {paths}")

    datas = {}
    for path in allfiles:
        data = read_csv(path, low_memory=False)
        data = data.drop(columns=[k for k in drop if k in data.columns])
        data.index = MultiIndex.from_arrays(
            [data[u] for u in indices if u in data.columns]
        )
        data.index.name = "asset"
        data.columns.name = columns
        data = data.drop(columns=list(indices))

        reyear = match(r"\S*.(\d{4})\S*\.csv", path.name)
        if reyear is None:
            raise IOError(f"Unexpected filename {path.name}")
        year = int(reyear.group(1))
        if year in datas:
            raise IOError(f"Year f{year} was found twice")
        data.year = year
        datas[year] = DataArray(data)

    result = (
        Dataset(datas)
        .to_array(dim="year")
        .sortby("year")
        .fillna(0)
        .unstack("asset")
        .rename({k: k.replace("Name", "").lower() for k in indices})
    )

    if "commodity" in result.coords:
        result.coords["commodity"] = [
            camel_to_snake(u) for u in result.commodity.values
        ]
    return result
