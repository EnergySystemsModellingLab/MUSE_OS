"""Ensemble of functions to read MUSE data."""

__all__ = [
    "read_technodictionary",
    "read_io_technodata",
    "read_initial_assets",
    "read_technologies",
    "read_csv_timeslices",
    "read_global_commodities",
    "read_timeslice_shares",
    "read_csv_agent_parameters",
    "read_macro_drivers",
    "read_initial_market",
    "read_attribute_table",
    "read_regression_parameters",
    "read_presets",
]

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import pandas as pd
import xarray as xr

from muse.defaults import DEFAULT_SECTORS_DIRECTORY
from muse.errors import UnitsConflictInCommodities


def to_numeric(x):
    """Converts a value to numeric if possible.

    Args:
        x: The value to convert.

    Returns:
        The value converted to numeric if possible, otherwise the original value.
    """
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x


def find_sectors_file(
    filename: Union[str, Path],
    sector: Optional[str] = None,
    sectors_directory: Union[str, Path] = DEFAULT_SECTORS_DIRECTORY,
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
        msg = f"Could not find sector {sector.title()} file {filename}."
    else:
        msg = f"Could not find file {filename}."
    raise OSError(msg)


def read_technodictionary(filename: Union[str, Path]) -> xr.Dataset:
    """Reads and formats technodata into a dataset.

    There are three axes: technologies, regions, and year.
    """
    from re import sub

    from muse.readers import camel_to_snake

    def to_agent_share(name):
        return sub(r"agent(\d)", r"agent_share_\1", name)

    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    csv.drop(csv.filter(regex="Unname"), axis=1, inplace=True)
    csv = (
        csv.rename(columns=camel_to_snake)
        .rename(columns=to_agent_share)
        .rename(columns={"end_use": "enduse", "availabiliy year": "availability"})
    )
    data = csv[csv.process_name != "Unit"]

    ts = pd.MultiIndex.from_arrays(
        [data.process_name, data.region_name, [int(u) for u in data.time]],
        names=("technology", "region", "year"),
    )
    data.index = ts
    data.columns.name = "technodata"
    data.index.name = "technology"
    data = data.drop(["process_name", "region_name", "time"], axis=1)
    data = data.apply(to_numeric, axis=0)

    check_utilization_and_minimum_service_factors(data, filename)

    result = xr.Dataset.from_dataframe(data.sort_index())
    if "fuel" in result.variables:
        result["fuel"] = result.fuel.isel(region=0, year=0)
        result["fuel"].values = [camel_to_snake(name) for name in result["fuel"].values]
    if "type" in result.variables:
        result["tech_type"] = result.type.isel(region=0, year=0)
        result["tech_type"].values = [
            camel_to_snake(name) for name in result["tech_type"].values
        ]
    if "enduse" in result.variables:
        result["enduse"] = result.enduse.isel(region=0, year=0)
        result["enduse"].values = [
            camel_to_snake(name) for name in result["enduse"].values
        ]

    units = csv[csv.process_name == "Unit"].drop(
        ["process_name", "region_name", "time"], axis=1
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


def read_technodata_timeslices(filename: Union[str, Path]) -> xr.Dataset:
    from muse.readers import camel_to_snake

    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    csv = csv.rename(columns=camel_to_snake)

    csv = csv.rename(
        columns={"process_name": "technology", "region_name": "region", "time": "year"}
    )
    data = csv[csv.technology != "Unit"]

    data = data.apply(to_numeric)
    check_utilization_and_minimum_service_factors(data, filename)

    ts = pd.MultiIndex.from_frame(
        data.drop(
            columns=["utilization_factor", "minimum_service_factor", "obj_sort"],
            errors="ignore",
        )
    )

    data.index = ts
    data.columns.name = "technodata_timeslice"
    data.index.name = "technology"

    data = data.filter(["utilization_factor", "minimum_service_factor"])

    result = xr.Dataset.from_dataframe(data.sort_index())

    timeslice_levels = [
        item
        for item in list(result.coords)
        if item not in ["technology", "region", "year"]
    ]
    result = result.stack(timeslice=timeslice_levels)
    return result


def read_io_technodata(filename: Union[str, Path]) -> xr.Dataset:
    """Reads process inputs or outputs.

    There are four axes: (technology, region, year, commodity)
    """
    from muse.readers import camel_to_snake

    csv = pd.read_csv(filename, float_precision="high", low_memory=False)

    # Unspecified Level values default to "fixed"
    if "Level" in csv.columns:
        csv["Level"] = csv["Level"].fillna("fixed")
    else:
        # Particularly relevant to outputs files where the Level column is omitted by
        # default, as only "fixed" outputs are allowed.
        csv["Level"] = "fixed"

    data = csv[csv.ProcessName != "Unit"]
    region = np.array(data.RegionName, dtype=str)
    process = data.ProcessName
    year = [int(u) for u in data.Time]

    data = data.drop(["ProcessName", "RegionName", "Time"], axis=1)

    ts = pd.MultiIndex.from_arrays(
        [process, region, year], names=("technology", "region", "year")
    )
    data.index = ts
    data.columns.name = "commodity"
    data.index.name = "technology"
    data = data.rename(columns=camel_to_snake)
    data = data.apply(to_numeric, axis=0)

    fixed_set = xr.Dataset.from_dataframe(data[data.level == "fixed"]).drop_vars(
        "level"
    )
    flexible_set = xr.Dataset.from_dataframe(data[data.level == "flexible"]).drop_vars(
        "level"
    )
    commodity = xr.DataArray(
        list(fixed_set.data_vars.keys()), dims="commodity", name="commodity"
    )
    fixed = xr.concat(fixed_set.data_vars.values(), dim=commodity)
    flexible = xr.concat(flexible_set.data_vars.values(), dim=commodity)

    result = xr.Dataset(data_vars={"fixed": fixed, "flexible": flexible})
    result["flexible"] = result.flexible.fillna(0)

    # add units for flexible and fixed
    units = csv[csv.ProcessName == "Unit"].drop(
        ["ProcessName", "RegionName", "Time", "Level"], axis=1
    )
    units.index.name = "units"
    units.columns.name = "commodity"
    units = xr.DataArray(units).isel(units=0, drop=True)
    result["commodity_units"] = units
    return result


def read_initial_assets(filename: Union[str, Path]) -> xr.DataArray:
    """Reads and formats data about initial capacity into a dataframe."""
    data = pd.read_csv(filename, float_precision="high", low_memory=False)
    if "Time" in data.columns:
        result = cast(
            xr.DataArray, read_trade(filename, skiprows=[1], columns_are_source=True)
        )
    else:
        result = read_initial_capacity(data)
    technology = result.technology
    result = result.drop_vars("technology").rename(technology="asset")
    result["technology"] = "asset", technology.values
    result["installed"] = ("asset", [int(result.year.min())] * len(result.technology))
    result["year"] = result.year.astype(int)
    return result


def read_initial_capacity(data: Union[str, Path, pd.DataFrame]) -> xr.DataArray:
    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data, float_precision="high", low_memory=False)
    if "Unit" in data.columns:
        data = data.drop(columns="Unit")
    data = (
        data.rename(columns=dict(ProcessName="technology", RegionName="region"))
        .melt(id_vars=["technology", "region"], var_name="year")
        .set_index(["region", "technology", "year"])
    )
    result = xr.DataArray.from_series(data["value"])
    # inconsistent legacy data files.
    result = result.sel(year=result.year != "2100.1")
    result["year"] = result.year.astype(int)
    return result


def read_technologies(
    technodata_path_or_sector: Optional[Union[str, Path]] = None,
    technodata_timeslices_path: Optional[Union[str, Path]] = None,
    comm_out_path: Optional[Union[str, Path]] = None,
    comm_in_path: Optional[Union[str, Path]] = None,
    commodities: Optional[Union[str, Path, xr.Dataset]] = None,
    sectors_directory: Union[str, Path] = DEFAULT_SECTORS_DIRECTORY,
) -> xr.Dataset:
    """Reads data characterising technologies from files.

    Arguments:
        technodata_path_or_sector: If `comm_out_path` and `comm_in_path` are not given,
            then this argument refers to the name of the sector. The three paths are
            then determined using standard locations and name. Specifically, technodata
            looks for a "technodataSECTORNAME.csv" file in the standard location for
            that sector. However, if  `comm_out_path` and `comm_in_path` are given, then
            this should be the path to the the technodata file.
        technodata_timeslices_path: This argument refers to the TechnodataTimeslices
            file which specifies the utilization factor per timeslice for the specified
            technology.
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
    from logging import getLogger

    from muse.commodities import CommodityUsage

    if (not comm_out_path) and (not comm_in_path):
        sector = technodata_path_or_sector
        assert sector is None or isinstance(sector, str)
        tpath = find_sectors_file(
            f"technodata{sector.title()}.csv",
            sector,
            sectors_directory,  # type: ignore
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
        assert isinstance(technodata_path_or_sector, (str, Path))
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
    if technodata_timeslices_path and isinstance(
        technodata_timeslices_path, (str, Path)
    ):
        ttpath = Path(technodata_timeslices_path)
        msg += f"""- technodata_timeslices: {ttpath}
        """
    else:
        ttpath = None

    if isinstance(commodities, (str, Path)):
        msg += f"""- global commodities file: {commodities}"""

    logger = getLogger(__name__)
    logger.info(msg)

    result = read_technodictionary(tpath)
    if any(result[u].isnull().any() for u in result.data_vars):
        raise ValueError(f"Inconsistent data in {tpath} (e.g. inconsistent years)")

    outs = read_io_technodata(opath).rename(
        flexible="flexible_outputs", fixed="fixed_outputs"
    )
    if not (outs["flexible_outputs"] == 0).all():
        raise ValueError(
            f"'flexible' outputs are not permitted in {opath}. "
            "All outputs must be 'fixed'"
        )
    outs = outs.drop_vars("flexible_outputs")
    ins = read_io_technodata(ipath).rename(
        flexible="flexible_inputs", fixed="fixed_inputs"
    )
    if "year" in result.dims and len(result.year) > 1:
        if all(len(outs[d]) > 1 for d in outs.dims if outs[d].dtype.kind in "uifc"):
            outs = outs.interp(year=result.year)
        if all(len(ins[d]) > 1 for d in ins.dims if ins[d].dtype.kind in "uifc"):
            ins = ins.interp(year=result.year)

    try:
        result = result.merge(outs).merge(ins)
    except xr.core.merge.MergeError:
        raise UnitsConflictInCommodities

    if isinstance(ttpath, (str, Path)):
        technodata_timeslice = read_technodata_timeslices(ttpath)
        result = result.drop_vars("utilization_factor")
        result = result.merge(technodata_timeslice)
    else:
        technodata_timeslice = None
    # try and add info about commodities
    if isinstance(commodities, (str, Path)):
        try:
            commodities = read_global_commodities(commodities)
        except OSError:
            logger.warning("Could not load global commodities file.")
            commodities = None

    if isinstance(commodities, xr.Dataset):
        if result.commodity.isin(commodities.commodity).all():
            result = result.merge(commodities.sel(commodity=result.commodity))

        else:
            raise OSError(
                "Commodities not found in global commodities file: check spelling."
            )

    result["comm_usage"] = (
        "commodity",
        CommodityUsage.from_technologies(result).values,
    )
    result = result.set_coords("comm_usage")
    if "comm_type" in result.data_vars or "comm_type" in result.coords:
        result = result.drop_vars("comm_type")

    return result


def read_csv_timeslices(path: Union[str, Path], **kwargs) -> xr.DataArray:
    """Reads timeslice information from input."""
    from logging import getLogger

    getLogger(__name__).info(f"Reading timeslices from {path}")
    data = pd.read_csv(path, float_precision="high", **kwargs)

    def snake_case(string):
        from re import sub

        result = sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r"-\1", string)
        return result.lower().strip()

    months = [snake_case(u) for u in data.Month.dropna()]
    days = [snake_case(u) for u in data.Day.dropna()]
    hours = [snake_case(u) for u in data.Hour.dropna()]
    ts_index = pd.MultiIndex.from_arrays(
        (months, days, hours), names=("month", "day", "hour")
    )
    result = xr.DataArray(
        data.RepresentHours.dropna().astype(int),
        coords={"timeslice": ts_index},
        dims="timeslice",
        name="represent_hours",
    )
    result.coords["represent_hours"] = result
    return result.timeslice


def read_global_commodities(path: Union[str, Path]) -> xr.Dataset:
    """Reads commodities information from input."""
    from logging import getLogger

    from muse.readers import camel_to_snake

    path = Path(path)
    if path.is_dir():
        path = path / "MuseGlobalCommodities.csv"
    if not path.is_file():
        raise OSError(f"File {path} does not exist.")

    getLogger(__name__).info(f"Reading global commodities from {path}.")

    data = pd.read_csv(path, float_precision="high", low_memory=False)
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
    return xr.Dataset(data)


def read_timeslice_shares(
    path: Union[str, Path] = DEFAULT_SECTORS_DIRECTORY,
    sector: Optional[str] = None,
    timeslice: Union[str, Path, xr.DataArray] = "Timeslices{sector}.csv",
) -> xr.Dataset:
    """Reads sliceshare information into a xr.Dataset.

    Additionally, this function will try and recover the timeslice multi- index from a
    import file "Timeslices{sector}.csv" in the same directory as the timeslice shares.
    Pass `None` if this behaviour is not required.
    """
    from logging import getLogger
    from re import match

    path = Path(path)
    if sector is None:
        if path.is_dir():
            sector = path.name
        else:
            path, filename = path.parent, path.name
            re = match(r"TimesliceShare(.*)\.csv", filename)
            sector = path.name if re is None else re.group(1)
    if isinstance(timeslice, str) and "{sector}" in timeslice:
        timeslice = timeslice.format(sector=sector)
    if isinstance(timeslice, (str, Path)) and not Path(timeslice).is_file():
        timeslice = find_sectors_file(timeslice, sector, path)
    if isinstance(timeslice, (str, Path)):
        timeslice = read_csv_timeslices(timeslice, low_memory=False)

    share_path = find_sectors_file(f"TimesliceShare{sector}.csv", sector, path)
    getLogger(__name__).info(f"Reading timeslice shares from {share_path}")
    data = pd.read_csv(share_path, float_precision="high", low_memory=False)
    data.index = pd.MultiIndex.from_arrays(
        (data.RegionName, data.SN), names=("region", "timeslice")
    )
    data.index.name = "rt"
    data = data.drop(["RegionName", "SN"], axis=1)
    data.columns.name = "commodity"

    result = xr.DataArray(data).unstack("rt").to_dataset(name="shares")

    if timeslice is None:
        result = result.drop_vars("timeslice")
    elif isinstance(timeslice, xr.DataArray) and hasattr(timeslice, "timeslice"):
        result["timeslice"] = timeslice.timeslice
    else:
        result["timeslice"] = timeslice
    return result.shares


def read_csv_agent_parameters(filename) -> list:
    """Reads standard MUSE agent-declaration csv-files.

    Returns a list of dictionaries, where each dictionary can be used to instantiate an
    agent in :py:func:`muse.agents.factories.factory`.
    """
    from re import sub

    if (
        isinstance(filename, str)
        and Path(filename).suffix != ".csv"
        and not Path(filename).is_file()
    ):
        filename = find_sectors_file(f"BuildingAgent{filename}.csv", filename)

    data = pd.read_csv(filename, float_precision="high", low_memory=False)
    if "AgentNumber" in data.columns:
        data = data.drop(["AgentNumber"], axis=1)
    result = []

    # We remove rows with missing information, and next over the rest
    for _, row in data.iterrows():
        objectives = row[[i.startswith("Objective") for i in row.index]]
        floats = row[[i.startswith("ObjData") for i in row.index]]
        sorting = row[[i.startswith("Objsort") for i in row.index]]

        if len(objectives) != len(floats) or len(objectives) != len(sorting):
            raise ValueError(
                f"Agent Objective, ObjData, and Objsort columns are inconsistent in {filename}"  # noqa: E501
            )
        objectives = objectives.dropna().to_list()
        for u in objectives:
            if not issubclass(type(u), str):
                raise ValueError(
                    f"Agent Objective requires a string entry in {filename}"
                )
        sort = sorting.dropna().to_list()
        for u in sort:
            if not issubclass(type(u), bool):
                raise ValueError(
                    f"Agent Objsort requires a boolean entry in {filename}"
                )
        floats = floats.dropna().to_list()
        for u in floats:
            if not issubclass(type(u), (int, float)):
                raise ValueError(f"Agent ObjData requires a float entry in {filename}")
        decision_params = [
            u for u in zip(objectives, sorting, floats) if isinstance(u[0], str)
        ]

        agent_type = {
            "new": "newcapa",
            "newcapa": "newcapa",
            "retrofit": "retrofit",
            "retro": "retrofit",
            "agent": "agent",
            "default": "agent",
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
            data["maturity_threshold"] = row.MaturityThreshold
        if hasattr(row, "SpendLimit"):
            data["spend_limit"] = row.SpendLimit
        # if agent_type != "newcapa":
        data["share"] = sub(r"Agent(\d)", r"agent_share_\1", row.AgentShare)
        if agent_type == "retrofit" and data["decision"] == "lexo":
            data["decision"] = "retro_lexo"
        result.append(data)
    return result


def read_macro_drivers(path: Union[str, Path]) -> xr.Dataset:
    """Reads a standard MUSE csv file for macro drivers."""
    from logging import getLogger

    path = Path(path)

    getLogger(__name__).info(f"Reading macro drivers from {path}")

    table = pd.read_csv(path, float_precision="high", low_memory=False)
    table.index = table.RegionName
    table.index.name = "region"
    table.columns.name = "year"
    table = table.drop(["Unit", "RegionName"], axis=1)

    population = table[table.Variable == "Population"]
    population = population.drop("Variable", axis=1)
    gdp = table[table.Variable == "GDP|PPP"].drop("Variable", axis=1)

    result = xr.Dataset({"gdp": gdp, "population": population})
    result["year"] = "year", result.year.values.astype(int)
    result["region"] = "region", result.region.values.astype(str)
    return result


def read_initial_market(
    projections: Union[xr.DataArray, Path, str],
    base_year_import: Optional[Union[str, Path, xr.DataArray]] = None,
    base_year_export: Optional[Union[str, Path, xr.DataArray]] = None,
    timeslices: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Read projections, import and export csv files."""
    from logging import getLogger

    from muse.timeslices import QuantityType, convert_timeslice

    # Projections must always be present
    if isinstance(projections, (str, Path)):
        getLogger(__name__).info(f"Reading projections from {projections}")
        projections = read_attribute_table(projections)
    if timeslices is not None:
        projections = convert_timeslice(projections, timeslices, QuantityType.INTENSIVE)

    # Base year export is optional. If it is not there, it's set to zero
    if isinstance(base_year_export, (str, Path)):
        getLogger(__name__).info(f"Reading base year export from {base_year_export}")
        base_year_export = read_attribute_table(base_year_export)
    elif base_year_export is None:
        getLogger(__name__).info("Base year export not provided. Set to zero.")
        base_year_export = xr.zeros_like(projections)

    # Base year import is optional. If it is not there, it's set to zero
    if isinstance(base_year_import, (str, Path)):
        getLogger(__name__).info(f"Reading base year import from {base_year_import}")
        base_year_import = read_attribute_table(base_year_import)
    elif base_year_import is None:
        getLogger(__name__).info("Base year import not provided. Set to zero.")
        base_year_import = xr.zeros_like(projections)

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

    result = xr.Dataset(
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
    result["prices"] = (
        result["prices"].expand_dims({"timeslice": timeslices}).drop_vars("timeslice")
    )

    return result


def read_attribute_table(path: Union[str, Path]) -> xr.DataArray:
    """Read a standard MUSE csv file for price projections."""
    from logging import getLogger

    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise OSError(f"{path} does not exist.")

    getLogger(__name__).info(f"Reading prices from {path}")

    table = pd.read_csv(path, float_precision="high", low_memory=False)
    units = table.loc[0].drop(["RegionName", "Attribute", "Time"])
    table = table.drop(0)

    table.columns.name = "commodity"
    table = table.rename(
        columns={"RegionName": "region", "Attribute": "attribute", "Time": "year"}
    )

    region, year = table.region, table.year.astype(int)
    table = table.drop(["region", "year"], axis=1)
    table.index = pd.MultiIndex.from_arrays([region, year], names=["region", "year"])

    attribute = camel_to_snake(table.attribute.unique()[0])
    table = table.drop(["attribute"], axis=1)
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    result = xr.DataArray(table, name=attribute).astype(float)
    result = result.unstack("dim_0").fillna(0)

    result.coords["units_" + attribute] = ("commodity", units)

    return result


def read_regression_parameters(path: Union[str, Path]) -> xr.Dataset:
    """Reads the regression parameters from a standard MUSE csv file."""
    from logging import getLogger

    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise OSError(f"{path} does not exist or is not a file.")
    getLogger(__name__).info(f"Reading regression parameters from {path}.")
    table = pd.read_csv(path, float_precision="high", low_memory=False)

    # Normalize column names
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
    table.index = pd.MultiIndex.from_arrays(
        [sector, region], names=["sector", "region"]
    )
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    # Create a dataset, separating each type of coeeficient as a separate xr.DataArray
    coeffs = xr.Dataset(
        {
            k: xr.DataArray(table[table.coeff == k].drop("coeff", axis=1))
            for k in table.coeff.unique()
        }
    )

    # Unstack the multi-index into separate dimensions
    coeffs = coeffs.unstack("dim_0").fillna(0)

    # We pair each sector with its function type
    function_type = list(zip(*set(zip(sector, function_type))))
    function_type = xr.DataArray(
        list(function_type[1]),
        dims=["sector"],
        coords={"sector": list(function_type[0])},
    )
    coeffs["function_type"] = function_type

    return coeffs


def read_presets(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    columns: str = "commodity",
    indices: Sequence[str] = ("RegionName", "Timeslice"),
    drop: Sequence[str] = ("Unnamed: 0",),
) -> xr.Dataset:
    """Read consumption or supply files for preset sectors."""
    from logging import getLogger
    from re import match

    from muse.readers import camel_to_snake

    def expand_paths(path):
        from glob import glob

        if isinstance(paths, str):
            return [Path(p) for p in glob(path)]
        return Path(path)

    if isinstance(paths, str):
        allfiles = expand_paths(paths)
    else:
        allfiles = [expand_paths(p) for p in cast(Sequence, paths)]
    if len(allfiles) == 0:
        raise OSError(f"No files found with paths {paths}")

    datas = {}
    for path in allfiles:
        data = pd.read_csv(path, low_memory=False)
        assert all(u in data.columns for u in indices)

        # Legacy: drop ProcessName column and sum data (PR #448)
        if "ProcessName" in data.columns:
            data = (
                data.drop(columns=["ProcessName"])
                .groupby(list(indices))
                .sum()
                .reset_index()
            )
            msg = (
                f"The ProcessName column (in file {path}) is deprecated. "
                "Data has been summed across processes, and this column has been "
                "dropped."
            )
            getLogger(__name__).warning(msg)

        data = data.drop(columns=[k for k in drop if k in data.columns])
        data.index = pd.MultiIndex.from_arrays([data[u] for u in indices])
        data.index.name = "asset"
        data.columns.name = columns
        data = data.drop(columns=list(indices))

        reyear = match(r"\S*.(\d{4})\S*\.csv", path.name)
        if reyear is None:
            raise OSError(f"Unexpected filename {path.name}")
        year = int(reyear.group(1))
        if year in datas:
            raise OSError(f"Year f{year} was found twice")
        data.year = year
        datas[year] = xr.DataArray(data)

    result = (
        xr.Dataset(datas)
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


def read_trade(
    data: Union[pd.DataFrame, str, Path],
    columns_are_source: bool = True,
    parameters: Optional[str] = None,
    skiprows: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    drop: Optional[Union[str, Sequence[str]]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """Read CSV table with source and destination regions."""
    from muse.readers import camel_to_snake

    if not isinstance(data, pd.DataFrame):
        data = pd.read_csv(data, skiprows=skiprows)

    if parameters is None and "Parameter" in data.columns:
        parameters = "Parameter"
    if columns_are_source:
        col_region = "src_region"
        row_region = "dst_region"
    else:
        row_region = "src_region"
        col_region = "dst_region"
    data = data.apply(to_numeric, axis=0)
    if isinstance(drop, str):
        drop = [drop]
    if drop:
        drop = list(set(drop).intersection(data.columns))
    if drop:
        data = data.drop(columns=drop)
    data = data.rename(
        columns=dict(
            Time="year",
            ProcessName="technology",
            RegionName=row_region,
            Commodity="commodity",
        )
    )
    indices = list(
        {"commodity", "year", "src_region", "dst_region", "technology"}.intersection(
            data.columns
        )
    )
    data = data.melt(
        id_vars={parameters}.union(indices).intersection(data.columns),
        var_name=col_region,
    )
    if parameters is None:
        result: Union[xr.DataArray, xr.Dataset] = xr.DataArray.from_series(
            data.set_index([*indices, col_region])["value"]
        ).rename(name)
    else:
        result = xr.Dataset.from_dataframe(
            data.pivot_table(
                values="value", columns=parameters, index=[*indices, col_region]
            ).rename(columns=camel_to_snake)
        )

    return result.rename(src_region="region")


def read_finite_resources(path: Union[str, Path]) -> xr.DataArray:
    """Reads finite resources from csv file.

    The CSV file is made up of columns "Region", "Year", as well
    as three timeslice columns ("Month", "Day", "Hour"). All three sets of columns are
    optional. The timeslice set should contain a full set of timeslices, if present.
    Other columns correspond to commodities.
    """
    from muse.timeslices import TIMESLICE

    data = pd.read_csv(path)
    data.columns = [c.lower() for c in data.columns]
    ts_levels = TIMESLICE.get_index("timeslice").names

    if set(data.columns).issuperset(ts_levels):
        timeslice = pd.MultiIndex.from_arrays(
            [data[u] for u in ts_levels], names=ts_levels
        )
        timeslice = pd.DataFrame(timeslice, columns=["timeslice"])
        data = pd.concat((data, timeslice), axis=1)
        data.drop(columns=ts_levels, inplace=True)
    indices = list({"year", "region", "timeslice"}.intersection(data.columns))
    data.set_index(indices, inplace=True)

    return xr.Dataset.from_dataframe(data).to_array(dim="commodity")


def check_utilization_and_minimum_service_factors(data, filename):
    if "utilization_factor" not in data.columns:
        raise ValueError(
            f"""A technology needs to have a utilization factor defined for every
             timeslice. Please check file {filename}."""
        )

    _check_utilization_not_all_zero(data, filename)
    _check_utilization_in_range(data, filename)

    if "minimum_service_factor" in data.columns:
        _check_minimum_service_factors_in_range(data, filename)
        _check_utilization_not_below_minimum(data, filename)


def _check_utilization_not_all_zero(data, filename):
    utilization_sum = data.groupby(["technology", "region", "year"]).sum()

    if (utilization_sum.utilization_factor == 0).any():
        raise ValueError(
            f"""A technology can not have a utilization factor of 0 for every
                timeslice. Please check file {filename}."""
        )


def _check_utilization_in_range(data, filename):
    utilization = data["utilization_factor"]
    if not np.all((0 <= utilization) & (utilization <= 1)):
        raise ValueError(
            f"""Utilization factor values must all be between 0 and 1 inclusive.
            Please check file {filename}."""
        )


def _check_utilization_not_below_minimum(data, filename):
    if (data["utilization_factor"] < data["minimum_service_factor"]).any():
        raise ValueError(f"""Utilization factors must all be greater than or equal to
                          their corresponding minimum service factors. Please check
                         {filename}.""")


def _check_minimum_service_factors_in_range(data, filename):
    min_service_factor = data["minimum_service_factor"]

    if not np.all((0 <= min_service_factor) & (min_service_factor <= 1)):
        raise ValueError(
            f"""Minimum service factor values must all be between 0 and 1 inclusive.
             Please check file {filename}."""
        )
