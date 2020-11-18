"""Ensemble of functions to read MUSE data."""
__all__ = [
    "read_technodictionary",
    "read_io_technodata",
<<<<<<< HEAD
    "read_initial_assets",
=======
    "read_initial_capacity",
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
from typing import Hashable, List, Optional, Sequence, Text, Union, cast

import numpy as np
import pandas as pd
import xarray as xr
=======
from typing import List, Optional, Sequence, Text, Union

from xarray import DataArray, Dataset
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

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


<<<<<<< HEAD
def read_technodictionary(filename: Union[Text, Path]) -> xr.Dataset:
=======
def read_technodictionary(filename: Union[Text, Path]) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    """Reads and formats technodata into a dataset.

    There are three axes: technologies, regions, and year.
    """
    from re import sub
<<<<<<< HEAD
=======
    from pandas import MultiIndex, read_csv, to_numeric
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from muse.readers import camel_to_snake

    def to_agent_share(name):
        return sub(r"agent(\d)", r"agent_share_\1", name)

<<<<<<< HEAD
    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    csv.drop(csv.filter(regex="Unname"), axis=1, inplace=True)
    csv = (
        csv.rename(columns=camel_to_snake)
        .rename(columns=to_agent_share)
        .rename(columns={"end_use": "enduse", "availabiliy year": "availability"})
    )
    data = csv[csv.process_name != "Unit"]
    ts = pd.MultiIndex.from_arrays(
=======
    csv = read_csv(filename, float_precision="high", low_memory=False)
    csv = csv.rename(columns=camel_to_snake)
    csv = csv.rename(columns=to_agent_share)
    csv = csv.rename(columns={"end_use": "enduse", "availabiliy year": "availability"})
    data = csv[csv.process_name != "Unit"]
    ts = MultiIndex.from_arrays(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        [data.process_name, data.region_name, [int(u) for u in data.time]],
        names=("technology", "region", "year"),
    )
    data.index = ts
    data.columns.name = "technodata"
    data.index.name = "technology"
    data = data.drop(["process_name", "region_name", "time"], axis=1)

<<<<<<< HEAD
    data = data.apply(lambda x: pd.to_numeric(x, errors="ignore"), axis=0)

    result = xr.Dataset.from_dataframe(data.sort_index())
=======
    data = data.apply(lambda x: to_numeric(x, errors="ignore"), axis=0)

    result = Dataset.from_dataframe(data.sort_index())
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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


<<<<<<< HEAD
def read_io_technodata(filename: Union[Text, Path]) -> xr.Dataset:
=======
def read_io_technodata(filename: Union[Text, Path]) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    """Reads process inputs or ouputs.

    There are four axes: (technology, region, year, commodity)
    """
<<<<<<< HEAD
    from muse.readers import camel_to_snake
    from functools import partial

    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    data = csv[csv.ProcessName != "Unit"]

    region = np.array(data.RegionName, dtype=str)
=======
    from pandas import read_csv, MultiIndex, to_numeric
    from xarray import concat
    from numpy import array
    from muse.readers import camel_to_snake

    csv = read_csv(filename, float_precision="high", low_memory=False)
    data = csv[csv.ProcessName != "Unit"]

    region = array(data.RegionName, dtype=str)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    process = data.ProcessName
    year = [int(u) for u in data.Time]

    data = data.drop(["ProcessName", "RegionName", "Time"], axis=1)

<<<<<<< HEAD
    ts = pd.MultiIndex.from_arrays(
=======
    ts = MultiIndex.from_arrays(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        [process, region, year], names=("technology", "region", "year")
    )
    data.index = ts
    data.columns.name = "commodity"
    data.index.name = "technology"

    data = data.rename(columns=camel_to_snake)
<<<<<<< HEAD
    data = data.apply(partial(pd.to_numeric, errors="ignore"), axis=0)

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
=======
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
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    result["flexible"] = result.flexible.fillna(0)

    # add units for flexible and fixed
    units = csv[csv.ProcessName == "Unit"].drop(
        ["ProcessName", "RegionName", "Time", "Level"], axis=1
    )
    units.index.name = "units"
    units.columns.name = "commodity"
<<<<<<< HEAD
    units = xr.DataArray(units).isel(units=0, drop=True)
=======
    units = DataArray(units).isel(units=0, drop=True)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    result["commodity_units"] = units
    return result


<<<<<<< HEAD
def read_initial_assets(filename: Union[Text, Path]) -> xr.DataArray:
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


def read_initial_capacity(data: Union[Text, Path, pd.DataFrame]) -> xr.DataArray:
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
=======
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
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


def read_technologies(
    technodata_path_or_sector: Optional[Union[Text, Path]] = None,
    comm_out_path: Optional[Union[Text, Path]] = None,
    comm_in_path: Optional[Union[Text, Path]] = None,
<<<<<<< HEAD
    commodities: Optional[Union[Text, Path, xr.Dataset]] = None,
    sectors_directory: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
) -> xr.Dataset:
=======
    commodities: Optional[Union[Text, Path, Dataset]] = None,
    sectors_directory: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
) -> Dataset:
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
=======
    from pathlib import Path
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
    if any(result[u].isnull().any() for u in result.data_vars):
        raise ValueError(f"Inconsistent data in {tpath} (e.g. inconsistent years)")
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    outs = read_io_technodata(opath).rename(
        flexible="flexible_outputs", fixed="fixed_outputs"
    )
    ins = read_io_technodata(ipath).rename(
        flexible="flexible_inputs", fixed="fixed_inputs"
    )
<<<<<<< HEAD
    if "year" in result.dims and len(result.year) > 1:
        if all(len(outs[d]) > 1 for d in outs.dims if outs[d].dtype.kind in "uifc"):
            outs = outs.interp(year=result.year)
        if all(len(ins[d]) > 1 for d in ins.dims if ins[d].dtype.kind in "uifc"):
            ins = ins.interp(year=result.year)
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    result = result.merge(outs).merge(ins)

    # try and add info about commodities
    if isinstance(commodities, (Text, Path)):
        try:
            commodities = read_global_commodities(commodities)
        except IOError:
            logger.warning("Could not load global commodities file.")
            commodities = None

<<<<<<< HEAD
    if isinstance(commodities, xr.Dataset):
=======
    if isinstance(commodities, Dataset):
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        if result.commodity.isin(commodities.commodity).all():
            result = result.merge(commodities.sel(commodity=result.commodity))
        else:
            logger.warn("Commodities missing in global commodities file.")

    result["comm_usage"] = "commodity", CommodityUsage.from_technologies(result)
    result = result.set_coords("comm_usage")
    if "comm_type" in result.data_vars or "comm_type" in result.coords:
        result = result.drop_vars("comm_type")

    return result


<<<<<<< HEAD
def read_csv_timeslices(path: Union[Text, Path], **kwargs) -> xr.DataArray:
    """Reads timeslice information from input."""
    from logging import getLogger

    getLogger(__name__).info("Reading timeslices from %s" % path)
    data = pd.read_csv(path, float_precision="high", **kwargs)
=======
def read_csv_timeslices(path: Union[Text, Path], **kwargs) -> DataArray:
    """Reads timeslice information from input."""
    from pandas import read_csv, MultiIndex
    from logging import getLogger

    getLogger(__name__).info("Reading timeslices from %s" % path)
    data = read_csv(path, float_precision="high", **kwargs)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    def snake_case(string):
        from re import sub

        result = sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r"-\1", string)
        return result.lower().strip()

    months = [snake_case(u) for u in data.Month.dropna()]
    days = [snake_case(u) for u in data.Day.dropna()]
    hours = [snake_case(u) for u in data.Hour.dropna()]
<<<<<<< HEAD
    ts_index = pd.MultiIndex.from_arrays(
        (months, days, hours), names=("month", "day", "hour")
    )
    result = xr.DataArray(
=======
    ts_index = MultiIndex.from_arrays(
        (months, days, hours), names=("month", "day", "hour")
    )
    result = DataArray(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        data.RepresentHours.dropna().astype(int),
        coords={"timeslice": ts_index},
        dims="timeslice",
        name="represent_hours",
    )
    result.coords["represent_hours"] = result
    return result.timeslice


<<<<<<< HEAD
def read_global_commodities(path: Union[Text, Path]) -> xr.Dataset:
    """Reads commodities information from input."""
=======
def read_global_commodities(path: Union[Text, Path]) -> Dataset:
    """Reads commodities information from input."""
    from pandas import read_csv
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if path.is_dir():
        path = path / "MuseGlobalCommodities.csv"
    if not path.is_file():
        raise IOError(f"File {path} does not exist.")

    getLogger(__name__).info(f"Reading global commodities from {path}.")

<<<<<<< HEAD
    data = pd.read_csv(path, float_precision="high", low_memory=False)
=======
    data = read_csv(path, float_precision="high", low_memory=False)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
    return xr.Dataset(data)
=======
    return Dataset(data)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1


def read_timeslice_shares(
    path: Union[Text, Path] = DEFAULT_SECTORS_DIRECTORY,
    sector: Optional[Text] = None,
<<<<<<< HEAD
    timeslice: Union[Text, Path, xr.DataArray] = "Timeslices{sector}.csv",
) -> xr.Dataset:
    """Reads sliceshare information into a xr.Dataset.
=======
    timeslice: Union[Text, Path, DataArray] = "Timeslices{sector}.csv",
) -> Dataset:
    """Reads sliceshare information into a Dataset.
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    Additionaly, this function will try and recover the timeslice multi- index from a
    import file "Timeslices{sector}.csv" in the same directory as the timeslice shares.
    Pass `None` if this behaviour is not required.
    """
    from re import match
<<<<<<< HEAD
=======
    from pandas import read_csv, MultiIndex
    from xarray import DataArray
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
    data = pd.read_csv(share_path, float_precision="high", low_memory=False)
    data.index = pd.MultiIndex.from_arrays(
=======
    data = read_csv(share_path, float_precision="high", low_memory=False)
    data.index = MultiIndex.from_arrays(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        (data.RegionName, data.SN), names=("region", "timeslice")
    )
    data.index.name = "rt"
    data = data.drop(["RegionName", "SN"], axis=1)
    data.columns.name = "commodity"

<<<<<<< HEAD
    result = xr.DataArray(data).unstack("rt").to_dataset(name="shares")

    if timeslice is None:
        result = result.drop_vars("timeslice")
    elif isinstance(timeslice, xr.DataArray) and hasattr(timeslice, "timeslice"):
        result["timeslice"] = timeslice.timeslice
        result[cast(Hashable, timeslice.name)] = timeslice
=======
    result = DataArray(data).unstack("rt").to_dataset(name="shares")

    if timeslice is None:
        result = result.drop_vars("timeslice")
    elif isinstance(timeslice, DataArray) and hasattr(timeslice, "timeslice"):
        result["timeslice"] = timeslice.timeslice
        result[timeslice.name] = timeslice
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    else:
        result["timeslice"] = timeslice
    return result.shares


def read_csv_agent_parameters(filename) -> List:
    """Reads standard MUSE agent-declaration csv-files.

    Returns a list of dictionaries, where each dictionary can be used to instantiate an
<<<<<<< HEAD
    agent in :py:func:`muse.agents.factories.factory`.
    """
=======
    agent in `muse.agent.factory`.
    """
    from pandas import read_csv
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from re import sub

    if (
        isinstance(filename, Text)
        and Path(filename).suffix != ".csv"
        and not Path(filename).is_file()
    ):
        filename = find_sectors_file(f"BuildingAgent{filename}.csv", filename)

<<<<<<< HEAD
    data = pd.read_csv(filename, float_precision="high", low_memory=False)
=======
    data = read_csv(filename, float_precision="high", low_memory=False)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    if "AgentNumber" in data.columns:
        data = data.drop(["AgentNumber"], axis=1)
    result = []

    # We remove rows with missing information, and next over the rest
    for _, row in data.iterrows():
<<<<<<< HEAD
        objectives = row[[i.startswith("Objective") for i in row.index]]
        floats = row[[i.startswith("ObjData") for i in row.index]]
        sorting = row[[i.startswith("Objsort") for i in row.index]]
        if len(objectives) != len(floats) or len(objectives) != len(sorting):
            raise ValueError("Objective, ObjData, and Objsort columns are inconsistent")
=======
        objectives = [row.Objective1, row.Objective2, row.Objective3]
        floats = [row.ObjData1, row.ObjData2, row.ObjData3]
        sorting = [row.Objsort1, row.Objsort2, row.Objsort3]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        decision_params = [
            u for u in zip(objectives, sorting, floats) if isinstance(u[0], Text)
        ]
        agent_type = {
            "new": "newcapa",
            "newcapa": "newcapa",
            "retrofit": "retrofit",
            "retro": "retrofit",
<<<<<<< HEAD
            "agent": "agent",
            "default": "agent",
        }[getattr(row, "Type", "agent").lower()]
=======
        }[row.Type.lower()]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
        if agent_type != "newcapa":
=======
        if agent_type == "retrofit":
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            data["share"] = sub(r"Agent(\d)", r"agent_share_\1", row.AgentShare)
        if agent_type == "retrofit" and data["decision"] == "lexo":
            data["decision"] = "retro_lexo"
        result.append(data)
    return result


<<<<<<< HEAD
def read_macro_drivers(path: Union[Text, Path]) -> xr.Dataset:
    """Reads a standard MUSE csv file for macro drivers."""
=======
def read_macro_drivers(path: Union[Text, Path]) -> Dataset:
    """Reads a standard MUSE csv file for macro drivers."""
    from pandas import read_csv
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from logging import getLogger

    path = Path(path)

    getLogger(__name__).info(f"Reading macro drivers from {path}")

<<<<<<< HEAD
    table = pd.read_csv(path, float_precision="high", low_memory=False)
=======
    table = read_csv(path, float_precision="high", low_memory=False)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    table.index = table.RegionName
    table.index.name = "region"
    table.columns.name = "year"
    table = table.drop(["Unit", "RegionName"], axis=1)

    population = table[table.Variable == "Population"]
    population = population.drop("Variable", axis=1)
    gdp = table[table.Variable == "GDP|PPP"].drop("Variable", axis=1)

<<<<<<< HEAD
    result = xr.Dataset({"gdp": gdp, "population": population})
=======
    result = Dataset({"gdp": gdp, "population": population})
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    result["year"] = "year", result.year.astype(int)
    result["region"] = "region", result.region.astype(str)
    return result


def read_initial_market(
<<<<<<< HEAD
    projections: Union[xr.DataArray, Path, Text],
    base_year_import: Optional[Union[Text, Path, xr.DataArray]] = None,
    base_year_export: Optional[Union[Text, Path, xr.DataArray]] = None,
    timeslices: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Read projections, import and export csv files."""
=======
    projections: Union[DataArray, Path, Text],
    base_year_import: Optional[Union[Text, Path, DataArray]] = None,
    base_year_export: Optional[Union[Text, Path, DataArray]] = None,
    timeslices: Optional[DataArray] = None,
) -> Dataset:
    """Read projections, import and export csv files."""
    from xarray import zeros_like
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
        base_year_export = xr.zeros_like(projections)
=======
        base_year_export = zeros_like(projections)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    # Base year import is optional. If it is not there, it's set to zero
    if isinstance(base_year_import, (Text, Path)):
        getLogger(__name__).info(f"Reading base year import from {base_year_import}")
        base_year_import = read_attribute_table(base_year_import)
    elif base_year_import is None:
        getLogger(__name__).info("Base year import not provided. Set to zero.")
<<<<<<< HEAD
        base_year_import = xr.zeros_like(projections)
=======
        base_year_import = zeros_like(projections)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

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

<<<<<<< HEAD
    result = xr.Dataset(
=======
    result = Dataset(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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


<<<<<<< HEAD
def read_attribute_table(path: Union[Text, Path]) -> xr.DataArray:
    """Read a standard MUSE csv file for price projections."""
=======
def read_attribute_table(path: Union[Text, Path]) -> DataArray:
    """Read a standard MUSE csv file for price projections."""
    from pandas import read_csv, MultiIndex
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise IOError(f"{path} does not exist.")

    getLogger(__name__).info(f"Reading prices from {path}")

<<<<<<< HEAD
    table = pd.read_csv(path, float_precision="high", low_memory=False)
=======
    table = read_csv(path, float_precision="high", low_memory=False)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    units = table.loc[0].drop(["RegionName", "Attribute", "Time"])
    table = table.drop(0)

    table.columns.name = "commodity"
    table = table.rename(
        columns={"RegionName": "region", "Attribute": "attribute", "Time": "year"}
    )

    region, year = table.region, table.year.astype(int)
    table = table.drop(["region", "year"], axis=1)
<<<<<<< HEAD
    table.index = pd.MultiIndex.from_arrays([region, year], names=["region", "year"])
=======
    table.index = MultiIndex.from_arrays([region, year], names=["region", "year"])
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    attribute = camel_to_snake(table.attribute.unique()[0])
    table = table.drop(["attribute"], axis=1)
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

<<<<<<< HEAD
    result = xr.DataArray(table, name=attribute).astype(float)
=======
    result = DataArray(table, name=attribute).astype(float)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    result = result.unstack("dim_0").fillna(0)

    result.coords["units_" + attribute] = ("commodity", units)

    return result


<<<<<<< HEAD
def read_regression_parameters(path: Union[Text, Path]) -> xr.Dataset:
    """Reads the regression parameters from a standard MUSE csv file."""
=======
def read_regression_parameters(path: Union[Text, Path]) -> Dataset:
    """Reads the regression parameters from a standard MUSE csv file."""
    from pandas import read_csv, MultiIndex
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from logging import getLogger
    from muse.readers import camel_to_snake

    path = Path(path)
    if not path.is_file():
        raise IOError(f"{path} does not exist or is not a file.")
    getLogger(__name__).info(f"Reading regression parameters from {path}.")
<<<<<<< HEAD
    table = pd.read_csv(path, float_precision="high", low_memory=False)
=======
    table = read_csv(path, float_precision="high", low_memory=False)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

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
<<<<<<< HEAD
    table.index = pd.MultiIndex.from_arrays(
        [sector, region], names=["sector", "region"]
    )
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    # Create a dataset, separating each type of coeeficient as a separate xr.DataArray
    coeffs = xr.Dataset(
        {
            k: xr.DataArray(table[table.coeff == k].drop("coeff", axis=1))
=======
    table.index = MultiIndex.from_arrays([sector, region], names=["sector", "region"])
    table = table.rename(columns={c: camel_to_snake(c) for c in table.columns})

    # Create a dataset, separating each type of coeeficient as a separate DataArray
    coeffs = Dataset(
        {
            k: DataArray(table[table.coeff == k].drop("coeff", axis=1))
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
            for k in table.coeff.unique()
        }
    )

    # Unstack the multi-index into separate dimensions
    coeffs = coeffs.unstack("dim_0").fillna(0)

    # We pair each sector with its function type
    function_type = list(zip(*set(zip(sector, function_type))))
<<<<<<< HEAD
    function_type = xr.DataArray(
=======
    function_type = DataArray(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
) -> xr.Dataset:
    """Read standard MUSE output files for consumption or supply."""
    from re import match
=======
) -> Dataset:
    """Read standard MUSE output files for consumption or supply."""
    from re import match
    from pandas import MultiIndex, read_csv
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from muse.readers import camel_to_snake

    def expand_paths(path):
        from glob import glob

        if isinstance(paths, Text):
            return [Path(p) for p in glob(path)]
        return Path(path)

    if isinstance(paths, Text):
        allfiles = expand_paths(paths)
    else:
<<<<<<< HEAD
        allfiles = [expand_paths(p) for p in cast(Sequence, paths)]
=======
        allfiles = [expand_paths(p) for p in paths]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    if len(allfiles) == 0:
        raise IOError(f"No files found with paths {paths}")

    datas = {}
    for path in allfiles:
<<<<<<< HEAD
        data = pd.read_csv(path, low_memory=False)
        data = data.drop(columns=[k for k in drop if k in data.columns])
        data.index = pd.MultiIndex.from_arrays(
=======
        data = read_csv(path, low_memory=False)
        data = data.drop(columns=[k for k in drop if k in data.columns])
        data.index = MultiIndex.from_arrays(
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD
        datas[year] = xr.DataArray(data)

    result = (
        xr.Dataset(datas)
=======
        datas[year] = DataArray(data)

    result = (
        Dataset(datas)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
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
<<<<<<< HEAD


def read_trade(
    data: Union[pd.DataFrame, Text, Path],
    columns_are_source: bool = True,
    parameters: Optional[Text] = None,
    skiprows: Optional[Sequence[int]] = None,
    name: Optional[Text] = None,
    drop: Optional[Union[Text, Sequence[Text]]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """Read CSV table with source and destination regions."""
    from functools import partial
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
    data = data.apply(partial(pd.to_numeric, errors="ignore"), axis=0)
    if isinstance(drop, Text):
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
        result: Union[xr.DataArray, xr.Dataset] = (
            xr.DataArray.from_series(
                data.set_index(indices + [col_region])["value"]
            ).rename(name)
        )
    else:
        result = xr.Dataset.from_dataframe(
            data.pivot_table(
                values="value", columns=parameters, index=indices + [col_region]
            ).rename(columns=camel_to_snake)
        )
    return result.rename(src_region="region")


def read_finite_resources(path: Union[Text, Path]) -> xr.DataArray:
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
        data["timeslice"] = pd.MultiIndex.from_arrays([data[u] for u in ts_levels])
        data.drop(columns=ts_levels, inplace=True)
    indices = list({"year", "region", "timeslice"}.intersection(data.columns))
    data.set_index(indices, inplace=True)

    return xr.Dataset.from_dataframe(data).to_array(dim="commodity")
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
