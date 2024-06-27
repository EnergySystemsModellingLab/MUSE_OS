import duckdb
import numpy as np
import xarray as xr


def read_inputs(data_dir):
    data = {}
    con = duckdb.connect(":memory:")

    with open(data_dir / "regions.csv") as f:
        regions = read_regions_csv(f, con)  # noqa: F841

    with open(data_dir / "commodities.csv") as f:
        commodities = read_commodities_csv(f, con)

    with open(data_dir / "demand.csv") as f:
        demand = read_demand_csv(f, con)  # noqa: F841

    data["global_commodities"] = calculate_global_commodities(commodities)
    return data


def read_regions_csv(buffer_, con):
    sql = """CREATE TABLE regions (
      name VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO regions SELECT name FROM rel;")
    return con.sql("SELECT name from regions").fetchnumpy()


def read_commodities_csv(buffer_, con):
    sql = """CREATE TABLE commodities (
      name VARCHAR PRIMARY KEY,
      type VARCHAR CHECK (type IN ('energy', 'service', 'material', 'environmental')),
      unit VARCHAR,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO commodities SELECT name, type, unit FROM rel;")

    return con.sql("select name, type, unit from commodities").fetchnumpy()


def calculate_global_commodities(commodities):
    names = commodities["name"].astype(np.dtype("str"))
    types = commodities["type"].astype(np.dtype("str"))
    units = commodities["unit"].astype(np.dtype("str"))

    type_array = xr.DataArray(
        data=types, dims=["commodity"], coords=dict(commodity=names)
    )

    unit_array = xr.DataArray(
        data=units, dims=["commodity"], coords=dict(commodity=names)
    )

    data = xr.Dataset(data_vars=dict(type=type_array, unit=unit_array))
    return data


def read_demand_csv(buffer_, con):
    sql = """CREATE TABLE demand (
    year BIGINT,
    commodity VARCHAR REFERENCES commodities(name),
    region VARCHAR REFERENCES regions(name),
    demand DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO demand SELECT year, commodity_name, region, demand FROM rel;")
    return con.sql("SELECT * from demand").fetchnumpy()
