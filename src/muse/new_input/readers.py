import duckdb
import numpy as np
import xarray as xr


def read_inputs(data_dir):
    data = {}
    con = duckdb.connect(":memory:")

    with open(data_dir / "commodities.csv") as f:
        commodities = read_commodities_csv(f, con)

    with open(data_dir / "commodity_trade.csv") as f:
        commodity_trade = read_commodity_trade_csv(f, con)  # noqa: F841

    with open(data_dir / "commodity_costs.csv") as f:
        commodity_costs = read_commodity_costs_csv(f, con)  # noqa: F841

    with open(data_dir / "demand.csv") as f:
        demand = read_demand_csv(f, con)  # noqa: F841

    with open(data_dir / "demand_slicing.csv") as f:
        demand_slicing = read_demand_slicing_csv(f, con)  # noqa: F841

    with open(data_dir / "regions.csv") as f:
        regions = read_regions_csv(f, con)  # noqa: F841

    data["global_commodities"] = calculate_global_commodities(commodities)
    return data


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


def read_commodity_trade_csv(buffer_, con):
    sql = """CREATE TABLE commodity_trade (
    commodity VARCHAR REFERENCES commodities(name),
    region VARCHAR REFERENCES regions(name),
    year BIGINT,
    import DOUBLE,
    export DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO commodity_trade SELECT
            commodity, region, year, import, export FROM rel;""")
    return con.sql("SELECT * from commodity_trade").fetchnumpy()


def read_commodity_costs_csv(buffer_, con):
    sql = """CREATE TABLE commodity_costs (
    year BIGINT,
    region VARCHAR REFERENCES regions(name),
    commodity VARCHAR REFERENCES commodities(name),
    value DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO commodity_costs SELECT
            year, region, commodity_name, value FROM rel;""")
    return con.sql("SELECT * from commodity_costs").fetchnumpy()


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


def read_demand_slicing_csv(buffer_, con):
    sql = """CREATE TABLE demand_slicing (
    commodity VARCHAR REFERENCES commodities(name),
    region VARCHAR REFERENCES regions(name),
    timeslice VARCHAR,
    fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    year BIGINT,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO demand_slicing SELECT
            commodity, region, timeslice, fraction, year FROM rel;""")
    return con.sql("SELECT * from demand_slicing").fetchnumpy()


def read_regions_csv(buffer_, con):
    sql = """CREATE TABLE regions (
      name VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO regions SELECT name FROM rel;")
    return con.sql("SELECT name from regions").fetchnumpy()


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
