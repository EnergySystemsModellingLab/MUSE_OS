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
      id VARCHAR PRIMARY KEY,
      type VARCHAR CHECK (type IN ('energy', 'service', 'material', 'environmental')),
      unit VARCHAR,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO commodities SELECT id, type, unit FROM rel;")
    return con.sql("select * from commodities").fetchnumpy()


def read_commodity_trade_csv(buffer_, con):
    sql = """CREATE TABLE commodity_trade (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT,
    import DOUBLE,
    export DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO commodity_trade SELECT
            commodity_id, region_id, year, import, export FROM rel;""")
    return con.sql("SELECT * from commodity_trade").fetchnumpy()


def read_commodity_costs_csv(buffer_, con):
    sql = """CREATE TABLE commodity_costs (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT,
    value DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO commodity_costs SELECT
            commodity_id, region_id, year, value FROM rel;""")
    return con.sql("SELECT * from commodity_costs").fetchnumpy()


def read_demand_csv(buffer_, con):
    sql = """CREATE TABLE demand (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT,
    demand DOUBLE,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO demand SELECT commodity_id, region_id, year, demand FROM rel;")
    return con.sql("SELECT * from demand").fetchnumpy()


def read_demand_slicing_csv(buffer_, con):
    sql = """CREATE TABLE demand_slicing (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT,
    timeslice VARCHAR,
    fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO demand_slicing SELECT
            commodity_id, region_id, year, timeslice, fraction FROM rel;""")
    return con.sql("SELECT * from demand_slicing").fetchnumpy()


def read_regions_csv(buffer_, con):
    sql = """CREATE TABLE regions (
      id VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO regions SELECT id FROM rel;")
    return con.sql("SELECT * from regions").fetchnumpy()


def calculate_global_commodities(commodities):
    names = commodities["id"].astype(np.dtype("str"))
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
