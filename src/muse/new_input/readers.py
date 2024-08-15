import duckdb
import numpy as np
import pandas as pd
import xarray as xr


def read_inputs(data_dir):
    data = {}
    con = duckdb.connect(":memory:")

    with open(data_dir / "timeslices.csv") as f:
        timeslices = read_timeslices_csv(f, con)

    with open(data_dir / "commodities.csv") as f:
        commodities = read_commodities_csv(f, con)

    with open(data_dir / "regions.csv") as f:
        regions = read_regions_csv(f, con)

    with open(data_dir / "commodity_trade.csv") as f:
        commodity_trade = read_commodity_trade_csv(f, con)

    with open(data_dir / "commodity_costs.csv") as f:
        commodity_costs = read_commodity_costs_csv(f, con)

    with open(data_dir / "demand.csv") as f:
        demand = read_demand_csv(f, con)

    with open(data_dir / "demand_slicing.csv") as f:
        demand_slicing = read_demand_slicing_csv(f, con)

    data["global_commodities"] = calculate_global_commodities(commodities)
    data["demand"] = calculate_demand(
        commodities, regions, timeslices, demand, demand_slicing
    )
    data["initial_market"] = calculate_initial_market(
        commodities, regions, timeslices, commodity_trade, commodity_costs
    )
    return data


def read_timeslices_csv(buffer_, con):
    sql = """CREATE TABLE timeslices (
      id VARCHAR PRIMARY KEY,
      season VARCHAR,
      day VARCHAR,
      time_of_day VARCHAR,
      fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        "INSERT INTO timeslices SELECT id, season, day, time_of_day, fraction FROM rel;"
    )
    return con.sql("SELECT * from timeslices").fetchnumpy()


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


def read_regions_csv(buffer_, con):
    sql = """CREATE TABLE regions (
      id VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO regions SELECT id FROM rel;")
    return con.sql("SELECT * from regions").fetchnumpy()


def read_commodity_trade_csv(buffer_, con):
    sql = """CREATE TABLE commodity_trade (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT,
    import DOUBLE,
    export DOUBLE,
    PRIMARY KEY (commodity, region, year)
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
    PRIMARY KEY (commodity, region, year)
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
    PRIMARY KEY (commodity, region, year)
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
    timeslice VARCHAR REFERENCES timeslices(id),
    fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    PRIMARY KEY (commodity, region, year, timeslice),
    FOREIGN KEY (commodity, region, year) REFERENCES demand(commodity, region, year)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO demand_slicing SELECT
            commodity_id, region_id, year, timeslice_id, fraction FROM rel;""")
    return con.sql("SELECT * from demand_slicing").fetchnumpy()


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


def calculate_demand(
    commodities, regions, timeslices, demand, demand_slicing
) -> xr.DataArray:
    """Calculate demand data for all commodities, regions, years, and timeslices.

    Result: A DataArray with a demand value for every combination of:
    - commodity: all commodities specified in the commodities table
    - region: all regions specified in the regions table
    - year: all years specified in the demand table
    - timeslice: all timeslices specified in the timeslices table

    Checks:
    - If demand data is specified for one year, it must be specified for all years.
    - If demand is nonzero, slicing data must be present.
    - If slicing data is specified for a commodity/region/year, the sum of
    the fractions must be 1, and all timeslices must be present.

    Fills:
    - If demand data is not specified for a commodity/region combination, the demand is
    0 for all years and timeslices.

    Todo:
    - Interpolation to allow for missing years in demand data.
    - Ability to leave the year field blank in both tables to indicate all years
    - Allow slicing data to be missing -> demand is spread equally across timeslices
    - Allow more flexibility for timeslices (e.g. can specify "winter" to apply to all
    winter timeslices, or "all" to apply to all timeslices)
    """
    # Prepare dataframes
    df_demand = pd.DataFrame(demand).set_index(["commodity", "region", "year"])
    df_slicing = pd.DataFrame(demand_slicing).set_index(
        ["commodity", "region", "year", "timeslice"]
    )

    # DataArray dimensions
    all_commodities = commodities["id"].astype(np.dtype("str"))
    all_regions = regions["id"].astype(np.dtype("str"))
    all_years = df_demand.index.get_level_values("year").unique()
    all_timeslices = timeslices["id"].astype(np.dtype("str"))

    # CHECK: all years are specified for each commodity/region combination
    check_all_values_specified(df_demand, ["commodity", "region"], "year", all_years)

    # CHECK: if slicing data is present, all timeslices must be specified
    check_all_values_specified(
        df_slicing, ["commodity", "region", "year"], "timeslice", all_timeslices
    )

    # CHECK: timeslice fractions sum to 1
    check_timeslice_sum = df_slicing.groupby(["commodity", "region", "year"]).apply(
        lambda x: np.isclose(x["fraction"].sum(), 1)
    )
    if not check_timeslice_sum.all():
        raise DataValidationError

    # CHECK: if demand data >0, fraction data must be specified
    check_fraction_data_present = (
        df_demand[df_demand["demand"] > 0]
        .index.isin(df_slicing.droplevel("timeslice").index)
        .all()
    )
    if not check_fraction_data_present.all():
        raise DataValidationError

    # FILL: demand is zero if unspecified
    df_demand = df_demand.reindex(
        pd.MultiIndex.from_product(
            [all_commodities, all_regions, all_years],
            names=["commodity", "region", "year"],
        ),
        fill_value=0,
    )

    # FILL: slice data is zero if unspecified
    df_slicing = df_slicing.reindex(
        pd.MultiIndex.from_product(
            [all_commodities, all_regions, all_years, all_timeslices],
            names=["commodity", "region", "year", "timeslice"],
        ),
        fill_value=0,
    )

    # Create DataArray
    da_demand = df_demand.to_xarray()["demand"]
    da_slicing = df_slicing.to_xarray()["fraction"]
    data = da_demand * da_slicing
    return data


def calculate_initial_market(
    commodities, regions, timeslices, commodity_trade, commodity_costs
) -> xr.Dataset:
    """Calculate trade and price data for all commodities, regions and years.

    Result: A Dataset with variables:
    - prices
    - exports
    - imports
    - static_trade
    For every combination of:
    - commodity: all commodities specified in the commodities table
    - region: all regions specified in the regions table
    - year: all years specified in the commodity_costs table
    - timeslice (multiindex): all timeslices specified in the timeslices table

    Checks:
    - If trade data is specified for one year, it must be specified for all years.
    - If price data is specified for one year, it must be specified for all years.

    Fills:
    - If trade data is not specified for a commodity/region combination, imports and
    exports are both zero
    - If price data is not specified for a commodity/region combination, the price is
    zero

    """
    from muse.timeslices import QuantityType, convert_timeslice

    # Prepare dataframes
    df_trade = pd.DataFrame(commodity_trade).set_index(["commodity", "region", "year"])
    df_costs = (
        pd.DataFrame(commodity_costs)
        .set_index(["commodity", "region", "year"])
        .rename(columns={"value": "prices"})
    )
    df_timeslices = pd.DataFrame(timeslices).set_index(["season", "day", "time_of_day"])

    # DataArray dimensions
    all_commodities = commodities["id"].astype(np.dtype("str"))
    all_regions = regions["id"].astype(np.dtype("str"))
    all_years = df_costs.index.get_level_values("year").unique()

    # CHECK: all years are specified for each commodity/region combination
    check_all_values_specified(df_trade, ["commodity", "region"], "year", all_years)
    check_all_values_specified(df_costs, ["commodity", "region"], "year", all_years)

    # FILL: price is zero if unspecified
    df_costs = df_costs.reindex(
        pd.MultiIndex.from_product(
            [all_commodities, all_regions, all_years],
            names=["commodity", "region", "year"],
        ),
        fill_value=0,
    )

    # FILL: trade is zero if unspecified
    df_trade = df_trade.reindex(
        pd.MultiIndex.from_product(
            [all_commodities, all_regions, all_years],
            names=["commodity", "region", "year"],
        ),
        fill_value=0,
    )

    # Calculate static trade
    df_trade["static_trade"] = df_trade["export"] - df_trade["import"]

    # Create Data
    df_full = df_costs.join(df_trade)
    data = df_full.to_xarray()
    ts = df_timeslices.to_xarray()["fraction"]
    ts = ts.stack(timeslice=("season", "day", "time_of_day"))
    convert_timeslice(data, ts, QuantityType.EXTENSIVE)

    return data


class DataValidationError(ValueError):
    pass


def check_all_values_specified(
    df: pd.DataFrame, group_by_cols: list[str], column_name: str, values: list
) -> None:
    """Check that the required values are specified in a dataframe.

    Checks that a row exists for all specified values of column_name for each
    group in the grouped dataframe.
    """
    if not (
        df.groupby(group_by_cols)
        .apply(
            lambda x: (
                set(x.index.get_level_values(column_name).unique()) == set(values)
            )
        )
        .all()
    ).all():
        msg = ""  # TODO
        raise DataValidationError(msg)
