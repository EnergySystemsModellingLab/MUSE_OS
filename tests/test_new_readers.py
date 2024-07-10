from io import StringIO

import duckdb
import numpy as np
import xarray as xr
from pytest import approx, fixture, mark, raises


@fixture
def default_new_input(tmp_path):
    from muse.examples import copy_model

    copy_model("default_new_input", tmp_path)
    return tmp_path / "model"


@fixture
def con():
    return duckdb.connect(":memory:")


@fixture
def populate_commodities(default_new_input, con):
    from muse.new_input.readers import read_commodities_csv

    with open(default_new_input / "commodities.csv") as f:
        return read_commodities_csv(f, con)


@fixture
def populate_commodity_trade(
    default_new_input, con, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_commodity_trade_csv

    with open(default_new_input / "commodity_trade.csv") as f:
        return read_commodity_trade_csv(f, con)


@fixture
def populate_commodity_costs(
    default_new_input, con, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_commodity_costs_csv

    with open(default_new_input / "commodity_costs.csv") as f:
        return read_commodity_costs_csv(f, con)


@fixture
def populate_demand(default_new_input, con, populate_regions, populate_commodities):
    from muse.new_input.readers import read_demand_csv

    with open(default_new_input / "demand.csv") as f:
        return read_demand_csv(f, con)


@fixture
def populate_demand_slicing(
    default_new_input, con, populate_regions, populate_commodities
):
    from muse.new_input.readers import read_demand_slicing_csv

    with open(default_new_input / "demand_slicing.csv") as f:
        return read_demand_slicing_csv(f, con)


@fixture
def populate_regions(default_new_input, con):
    from muse.new_input.readers import read_regions_csv

    with open(default_new_input / "regions.csv") as f:
        return read_regions_csv(f, con)


def test_read_commodities_csv(populate_commodities):
    data = populate_commodities
    assert list(data["id"]) == ["electricity", "gas", "heat", "wind", "CO2f"]
    assert list(data["type"]) == ["energy"] * 5
    assert list(data["unit"]) == ["PJ"] * 4 + ["kt"]


def test_read_commodity_trade_csv(populate_commodity_trade):
    data = populate_commodity_trade
    assert data["commodity"].size == 0
    assert data["region"].size == 0
    assert data["year"].size == 0
    assert data["import"].size == 0
    assert data["export"].size == 0


def test_read_commodity_costs_csv(populate_commodity_costs):
    data = populate_commodity_costs
    # Only checking the first element of each array, as the table is large
    assert next(iter(data["commodity"])) == "electricity"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2010
    assert next(iter(data["value"])) == approx(14.81481)


def test_read_demand_csv(populate_demand):
    data = populate_demand
    assert np.all(data["year"] == np.array([2020, 2050]))
    assert np.all(data["commodity"] == np.array(["heat", "heat"]))
    assert np.all(data["region"] == np.array(["R1", "R1"]))
    assert np.all(data["demand"] == np.array([10, 30]))


def test_read_regions_csv(populate_regions):
    assert populate_regions["id"] == np.array(["R1"])


def test_calculate_global_commodities(populate_commodities):
    from muse.new_input.readers import calculate_global_commodities

    data = calculate_global_commodities(populate_commodities)

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"commodity"}
    for dt in data.dtypes.values():
        assert np.issubdtype(dt, np.dtype("str"))

    assert list(data.coords["commodity"].values) == list(populate_commodities["id"])
    assert list(data.data_vars["type"].values) == list(populate_commodities["type"])
    assert list(data.data_vars["unit"].values) == list(populate_commodities["unit"])


def test_read_global_commodities_type_constraint(default_new_input, con):
    from muse.new_input.readers import read_commodities_csv

    csv = StringIO("id,type,unit\nfoo,invalid,bar\n")
    with raises(duckdb.ConstraintException):
        read_commodities_csv(csv, con)


def test_read_demand_csv_commodity_constraint(
    default_new_input, con, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_demand_csv

    csv = StringIO("year,commodity_id,region_id,demand\n2020,invalid,R1,0\n")
    with raises(duckdb.ConstraintException, match=".*foreign key.*"):
        read_demand_csv(csv, con)


def test_read_demand_csv_region_constraint(
    default_new_input, con, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_demand_csv

    csv = StringIO("year,commodity_id,region_id,demand\n2020,heat,invalid,0\n")
    with raises(duckdb.ConstraintException, match=".*foreign key.*"):
        read_demand_csv(csv, con)


@mark.xfail
def test_demand_dataset(default_new_input):
    import duckdb
    from muse.new_input.readers import read_commodities, read_demand, read_regions

    con = duckdb.connect(":memory:")

    read_regions(default_new_input, con)
    read_commodities(default_new_input, con)
    data = read_demand(default_new_input, con)

    assert isinstance(data, xr.DataArray)
    assert data.dtype == np.float64

    assert set(data.dims) == {"year", "commodity", "region", "timeslice"}
    assert list(data.coords["region"].values) == ["R1"]
    assert list(data.coords["timeslice"].values) == list(range(1, 7))
    assert list(data.coords["year"].values) == [2020, 2050]
    assert set(data.coords["commodity"].values) == {
        "electricity",
        "gas",
        "heat",
        "wind",
        "CO2f",
    }

    assert data.sel(year=2020, commodity="electricity", region="R1", timeslice=0) == 1


@mark.xfail
def test_new_read_initial_market(default_new_input):
    from muse.new_input.readers import read_inputs

    all_data = read_inputs(default_new_input)
    data = all_data["initial_market"]

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"region", "year", "commodity", "timeslice"}
    assert dict(data.dtypes) == dict(
        prices=np.float64,
        exports=np.float64,
        imports=np.float64,
        static_trade=np.float64,
    )
    assert list(data.coords["region"].values) == ["R1"]
    assert list(data.coords["year"].values) == list(range(2010, 2105, 5))
    assert list(data.coords["commodity"].values) == [
        "electricity",
        "gas",
        "heat",
        "CO2f",
        "wind",
    ]
    month_values = ["all-year"] * 6
    day_values = ["all-week"] * 6
    hour_values = [
        "night",
        "morning",
        "afternoon",
        "early-peak",
        "late-peak",
        "evening",
    ]

    assert list(data.coords["timeslice"].values) == list(
        zip(month_values, day_values, hour_values)
    )
    assert list(data.coords["month"]) == month_values
    assert list(data.coords["day"]) == day_values
    assert list(data.coords["hour"]) == hour_values

    assert all(var.coords.equals(data.coords) for var in data.data_vars.values())

    prices = data.data_vars["prices"]
    assert approx(
        prices.sel(
            year=2010,
            region="R1",
            commodity="electricity",
            timeslice=("all-year", "all-week", "night"),
        )
        - 14.81481,
        abs=1e-4,
    )

    exports = data.data_vars["exports"]
    assert (
        exports.sel(
            year=2010,
            region="R1",
            commodity="electricity",
            timeslice=("all-year", "all-week", "night"),
        )
    ) == 0

    imports = data.data_vars["imports"]
    assert (
        imports.sel(
            year=2010,
            region="R1",
            commodity="electricity",
            timeslice=("all-year", "all-week", "night"),
        )
    ) == 0

    static_trade = data.data_vars["static_trade"]
    assert (
        static_trade.sel(
            year=2010,
            region="R1",
            commodity="electricity",
            timeslice=("all-year", "all-week", "night"),
        )
    ) == 0
