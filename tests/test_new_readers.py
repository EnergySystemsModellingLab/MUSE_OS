from io import StringIO

import duckdb
import numpy as np
import xarray as xr
from pytest import approx, fixture, raises


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
    default_new_input,
    con,
    populate_regions,
    populate_commodities,
    populate_demand,
    populate_timeslices,
):
    from muse.new_input.readers import read_demand_slicing_csv

    with open(default_new_input / "demand_slicing.csv") as f:
        return read_demand_slicing_csv(f, con)


@fixture
def populate_regions(default_new_input, con):
    from muse.new_input.readers import read_regions_csv

    with open(default_new_input / "regions.csv") as f:
        return read_regions_csv(f, con)


@fixture
def populate_timeslices(default_new_input, con):
    from muse.new_input.readers import read_timeslices_csv

    with open(default_new_input / "timeslices.csv") as f:
        return read_timeslices_csv(f, con)


def test_read_timeslices_csv(populate_timeslices):
    data = populate_timeslices
    assert len(data["id"]) == 6
    assert next(iter(data["id"])) == 1
    assert next(iter(data["season"])) == "all"
    assert next(iter(data["day"])) == "all"
    assert next(iter(data["time_of_day"])) == "night"
    assert next(iter(data["fraction"])) == approx(0.1667)


def test_read_regions_csv(populate_regions):
    assert populate_regions["id"] == np.array(["R1"])


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


def test_read_demand_slicing_csv(populate_demand_slicing):
    data = populate_demand_slicing
    assert np.all(data["commodity"] == "heat")
    assert np.all(data["region"] == "R1")
    # assert np.all(data["timeslice"] == np.array([0, 1]))
    assert np.all(
        data["fraction"]
        == np.array([0.1, 0.15, 0.1, 0.15, 0.3, 0.2, 0.1, 0.15, 0.1, 0.15, 0.3, 0.2])
    )


def test_read_commodities_csv_type_constraint(con):
    from muse.new_input.readers import read_commodities_csv

    csv = StringIO("id,type,unit\nfoo,invalid,bar\n")
    with raises(duckdb.ConstraintException):
        read_commodities_csv(csv, con)


def test_read_demand_csv_commodity_constraint(
    con, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_demand_csv

    csv = StringIO("year,commodity_id,region_id,demand\n2020,invalid,R1,0\n")
    with raises(duckdb.ConstraintException, match=".*foreign key.*"):
        read_demand_csv(csv, con)


def test_read_demand_csv_region_constraint(con, populate_commodities, populate_regions):
    from muse.new_input.readers import read_demand_csv

    csv = StringIO("year,commodity_id,region_id,demand\n2020,heat,invalid,0\n")
    with raises(duckdb.ConstraintException, match=".*foreign key.*"):
        read_demand_csv(csv, con)


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


def test_calculate_demand(
    populate_commodities,
    populate_regions,
    populate_timeslices,
    populate_demand,
    populate_demand_slicing,
):
    from muse.new_input.readers import calculate_demand

    data = calculate_demand(
        populate_commodities,
        populate_regions,
        populate_timeslices,
        populate_demand,
        populate_demand_slicing,
    )

    assert isinstance(data, xr.DataArray)
    assert data.dtype == np.float64

    assert set(data.dims) == {"year", "commodity", "region", "timeslice"}
    assert set(data.coords["region"].values) == {"R1"}
    assert set(data.coords["timeslice"].values) == set(range(1, 7))
    assert set(data.coords["year"].values) == {2020, 2050}
    assert set(data.coords["commodity"].values) == {
        "electricity",
        "gas",
        "heat",
        "wind",
        "CO2f",
    }

    assert data.sel(year=2020, commodity="heat", region="R1", timeslice=1) == 1


def test_calculate_initial_market(
    populate_commodities,
    populate_regions,
    populate_timeslices,
    populate_commodity_trade,
    populate_commodity_costs,
):
    from muse.new_input.readers import calculate_initial_market

    data = calculate_initial_market(
        populate_commodities,
        populate_regions,
        populate_timeslices,
        populate_commodity_trade,
        populate_commodity_costs,
    )

    assert isinstance(data, xr.Dataset)
    assert set(data.dims) == {"region", "year", "commodity", "timeslice"}
    for dt in data.dtypes.values():
        assert dt == np.dtype("float64")
    assert set(data.coords["region"].values) == {"R1"}
    assert set(data.coords["year"].values) == set(range(2010, 2105, 5))
    assert set(data.coords["commodity"].values) == {
        "electricity",
        "gas",
        "heat",
        "CO2f",
        "wind",
    }
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

    assert set(data.coords["timeslice"].values) == set(
        zip(month_values, day_values, hour_values)
    )
    assert set(data.coords["month"].values) == set(month_values)
    assert set(data.coords["day"].values) == set(day_values)
    assert set(data.coords["hour"].values) == set(hour_values)

    assert all(var.coords.equals(data.coords) for var in data.data_vars.values())

    prices = data.data_vars["prices"]
    assert (
        approx(
            prices.sel(
                year=2010,
                region="R1",
                commodity="electricity",
                timeslice=("all-year", "all-week", "night"),
            ),
            abs=1e-4,
        )
        == 14.81481
    )

    exports = data.data_vars["export"]
    assert (
        exports.sel(
            year=2010,
            region="R1",
            commodity="electricity",
            timeslice=("all-year", "all-week", "night"),
        )
    ) == 0

    imports = data.data_vars["import"]
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
