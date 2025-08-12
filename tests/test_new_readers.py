import duckdb
import numpy as np
import xarray as xr
from pytest import approx, fixture


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
    populate_time_slices,
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
def populate_time_slices(default_new_input, con):
    from muse.new_input.readers import read_time_slices_csv

    with open(default_new_input / "time_slices.csv") as f:
        return read_time_slices_csv(f, con)


def test_read_time_slices_csv(populate_time_slices):
    data = populate_time_slices
    assert next(iter(data["season"])) == "all-year"
    assert next(iter(data["day"])) == "all-week"
    assert next(iter(data["time_of_day"])) == "night"
    assert next(iter(data["fraction"])) == approx(0.166667)


def test_read_regions_csv(populate_regions):
    assert populate_regions["id"] == np.array(["R1"])


def test_read_commodities_csv(populate_commodities):
    data = populate_commodities
    assert list(data["id"]) == ["electricity", "gas", "heat", "wind", "CO2f"]
    assert list(data["type"]) == ["energy"] * 5
    assert list(data["unit"]) == ["PJ"] * 4 + ["kt"]


def test_read_commodity_costs_csv(populate_commodity_costs):
    data = populate_commodity_costs
    # Only checking the first element of each array, as the table is large
    assert next(iter(data["commodity"])) == "electricity"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["value"])) == approx(19.5)


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
    assert np.all(data["fraction"] == np.array([0.1, 0.15, 0.1, 0.15, 0.3, 0.2]))


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
