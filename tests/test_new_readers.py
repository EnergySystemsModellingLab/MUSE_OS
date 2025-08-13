import duckdb
import numpy as np
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


@fixture
def populate_sectors(default_new_input, con):
    from muse.new_input.readers import read_sectors_csv

    with open(default_new_input / "sectors.csv") as f:
        return read_sectors_csv(f, con)


@fixture
def populate_processes(default_new_input, con, populate_sectors):
    from muse.new_input.readers import read_processes_csv

    with open(default_new_input / "processes.csv") as f:
        return read_processes_csv(f, con)


@fixture
def populate_process_parameters(
    default_new_input, con, populate_regions, populate_processes
):
    from muse.new_input.readers import read_process_parameters_csv

    with open(default_new_input / "process_parameters.csv") as f:
        return read_process_parameters_csv(f, con)


@fixture
def populate_process_flows(
    default_new_input, con, populate_processes, populate_commodities, populate_regions
):
    from muse.new_input.readers import read_process_flows_csv

    with open(default_new_input / "process_flows.csv") as f:
        return read_process_flows_csv(f, con)


@fixture
def populate_agents(default_new_input, con, populate_regions, populate_sectors):
    from muse.new_input.readers import read_agents_csv

    with open(default_new_input / "agents.csv") as f:
        return read_agents_csv(f, con)


@fixture
def populate_agent_objectives(default_new_input, con, populate_agents):
    from muse.new_input.readers import read_agent_objectives_csv

    with open(default_new_input / "agent_objectives.csv") as f:
        return read_agent_objectives_csv(f, con)


@fixture
def populate_assets(
    default_new_input, con, populate_agents, populate_processes, populate_regions
):
    from muse.new_input.readers import read_assets_csv

    with open(default_new_input / "assets.csv") as f:
        return read_assets_csv(f, con)


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


def test_read_sectors_csv(populate_sectors):
    data = populate_sectors
    assert next(iter(data["id"])) == "gas"


def test_read_processes_csv(populate_processes):
    data = populate_processes
    assert next(iter(data["id"])) == "gassupply1"
    assert next(iter(data["sector"])) == "gas"


def test_read_process_parameters_csv(populate_process_parameters):
    data = populate_process_parameters
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["cap_par"])) == approx(0)
    assert next(iter(data["discount_rate"])) == approx(0.1)


def test_read_process_flows_csv(populate_process_flows):
    data = populate_process_flows
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["commodity"])) == "gas"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["coeff"])) == approx(1)


def test_read_agents_csv(populate_agents):
    data = populate_agents
    assert next(iter(data["id"])) == "A1_RES"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["sector"])) == "residential"
    assert next(iter(data["search_rule"])) == "all"
    assert next(iter(data["decision_rule"])) == "single"
    assert next(iter(data["quantity"])) == approx(1)


def test_read_agent_objectives_csv(populate_agent_objectives):
    data = populate_agent_objectives
    assert next(iter(data["agent"])) == "A1_RES"
    assert next(iter(data["objective_type"])) == "LCOE"
    assert next(iter(data["decision_weight"])) == approx(1)
    assert next(iter(data["objective_sort"])) is np.True_


def test_read_assets_csv(populate_assets):
    data = populate_assets
    assert next(iter(data["agent"])) == "A1_GAS"
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["commission_year"])) == 1995
    assert next(iter(data["capacity"])) == approx(7.5)
