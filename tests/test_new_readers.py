from pathlib import Path

import duckdb
import numpy as np
from pytest import approx, fixture


@fixture
def default_new_input(tmp_path) -> Path:
    from muse.examples import copy_model

    copy_model("default_new_input", tmp_path)
    return tmp_path / "model"


@fixture
def con(default_new_input) -> duckdb.DuckDBPyConnection:
    from muse.new_input.readers import read_inputs

    return read_inputs(
        default_new_input, years=[2020, 2025, 2030, 2035, 2040, 2045, 2050]
    )


def test_read_time_slices_csv(con):
    data = con.sql("SELECT * FROM time_slices").fetchnumpy()
    assert next(iter(data["season"])) == "all-year"
    assert next(iter(data["day"])) == "all-week"
    assert next(iter(data["time_of_day"])) == "night"
    assert next(iter(data["fraction"])) == approx(0.166667)


def test_read_regions_csv(con):
    data = con.sql("SELECT * FROM regions").fetchnumpy()
    assert next(iter(data["id"])) == "R1"


def test_read_commodities_csv(con):
    data = con.sql("SELECT * FROM commodities").fetchnumpy()
    assert next(iter(data["id"])) == "electricity"
    assert next(iter(data["type"])) == "energy"
    assert next(iter(data["unit"])) == "PJ"


def test_read_commodity_costs_csv(con):
    data = con.sql("SELECT * FROM commodity_costs").fetchnumpy()
    assert next(iter(data["commodity"])) == "electricity"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["value"])) == approx(19.5)


def test_read_demand_csv(con):
    data = con.sql("SELECT * FROM demand").fetchnumpy()
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["commodity"])) == "heat"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["demand"])) == approx(10)


def test_read_demand_slicing_csv(con):
    data = con.sql("SELECT * FROM demand_slicing").fetchnumpy()
    assert next(iter(data["commodity"])) == "heat"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["fraction"])) == approx(0.1)


def test_read_sectors_csv(con):
    data = con.sql("SELECT * FROM sectors").fetchnumpy()
    assert next(iter(data["id"])) == "gas"


def test_read_processes_csv(con):
    data = con.sql("SELECT * FROM processes").fetchnumpy()
    assert next(iter(data["id"])) == "gassupply1"
    assert next(iter(data["sector"])) == "gas"


def test_read_process_parameters_csv(con):
    data = con.sql("SELECT * FROM process_parameters").fetchnumpy()
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["cap_par"])) == approx(0)
    assert next(iter(data["discount_rate"])) == approx(0.1)


def test_read_process_flows_csv(con):
    data = con.sql("SELECT * FROM process_flows").fetchnumpy()
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["commodity"])) == "gas"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["year"])) == 2020
    assert next(iter(data["coeff"])) == approx(1)


def test_read_agents_csv(con):
    data = con.sql("SELECT * FROM agents").fetchnumpy()
    assert next(iter(data["id"])) == "A1_RES"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["sector"])) == "residential"
    assert next(iter(data["search_rule"])) == "all"
    assert next(iter(data["decision_rule"])) == "single"
    assert next(iter(data["quantity"])) == approx(1)


def test_read_agent_objectives_csv(con):
    data = con.sql("SELECT * FROM agent_objectives").fetchnumpy()
    assert next(iter(data["agent"])) == "A1_RES"
    assert next(iter(data["objective_type"])) == "LCOE"
    assert next(iter(data["decision_weight"])) == approx(1)
    assert next(iter(data["objective_sort"])) is np.True_


def test_read_assets_csv(con):
    data = con.sql("SELECT * FROM assets").fetchnumpy()
    assert next(iter(data["agent"])) == "A1_GAS"
    assert next(iter(data["process"])) == "gassupply1"
    assert next(iter(data["region"])) == "R1"
    assert next(iter(data["commission_year"])) == 1995
    assert next(iter(data["capacity"])) == approx(7.5)


def test_process_global_commodities(con):
    from muse.new_input.readers import process_global_commodities

    process_global_commodities(con)


def test_process_technodictionary(con):
    from muse.new_input.readers import process_technodictionary

    process_technodictionary(con, sector="power")


def test_process_agent_parameters(con):
    from muse.new_input.readers import process_agent_parameters

    process_agent_parameters(con, sector="power")


def test_process_initial_market(con):
    from muse.new_input.readers import process_initial_market

    process_initial_market(con, currency="EUR")


def test_process_initial_capacity(con):
    from muse.new_input.readers import process_initial_capacity

    process_initial_capacity(con, sector="power")
