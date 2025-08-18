from pathlib import Path

import duckdb
from pytest import fixture


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
    con.sql("SELECT * FROM time_slices").fetchnumpy()


def test_read_regions_csv(con):
    con.sql("SELECT * FROM regions").fetchnumpy()


def test_read_commodities_csv(con):
    con.sql("SELECT * FROM commodities").fetchnumpy()


def test_read_commodity_costs_csv(con):
    con.sql("SELECT * FROM commodity_costs").fetchnumpy()


def test_read_demand_csv(con):
    con.sql("SELECT * FROM demand").fetchnumpy()


def test_read_demand_slicing_csv(con):
    con.sql("SELECT * FROM demand_slicing").fetchnumpy()


def test_read_sectors_csv(con):
    con.sql("SELECT * FROM sectors").fetchnumpy()


def test_read_processes_csv(con):
    con.sql("SELECT * FROM processes").fetchnumpy()


def test_read_process_parameters_csv(con):
    con.sql("SELECT * FROM process_parameters").fetchnumpy()


def test_read_process_flows_csv(con):
    con.sql("SELECT * FROM process_flows").fetchnumpy()


def test_read_process_availabilities_csv(con):
    con.sql("SELECT * FROM process_availabilities").fetchnumpy()


def test_read_agents_csv(con):
    con.sql("SELECT * FROM agents").fetchnumpy()


def test_read_agent_objectives_csv(con):
    con.sql("SELECT * FROM agent_objectives").fetchnumpy()


def test_read_assets_csv(con):
    con.sql("SELECT * FROM assets").fetchnumpy()


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


def test_process_io_technodata(con):
    from muse.new_input.readers import process_io_technodata

    process_io_technodata(con, sector="power")


def test_process_initial_capacity(con):
    from muse.new_input.readers import process_initial_capacity

    process_initial_capacity(con, sector="power")
