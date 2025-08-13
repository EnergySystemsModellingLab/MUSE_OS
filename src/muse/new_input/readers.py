import duckdb


def read_inputs(data_dir) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")

    with open(data_dir / "time_slices.csv") as f:
        _time_slices = read_time_slices_csv(f, con)

    with open(data_dir / "regions.csv") as f:
        _regions = read_regions_csv(f, con)

    with open(data_dir / "sectors.csv") as f:
        _sectors = read_sectors_csv(f, con)

    with open(data_dir / "commodities.csv") as f:
        _commodities = read_commodities_csv(f, con)

    with open(data_dir / "processes.csv") as f:
        _processes = read_processes_csv(f, con)

    with open(data_dir / "process_parameters.csv") as f:
        _process_parameters = read_process_parameters_csv(f, con)

    with open(data_dir / "process_flows.csv") as f:
        _process_flows = read_process_flows_csv(f, con)

    with open(data_dir / "agents.csv") as f:
        _agents = read_agents_csv(f, con)

    with open(data_dir / "agent_objectives.csv") as f:
        _agent_objectives = read_agent_objectives_csv(f, con)

    with open(data_dir / "assets.csv") as f:
        _assets = read_assets_csv(f, con)

    with open(data_dir / "commodity_costs.csv") as f:
        _commodity_costs = read_commodity_costs_csv(f, con)

    with open(data_dir / "demand.csv") as f:
        _demand = read_demand_csv(f, con)

    with open(data_dir / "demand_slicing.csv") as f:
        _demand_slicing = read_demand_slicing_csv(f, con)

    return con


def read_time_slices_csv(buffer_, con):
    sql = """
    CREATE TABLE time_slices (
      id VARCHAR PRIMARY KEY,
      season VARCHAR,
      day VARCHAR,
      time_of_day VARCHAR,
      fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1)
    );
    """
    con.sql(sql)

    # Read CSV into a temporary relation
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841

    # Insert into the table with computed id
    con.sql("""
        INSERT INTO time_slices
        SELECT
            season || '.' || day || '.' || time_of_day AS id,
            season,
            day,
            time_of_day,
            fraction
        FROM rel
    """)

    return con.sql("SELECT * FROM time_slices").fetchnumpy()


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
    time_slice VARCHAR REFERENCES time_slices(id),
    fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    PRIMARY KEY (commodity, region, time_slice),
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO demand_slicing SELECT
            commodity_id, region_id, time_slice, fraction FROM rel;""")
    return con.sql("SELECT * from demand_slicing").fetchnumpy()


def read_sectors_csv(buffer_, con):
    sql = """CREATE TABLE sectors (
      id VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO sectors SELECT id FROM rel;")
    return con.sql("SELECT * from sectors").fetchnumpy()


def read_processes_csv(buffer_, con):
    sql = """CREATE TABLE processes (
      id VARCHAR PRIMARY KEY,
      sector VARCHAR REFERENCES sectors(id)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO processes SELECT id, sector_id FROM rel;")
    return con.sql("SELECT * from processes").fetchnumpy()


def read_process_parameters_csv(buffer_, con):
    sql = """CREATE TABLE process_parameters (
      process VARCHAR REFERENCES processes(id),
      region VARCHAR REFERENCES regions(id),
      year BIGINT,
      cap_par DOUBLE,
      fix_par DOUBLE,
      var_par DOUBLE,
      max_capacity_addition DOUBLE,
      max_capacity_growth DOUBLE,
      total_capacity_limit DOUBLE,
      lifetime DOUBLE,
      discount_rate DOUBLE,
      PRIMARY KEY (process, region, year)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        """
        INSERT INTO process_parameters SELECT
          process_id,
          region_id,
          year,
          cap_par,
          fix_par,
          var_par,
          max_capacity_addition,
          max_capacity_growth,
          total_capacity_limit,
          lifetime,
          discount_rate
        FROM rel;
        """
    )
    return con.sql("SELECT * from process_parameters").fetchnumpy()


def read_process_flows_csv(buffer_, con):
    sql = """CREATE TABLE process_flows (
      process VARCHAR REFERENCES processes(id),
      commodity VARCHAR REFERENCES commodities(id),
      region VARCHAR REFERENCES regions(id),
      year BIGINT,
      coeff DOUBLE,
      PRIMARY KEY (process, commodity, region, year)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        """
        INSERT INTO process_flows SELECT
          process_id,
          commodity_id,
          region_id,
          year,
          coeff
        FROM rel;
        """
    )
    return con.sql("SELECT * from process_flows").fetchnumpy()


def read_agents_csv(buffer_, con):
    sql = """CREATE TABLE agents (
      id VARCHAR PRIMARY KEY,
      region VARCHAR REFERENCES regions(id),
      sector VARCHAR REFERENCES sectors(id),
      search_rule VARCHAR,
      decision_rule VARCHAR,
      quantity DOUBLE
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        """
        INSERT INTO agents SELECT
          id,
          region_id,
          sector_id,
          search_rule,
          decision_rule,
          quantity
        FROM rel;
        """
    )
    return con.sql("SELECT * from agents").fetchnumpy()


def read_agent_objectives_csv(buffer_, con):
    sql = """CREATE TABLE agent_objectives (
      agent VARCHAR REFERENCES agents(id),
      objective_type VARCHAR,
      decision_weight DOUBLE,
      objective_sort BOOLEAN,
      PRIMARY KEY (agent, objective_type)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        """
        INSERT INTO agent_objectives SELECT
          agent_id,
          objective_type,
          decision_weight,
          objective_sort
        FROM rel;
        """
    )
    return con.sql("SELECT * from agent_objectives").fetchnumpy()


def read_assets_csv(buffer_, con):
    sql = """CREATE TABLE assets (
      agent VARCHAR REFERENCES agents(id),
      process VARCHAR REFERENCES processes(id),
      region VARCHAR REFERENCES regions(id),
      commission_year BIGINT,
      capacity DOUBLE,
      PRIMARY KEY (agent, process, region, commission_year)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql(
        """
        INSERT INTO assets SELECT
          agent_id,
          process_id,
          region_id,
          commission_year,
          capacity
        FROM rel;
        """
    )
    return con.sql("SELECT * from assets").fetchnumpy()
