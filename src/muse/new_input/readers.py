import duckdb


def read_inputs(data_dir) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    load_order = [
        ("time_slices.csv", read_time_slices_csv),
        ("regions.csv", read_regions_csv),
        ("sectors.csv", read_sectors_csv),
        ("commodities.csv", read_commodities_csv),
        ("processes.csv", read_processes_csv),
        ("process_parameters.csv", read_process_parameters_csv),
        ("process_flows.csv", read_process_flows_csv),
        ("agents.csv", read_agents_csv),
        ("agent_objectives.csv", read_agent_objectives_csv),
        ("assets.csv", read_assets_csv),
        ("commodity_costs.csv", read_commodity_costs_csv),
        ("demand.csv", read_demand_csv),
        ("demand_slicing.csv", read_demand_slicing_csv),
    ]

    for filename, reader in load_order:
        with open(data_dir / filename) as f:
            reader(f, con)

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


def read_commodities_csv(buffer_, con):
    sql = """CREATE TABLE commodities (
      id VARCHAR PRIMARY KEY,
      type VARCHAR CHECK (type IN ('energy', 'service', 'material', 'environmental')),
      unit VARCHAR
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO commodities SELECT id, type, unit FROM rel;")


def read_regions_csv(buffer_, con):
    sql = """CREATE TABLE regions (
      id VARCHAR PRIMARY KEY
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO regions SELECT id FROM rel;")


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


def read_demand_slicing_csv(buffer_, con):
    sql = """CREATE TABLE demand_slicing (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    time_slice VARCHAR REFERENCES time_slices(id),
    fraction DOUBLE CHECK (fraction >= 0 AND fraction <= 1),
    PRIMARY KEY (commodity, region, time_slice)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("""INSERT INTO demand_slicing SELECT
            commodity_id, region_id, time_slice, fraction FROM rel;""")


def read_sectors_csv(buffer_, con):
    sql = """CREATE TABLE sectors (
      id VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO sectors SELECT id FROM rel;")


def read_processes_csv(buffer_, con):
    sql = """CREATE TABLE processes (
      id VARCHAR PRIMARY KEY,
      sector VARCHAR REFERENCES sectors(id)
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO processes SELECT id, sector_id FROM rel;")


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
