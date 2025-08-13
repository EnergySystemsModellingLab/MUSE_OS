import duckdb
import pandas as pd
import xarray as xr

from muse.readers.csv import create_assets, create_multiindex, create_xarray_dataset


def expand_years(source_relation: str = "rel") -> str:
    """Return a composable SQL that expands 'year' over 'all' or semicolon lists."""
    return f"""
    SELECT s.* REPLACE (CAST(s.year AS BIGINT) AS year)
    FROM {source_relation} s
    WHERE lower(CAST(s.year AS VARCHAR)) <> 'all' AND POSITION(';' IN CAST(s.year AS VARCHAR)) = 0
    UNION ALL
    SELECT s.* REPLACE (CAST(TRIM(item) AS BIGINT) AS year)
    FROM {source_relation} s
    CROSS JOIN UNNEST(str_split(CAST(s.year AS VARCHAR), ';')) AS t(item)
    WHERE POSITION(';' IN CAST(s.year AS VARCHAR)) > 0
    UNION ALL
    SELECT s.* REPLACE (y.year AS year)
    FROM {source_relation} s
    CROSS JOIN years y
    WHERE lower(CAST(s.year AS VARCHAR)) = 'all'
    """  # noqa: E501


def expand_regions(source_relation: str = "rel") -> str:
    """Return a composable SQL that expands 'region_id' over 'all' or lists."""
    return f"""
    SELECT s.*
    FROM {source_relation} s
    WHERE lower(CAST(s.region_id AS VARCHAR)) <> 'all' AND POSITION(';' IN CAST(s.region_id AS VARCHAR)) = 0
    UNION ALL
    SELECT s.* REPLACE (TRIM(item) AS region_id)
    FROM {source_relation} s
    CROSS JOIN UNNEST(str_split(CAST(s.region_id AS VARCHAR), ';')) AS t(item)
    WHERE POSITION(';' IN CAST(s.region_id AS VARCHAR)) > 0
    UNION ALL
    SELECT s.* REPLACE (r.id AS region_id)
    FROM {source_relation} s
    JOIN regions r ON lower(CAST(s.region_id AS VARCHAR)) = 'all'
    """  # noqa: E501


def expand_time_slices(source_relation: str = "rel") -> str:
    """Return a composable SQL that expands 'time_slice' over 'annual'."""
    return f"""
    SELECT s.*
    FROM {source_relation} s
    WHERE lower(CAST(s.time_slice AS VARCHAR)) <> 'annual'
    UNION ALL
    SELECT s.* REPLACE (t.id AS time_slice)
    FROM {source_relation} s
    JOIN time_slices t ON lower(CAST(s.time_slice AS VARCHAR)) = 'annual'
    """


def read_inputs(data_dir, years: list[int]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    insert_years(con, years)
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


def insert_years(con: duckdb.DuckDBPyConnection, years: list[int]):
    con.sql("CREATE TABLE years(year BIGINT PRIMARY KEY);")
    con.sql(f"INSERT INTO years VALUES {', '.join(f'({y})' for y in years)};")


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


def read_sectors_csv(buffer_, con):
    sql = """CREATE TABLE sectors (
      id VARCHAR PRIMARY KEY,
    );
    """
    con.sql(sql)
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO sectors SELECT id FROM rel;")


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
    years_sql = expand_years(source_relation="rel")
    regions_sql = expand_regions(source_relation=f"({years_sql})")
    expansion_sql = regions_sql
    con.sql(
        f"""INSERT INTO commodity_costs SELECT
            commodity_id, region_id, year, value FROM ({expansion_sql}) AS unioned;
        """
    )


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
    regions_sql = expand_regions(source_relation="rel")
    ts_sql = expand_time_slices(source_relation=f"({regions_sql})")
    expansion_sql = ts_sql
    con.sql(
        f"""INSERT INTO demand_slicing SELECT
            commodity_id, region_id, time_slice, fraction FROM ({expansion_sql}) AS unioned;
        """  # noqa: E501
    )


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
    years_sql = expand_years(source_relation="rel")
    regions_sql = expand_regions(source_relation=f"({years_sql})")
    expansion_sql = regions_sql
    con.sql(
        f"""
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
        FROM ({expansion_sql}) AS unioned;
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
    years_sql = expand_years(source_relation="rel")
    regions_sql = expand_regions(source_relation=f"({years_sql})")
    expansion_sql = regions_sql
    con.sql(
        f"""
        INSERT INTO process_flows SELECT
          process_id,
          commodity_id,
          region_id,
          year,
          coeff
        FROM ({expansion_sql}) AS unioned;
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


def process_global_commodities(con: duckdb.DuckDBPyConnection) -> xr.Dataset:
    """Create an xarray Dataset of global commodities from the `commodities` table."""
    df = con.sql(
        """
        SELECT
          id AS commodity,
          type AS commodity_type,
          unit
        FROM commodities
        """
    ).df()

    df.index = df["commodity"]
    df = df.drop(columns=["commodity"])
    df.index.name = "commodity"
    return create_xarray_dataset(df)


def process_technodictionary(con: duckdb.DuckDBPyConnection, sector: str) -> xr.Dataset:
    """Create an xarray Dataset analogous to technodictionary from DB tables.

    Uses `processes` and `process_parameters` to build variables over
    dimensions (technology, region, year).
    """
    df = con.execute(
        """
            SELECT
              p.id AS technology,
              pp.region,
              pp.year,
              pp.cap_par,
              pp.fix_par,
              pp.var_par,
              pp.max_capacity_addition,
              pp.max_capacity_growth,
              pp.total_capacity_limit,
              pp.lifetime AS technical_life,
              pp.discount_rate AS interest_rate
            FROM process_parameters pp
            JOIN processes p ON p.id = pp.process
            WHERE p.sector = ?
            """,
        [sector],
    ).fetchdf()

    df = create_multiindex(
        df,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )

    result = create_xarray_dataset(df)
    return result


def process_initial_market(con: duckdb.DuckDBPyConnection, currency: str) -> xr.Dataset:
    """Create initial market dataset with prices and zero trade variables.

    Args:
        con: DuckDB connection with tables loaded.
        currency: Currency string, e.g. "USD". Mandatory.
        years: List of years to cover. Missing combinations are filled with zero.

    Returns:
        xr.Dataset with dims (region, year, commodity) and variables
        prices, exports, imports, static_trade. Adds coordinate
        units_prices = f"{currency}/{unit}" per commodity.
    """
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must be a non-empty string")

    df = con.execute(
        """
        SELECT
          r.id AS region,
          y.year AS year,
          c.id AS commodity,
          COALESCE(cc.value, 0) AS prices,
          (? || '/' || c.unit) AS units_prices
        FROM regions r
        CROSS JOIN years y
        CROSS JOIN commodities c
        LEFT JOIN commodity_costs cc
          ON cc.region = r.id AND cc.year = y.year AND cc.commodity = c.id
        """,
        [currency],
    ).fetchdf()

    if df.empty:
        raise ValueError("No commodity cost data found to build initial market.")

    # Build dataset from prices
    prices_df = create_multiindex(
        df,
        index_columns=["region", "year", "commodity"],
        index_names=["region", "year", "commodity"],
        drop_columns=True,
    )
    result = create_xarray_dataset(prices_df)

    # Add zero trade variables (legacy)
    result["exports"] = xr.zeros_like(result["prices"]).rename("exports")
    result["imports"] = xr.zeros_like(result["prices"]).rename("imports")
    result["static_trade"] = (result["imports"] - result["exports"]).rename(
        "static_trade"
    )
    return result


def process_agent_parameters(con: duckdb.DuckDBPyConnection, sector: str) -> list[dict]:
    """Create a list of agent dictionaries for a sector from DB tables.

    The result matches the structure returned by the legacy CSV-based
    process_agent_parameters, but only includes the required fields:
    - name, region, objectives, search_rules, decision, quantity

    The following legacy fields are intentionally omitted: agent_type,
    share, maturity_threshold, spend_limit.
    """
    # Gather agent base data for the sector
    agents_df = con.execute(
        """
        SELECT id AS name,
               region AS region,
               search_rule,
               decision_rule,
               quantity
        FROM agents
        WHERE sector = ?
        """,
        [sector],
    ).fetchdf()

    # Gather objectives per agent
    objectives_df = con.execute(
        """
        SELECT agent AS name,
               objective_type,
               objective_sort,
               decision_weight
        FROM agent_objectives
        WHERE agent IN (SELECT id FROM agents WHERE sector = ?)
        ORDER BY name
        """,
        [sector],
    ).fetchdf()

    # Assemble result
    result: list[dict] = []
    for _, row in agents_df.iterrows():
        agent_name = row["name"]
        agent_objectives = objectives_df[objectives_df["name"] == agent_name]

        # Objectives list: in legacy, these are strings like 'LCOE'
        objectives = agent_objectives["objective_type"].tolist()

        # Decision parameters: tuples of
        # (objective_type, objective_sort, decision_weight)
        decision_params = list(
            zip(
                agent_objectives["objective_type"].tolist(),
                agent_objectives["objective_sort"].tolist(),
                agent_objectives["decision_weight"].tolist(),
            )
        )

        agent_dict = {
            "name": agent_name,
            "region": row["region"],
            "objectives": objectives,
            "search_rules": row["search_rule"],
            "decision": {"name": row["decision_rule"], "parameters": decision_params},
            "quantity": row["quantity"],
        }
        result.append(agent_dict)

    return result


def process_initial_capacity(
    con: duckdb.DuckDBPyConnection, sector: str
) -> xr.DataArray:
    """Create existing capacity over time from assets and lifetimes.

    Args:
        con: DuckDB connection
        sector: Sector name to filter processes
        years: List of years to include (no interpolation)

    Returns:
        xr.DataArray with dims (asset) and coordinates (asset, technology, region, year)
        showing capacity available in each year based on commission year and lifetime.
    """
    # Compute capacity trajectory per technology/region/year
    # Note: this sums up the capacity of all assets in the same technology/region
    # I think ideally we wouldn't do that and would keep these as separate assets
    # Also, this isn't taking into account agent ownership
    assets_df = con.execute(
        """
        WITH lifetimes AS (
            SELECT DISTINCT pp.process, pp.region, pp.lifetime
            FROM process_parameters pp
            JOIN processes p ON p.id = pp.process
            WHERE p.sector = ?
        ),
        assets_enriched AS (
            SELECT
              a.process AS technology,
              a.region,
              a.commission_year,
              a.capacity,
              lt.lifetime
            FROM assets a
            JOIN lifetimes lt
              ON lt.process = a.process AND lt.region = a.region
        )
        SELECT
          ae.technology,
          ae.region,
          y.year,
          SUM(
            CASE
              WHEN y.year >= ae.commission_year AND
                   y.year < (ae.commission_year + ae.lifetime)
              THEN ae.capacity ELSE 0
            END
          ) AS value
        FROM assets_enriched ae
        CROSS JOIN years y
        GROUP BY ae.technology, ae.region, y.year
        ORDER BY ae.technology, ae.region, y.year
        """,
        [sector],
    ).fetchdf()

    # If no assets, return an empty DataArray
    if assets_df.empty:
        return xr.DataArray([], dims=("asset",))

    df = pd.DataFrame(assets_df)
    df = create_multiindex(
        df,
        index_columns=["technology", "region", "year"],
        index_names=["technology", "region", "year"],
        drop_columns=True,
    )
    da = create_xarray_dataset(df).value.astype(float)

    da = create_assets(da)
    return da
