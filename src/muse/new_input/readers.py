import uuid

import duckdb
import pandas as pd
import xarray as xr

from muse.readers.csv import create_assets, create_multiindex, create_xarray_dataset

# Global mapping from dimension name to (source_table, source_column)
DIM_TO_SOURCE: dict[str, tuple[str, str]] = {
    "process": ("processes", "id"),
    "commodity": ("commodities", "id"),
    "region": ("regions", "id"),
    "year": ("years", "year"),
    "time_slice": ("time_slices", "id"),
}


def _expand_list_or_all(
    col: str,
    *,
    domain_table: str,
    domain_col: str,
    source_relation: str = "rel",
) -> str:
    """Return composable SQL that expands a column over 'all' or ';'-lists.

    - For scalar values (not 'all' and no ';'), rows are passed through.
    - For lists, rows are duplicated for each trimmed item.
    - For 'all', rows are joined to the full domain table; value comes from
      `domain_table.domain_col`.
    """
    col_text = f"CAST(s.{col} AS VARCHAR)"

    return f"""
    SELECT s.* REPLACE (s.{col} AS {col})
    FROM {source_relation} s
    WHERE lower({col_text}) <> 'all'
      AND POSITION(';' IN {col_text}) = 0
    UNION ALL
    SELECT s.* REPLACE (TRIM(item) AS {col})
    FROM {source_relation} s
    CROSS JOIN UNNEST(str_split({col_text}, ';')) AS t(item)
    WHERE POSITION(';' IN {col_text}) > 0
    UNION ALL
    SELECT s.* REPLACE (d.{domain_col} AS {col})
    FROM {source_relation} s
    JOIN {domain_table} d ON lower({col_text}) = 'all'
    """


def expand_years(source_relation: str = "rel") -> str:
    """Expand `year` over 'all' and ';'-lists."""
    return _expand_list_or_all(
        "year",
        domain_table="years",
        domain_col="year",
        source_relation=source_relation,
    )


def expand_regions(source_relation: str = "rel") -> str:
    """Expand `region_id` over 'all' and ';'-lists."""
    return _expand_list_or_all(
        "region_id",
        domain_table="regions",
        domain_col="id",
        source_relation=source_relation,
    )


def expand_time_slices(source_relation: str = "rel") -> str:
    """Expand `time_slice` over 'all' and ';'-lists."""
    return _expand_list_or_all(
        "time_slice",
        domain_table="time_slices",
        domain_col="id",
        source_relation=source_relation,
    )


def chain_expanders(source: str, *expanders) -> str:
    """Compose multiple expander functions over a source relation name/SQL."""
    sql = source
    for i, expander in enumerate(expanders):
        src = sql if i == 0 else f"({sql})"
        sql = expander(source_relation=src)
    return sql


def insert_from_csv(
    con: duckdb.DuckDBPyConnection,
    buffer_,
    insert_into: str,
    select_sql: str,
    expanders: tuple = (),
) -> None:
    """Standardize: CSV -> unique temp view -> optional expanders -> INSERT."""
    view_name = f"rel_{uuid.uuid4().hex}"
    rel = con.read_csv(buffer_, header=True, delimiter=",")
    rel.create(view_name)
    src_sql = chain_expanders(view_name, *expanders) if expanders else view_name
    wrapped_src = src_sql if not expanders else f"({src_sql}) AS unioned"
    con.sql(f"INSERT INTO {insert_into} {select_sql.format(src=wrapped_src)}")


def validate_coverage(
    con: duckdb.DuckDBPyConnection,
    table: str,
    dims: list[str],
    present: list[str] | None = None,
) -> None:
    """Validate that required combinations exist in `table`.

    - If `present` is None: requires full cartesian product across `dims`.
    - If `present` is provided: for each distinct `present` key in `table`,
      requires all combinations across `dims`.
    """
    for d in dims:
        if d not in DIM_TO_SOURCE:
            raise ValueError(f"Unsupported dim: {d}")

    select_cols: list[str] = []
    joins: list[str] = []

    if present:
        present_csv = ", ".join(present)
        joins.append(f"(SELECT DISTINCT {present_csv} FROM {table}) p")
        select_cols.extend([f"p.{c} AS {c}" for c in present])

    for d in dims:
        src_table, src_col = DIM_TO_SOURCE[d]
        select_cols.append(f"{src_table}.{src_col} AS {d}")
        joins.append(src_table)

    proj_cols = [*(present or []), *dims]
    proj = ", ".join(proj_cols)

    sql = f"""
    WITH a AS (
      SELECT {", ".join(select_cols)}
      FROM {" CROSS JOIN ".join(joins)}
    ),
    missing AS (
      SELECT {proj} FROM a
      EXCEPT
      SELECT {proj} FROM {table}
    )
    SELECT COUNT(*) FROM missing
    """
    if con.execute(sql).fetchone()[0]:
        raise ValueError("Missing required combinations across dims")


def fill_missing_dim_combinations(
    con: duckdb.DuckDBPyConnection,
    table: str,
    dims: list[str],
    value_column: str,
    fill_value: float,
) -> None:
    """Insert fill_value for any missing combinations across the given dims.

    Anchors on the first dim's present values to avoid generating rows for
    completely absent entities, then uses an EXCEPT comparison to find and
    insert missing keys.
    The target table must use these exact column names for the dims.
    """
    for d in dims:
        if d not in DIM_TO_SOURCE:
            raise ValueError(f"Unsupported dim: {d}")

    present_key = dims[0]
    proj = ", ".join(dims)
    # Build column list: present key from p, other dims from their sources
    present_cols_sql = f"p.{present_key} AS {present_key}"
    dim_cols_sql = ", ".join(
        [f"{DIM_TO_SOURCE[d][0]}.{DIM_TO_SOURCE[d][1]} AS {d}" for d in dims[1:]]
    )
    cols_sql = ", ".join([c for c in [present_cols_sql, dim_cols_sql] if c])
    # Build CROSS JOIN chain: present set, then each dim source table
    joins = [f"(SELECT DISTINCT {present_key} FROM {table}) p"]
    joins += [DIM_TO_SOURCE[d][0] for d in dims[1:]]
    joins_sql = " CROSS JOIN ".join(joins)

    sql = f"""
    WITH a AS (
      SELECT {cols_sql}
      FROM {joins_sql}
    ),
    missing AS (
      SELECT {proj} FROM a
      EXCEPT
      SELECT {proj} FROM {table}
    )
    INSERT INTO {table} ({proj}, {value_column})
    SELECT {proj}, ? FROM missing
    """
    con.execute(sql, [fill_value])


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
        ("process_availabilities.csv", read_process_availabilities_csv),
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
    year BIGINT REFERENCES years(year),
    value DOUBLE,
    PRIMARY KEY (commodity, region, year)
    );
    """
    con.sql(sql)
    insert_from_csv(
        con,
        buffer_,
        "commodity_costs(commodity, region, year, value)",
        "SELECT commodity_id, region_id, year, value FROM {src}",
        expanders=(expand_years, expand_regions),
    )

    # Validate coverage
    validate_coverage(
        con,
        table="commodity_costs",
        dims=["region", "year"],
        present=["commodity"],
    )

    # Insert data for missing commodities
    fill_missing_dim_combinations(
        con,
        table="commodity_costs",
        dims=["commodity", "region", "year"],
        value_column="value",
        fill_value=0.0,
    )


def read_demand_csv(buffer_, con):
    sql = """CREATE TABLE demand (
    commodity VARCHAR REFERENCES commodities(id),
    region VARCHAR REFERENCES regions(id),
    year BIGINT REFERENCES years(year),
    demand DOUBLE CHECK (demand >= 0),
    PRIMARY KEY (commodity, region, year)
    );
    """
    con.sql(sql)
    insert_from_csv(
        con,
        buffer_,
        "demand(commodity, region, year, demand)",
        "SELECT commodity_id, region_id, year, demand FROM {src}",
    )

    # Validate coverage
    validate_coverage(
        con,
        table="demand",
        dims=["region", "year"],
        present=["commodity"],
    )

    # Insert data for missing commodities
    fill_missing_dim_combinations(
        con,
        table="demand",
        dims=["commodity", "region", "year"],
        value_column="demand",
        fill_value=0.0,
    )


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
    insert_from_csv(
        con,
        buffer_,
        "demand_slicing(commodity, region, time_slice, fraction)",
        "SELECT commodity_id, region_id, time_slice, fraction FROM {src}",
        expanders=(expand_regions, expand_time_slices),
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
      year BIGINT REFERENCES years(year),
      cap_par DOUBLE CHECK (cap_par >= 0),
      fix_par DOUBLE CHECK (fix_par >= 0),
      var_par DOUBLE CHECK (var_par >= 0),
      max_capacity_addition DOUBLE CHECK (max_capacity_addition >= 0),
      max_capacity_growth DOUBLE CHECK (max_capacity_growth >= 0),
      total_capacity_limit DOUBLE CHECK (total_capacity_limit >= 0),
      lifetime DOUBLE CHECK (lifetime > 0),
      discount_rate DOUBLE CHECK (discount_rate >= 0),
      PRIMARY KEY (process, region, year)
    );
    """
    con.sql(sql)
    insert_from_csv(
        con,
        buffer_,
        (
            "process_parameters("
            "process, region, year, cap_par, fix_par, var_par, "
            "max_capacity_addition, max_capacity_growth, total_capacity_limit, "
            "lifetime, discount_rate)"
        ),
        (
            """
        SELECT
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
        FROM {src}
            """
        ),
        expanders=(expand_years, expand_regions),
    )

    # Validate coverage
    validate_coverage(
        con, table="process_parameters", dims=["process", "region", "year"]
    )


def read_process_flows_csv(buffer_, con):
    sql = """CREATE TABLE process_flows (
      process VARCHAR REFERENCES processes(id),
      commodity VARCHAR REFERENCES commodities(id),
      region VARCHAR REFERENCES regions(id),
      year BIGINT REFERENCES years(year),
      input_coeff DOUBLE CHECK (input_coeff >= 0),
      output_coeff DOUBLE CHECK (output_coeff >= 0),
      PRIMARY KEY (process, commodity, region, year)
    );
    """
    con.sql(sql)
    insert_from_csv(
        con,
        buffer_,
        "process_flows(process, commodity, region, year, input_coeff, output_coeff)",
        """
        SELECT
          process_id,
          commodity_id,
          region_id,
          year,
          CASE WHEN coeff < 0 THEN -coeff ELSE 0 END,
          CASE WHEN coeff > 0 THEN  coeff ELSE 0 END
        FROM {src}
        """,
        expanders=(expand_years, expand_regions),
    )

    # Validate coverage
    validate_coverage(
        con,
        table="process_flows",
        dims=["region", "year"],
        present=["process", "commodity"],
    )


def read_process_availabilities_csv(buffer_, con):
    sql = """CREATE TABLE process_availabilities (
      process VARCHAR REFERENCES processes(id),
      region VARCHAR REFERENCES regions(id),
      year BIGINT REFERENCES years(year),
      time_slice VARCHAR REFERENCES time_slices(id),
      limit_type VARCHAR CHECK (limit_type IN ('up','down')),
      value DOUBLE,
      PRIMARY KEY (process, region, year, time_slice, limit_type)
    );
    """
    con.sql(sql)
    insert_from_csv(
        con,
        buffer_,
        "process_availabilities(process, region, year, time_slice, limit_type, value)",
        """
        SELECT
          process_id,
          region_id,
          year,
          time_slice,
          limit_type,
          value
        FROM {src}
        """,
        expanders=(expand_years, expand_regions, expand_time_slices),
    )


def read_agents_csv(buffer_, con):
    sql = """CREATE TABLE agents (
      id VARCHAR PRIMARY KEY,
      region VARCHAR REFERENCES regions(id),
      sector VARCHAR REFERENCES sectors(id),
      search_rule VARCHAR,
      decision_rule VARCHAR,
      quantity DOUBLE CHECK (quantity >= 0 AND quantity <= 1)
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

    # Validate coverage across region/sector
    ensure_agents_region_sector_coverage(con)


def ensure_agents_region_sector_coverage(
    con: duckdb.DuckDBPyConnection, table: str = "agents"
) -> None:
    """Validate there is at least one agent for every (region, sector)."""
    query = f"""
    WITH full_grid AS (
      SELECT r.id AS region, s.id AS sector
      FROM regions r
      CROSS JOIN sectors s
    ),
    present AS (
      SELECT DISTINCT region, sector FROM {table}
    )
    SELECT COUNT(*) AS missing_count
    FROM full_grid fg
    LEFT JOIN present p
      ON p.region = fg.region AND p.sector = fg.sector
    WHERE p.region IS NULL
    """
    missing_count = con.execute(query).fetchone()[0]
    if missing_count:
        raise ValueError("agents must include at least one agent per (region, sector)")


def read_agent_objectives_csv(buffer_, con):
    sql = """CREATE TABLE agent_objectives (
      agent VARCHAR REFERENCES agents(id),
      objective_type VARCHAR,
      decision_weight DOUBLE CHECK (decision_weight >= 0 AND decision_weight <= 1),
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

    # Validate: each agent must have at least one objective
    if con.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM agents a
          WHERE a.id NOT IN (SELECT agent FROM agent_objectives)
        )
        """
    ).fetchone()[0]:
        raise ValueError("Each agent must have at least one objective")


def read_assets_csv(buffer_, con):
    sql = """CREATE TABLE assets (
      agent VARCHAR REFERENCES agents(id),
      process VARCHAR REFERENCES processes(id),
      region VARCHAR REFERENCES regions(id),
      commission_year BIGINT,
      capacity DOUBLE CHECK (capacity > 0),
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
          cc.region AS region,
          cc.year AS year,
          cc.commodity AS commodity,
          cc.value AS prices,
          (? || '/' || c.unit) AS units_prices,
          CAST(0.0 AS DOUBLE) AS exports,
          CAST(0.0 AS DOUBLE) AS imports,
          CAST(0.0 AS DOUBLE) AS static_trade
        FROM commodity_costs cc
        JOIN commodities c ON c.id = cc.commodity
        """,
        [currency],
    ).fetchdf()

    # Build dataset from prices
    df = create_multiindex(
        df,
        index_columns=["region", "year", "commodity"],
        index_names=["region", "year", "commodity"],
        drop_columns=True,
    )
    result = create_xarray_dataset(df)
    return result


def process_agent_parameters(con: duckdb.DuckDBPyConnection, sector: str) -> list[dict]:
    """Create a list of agent dictionaries for a sector from DB tables."""
    df = con.execute(
        """
        SELECT
          a.id AS name,
          a.region AS region,
          a.search_rule,
          a.decision_rule,
          a.quantity,
          LIST(o.objective_type)
            FILTER (WHERE o.objective_type IS NOT NULL) AS objectives,
          LIST(struct_pack(
            objective_type := o.objective_type,
            objective_sort := o.objective_sort,
            decision_weight := o.decision_weight
          ))
            FILTER (WHERE o.objective_type IS NOT NULL) AS decision_params
        FROM agents a
        LEFT JOIN agent_objectives o ON o.agent = a.id
        WHERE a.sector = ?
        GROUP BY 1,2,3,4,5
        ORDER BY 1
        """,
        [sector],
    ).fetchdf()

    result: list[dict] = []
    for _, r in df.iterrows():
        params = [
            (d["objective_type"], d["objective_sort"], d["decision_weight"])  # type: ignore[index]
            for d in (r["decision_params"] or [])
        ]
        result.append(
            {
                "name": r["name"],
                "region": r["region"],
                "objectives": (r["objectives"] or []),
                "search_rules": r["search_rule"],
                "decision": {"name": r["decision_rule"], "parameters": params},
                "quantity": r["quantity"],
            }
        )
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
