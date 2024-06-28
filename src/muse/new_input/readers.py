import numpy as np
import xarray as xr
from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class TableBase(DeclarativeBase):
    pass


class Regions(TableBase):
    __tablename__ = "regions"

    name: Mapped[str] = mapped_column(primary_key=True)


class Commodities(TableBase):
    __tablename__ = "commodities"

    name: Mapped[str] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(
        CheckConstraint("type IN ('energy', 'service', 'material', 'environmental')")
    )
    unit: Mapped[str]


class Demand(TableBase):
    __tablename__ = "demand"

    year: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    commodity: Mapped[Commodities] = mapped_column(
        ForeignKey("commodities.name"), primary_key=True
    )
    region: Mapped[Regions] = mapped_column(
        ForeignKey("regions.name"), primary_key=True
    )
    demand: Mapped[float]


def read_inputs(data_dir):
    from sqlalchemy import create_engine

    engine = create_engine("duckdb:///:memory:")
    TableBase.metadata.create_all(engine)
    con = engine.raw_connection().driver_connection

    with open(data_dir / "regions.csv") as f:
        regions = read_regions_csv(f, con)  # noqa: F841

    with open(data_dir / "commodities.csv") as f:
        commodities = read_commodities_csv(f, con)

    with open(data_dir / "demand.csv") as f:
        demand = read_demand_csv(f, con)  # noqa: F841

    data = {}
    data["global_commodities"] = calculate_global_commodities(commodities)
    return data


def read_regions_csv(buffer_, con):
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.execute("INSERT INTO regions SELECT name FROM rel;")
    return con.sql("SELECT name from regions").fetchnumpy()


def read_commodities_csv(buffer_, con):
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO commodities SELECT name, type, unit FROM rel;")
    return con.sql("select name, type, unit from commodities").fetchnumpy()


def calculate_global_commodities(commodities):
    names = commodities["name"].astype(np.dtype("str"))
    types = commodities["type"].astype(np.dtype("str"))
    units = commodities["unit"].astype(np.dtype("str"))

    type_array = xr.DataArray(
        data=types, dims=["commodity"], coords=dict(commodity=names)
    )

    unit_array = xr.DataArray(
        data=units, dims=["commodity"], coords=dict(commodity=names)
    )

    data = xr.Dataset(data_vars=dict(type=type_array, unit=unit_array))
    return data


def read_demand_csv(buffer_, con):
    rel = con.read_csv(buffer_, header=True, delimiter=",")  # noqa: F841
    con.sql("INSERT INTO demand SELECT year, commodity_name, region, demand FROM rel;")
    return con.sql("SELECT * from demand").fetchnumpy()
