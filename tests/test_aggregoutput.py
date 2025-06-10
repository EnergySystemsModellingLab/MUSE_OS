from pandas import DataFrame, concat

from muse import examples
from muse.outputs.mca import _aggregate_sectors, sector_capacity


def _create_test_data(agents, years, sector_name):
    """Helper function to create test DataFrame from agent data."""
    frame = DataFrame()
    for agent in agents:
        for year in years:
            if year in agent.assets.year:
                capacity = agent.assets.capacity.sel(year=year).values
                if capacity > 0.0:
                    data = DataFrame(
                        {
                            "region": agent.region,
                            "agent": agent.name,
                            "type": agent.category,
                            "sector": sector_name,
                            "capacity": capacity[0],
                        },
                        index=[(year, agent.assets.technology.values[0])],
                    )
                    frame = concat([frame, data])
    return frame


def test_aggregate_sector():
    """Test aggregate_sector function with single sector data."""
    mca = examples.model("multiple_agents", test=True)
    years = [2020, 2025]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = sector_capacity(sector_list[0])

    frame = _create_test_data(
        agents=list(sector_list[0].agents), years=years, sector_name=sector_list[0].name
    )

    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (frame[columns].values == alldata[columns].values).all()


def test_aggregate_sectors():
    """Test aggregate_sectors function with multiple sectors."""
    mca = examples.model("multiple_agents", test=True)
    years = [2020, 2025, 2030]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = _aggregate_sectors(mca.sectors, op=sector_capacity)

    frame = DataFrame()
    for sector in sector_list:
        sector_frame = _create_test_data(
            agents=list(sector.agents), years=years, sector_name=sector.name
        )
        frame = concat([frame, sector_frame])

    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (frame[columns].values == alldata[columns].values).all()


def test_aggregate_sector_manyregions():
    """Test aggregate_sector function with multiple regions."""
    mca = examples.model("multiple_agents", test=True)
    residential = next(sector for sector in mca.sectors if sector.name == "residential")

    # Set up Belarus region for testing
    for agent in list(residential.agents)[:2]:
        agent.assets["region"] = "BELARUS"
        agent.region = "BELARUS"

    years = [2020, 2025, 2030]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = _aggregate_sectors(mca.sectors, op=sector_capacity)

    frame = DataFrame()
    for sector in sector_list:
        sector_frame = _create_test_data(
            agents=list(sector.agents), years=years, sector_name=sector.name
        )
        frame = concat([frame, sector_frame])

    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (frame[columns].values == alldata[columns].values).all()
