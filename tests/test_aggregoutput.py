from muse import examples
from muse.outputs.mca import sector_capacity, sectors_capacity


def test_aggregate_sector():
    """Test for aggregate_sector function check colum titles, number of
    agents/region/technologies and assets capacities."""
    from pandas import concat, DataFrame

    mca = examples.model("multiple-agents")
    year = [2020, 2025]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    agent_list = [list(a.agents) for a in sector_list]
    alldata = sector_capacity(sector_list[0])
    alldatadict = alldata.to_dict("split")
    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (sorted(columns)) == sorted(alldatadict["columns"])
    frame = DataFrame()
    for ai in agent_list[0]:

        for y in year:
            if y in ai.assets.year:
                if ai.assets.capacity.sel(year=y).values > 0.0:
                    data = DataFrame(
                        {
                            "region": ai.region,
                            "agent": ai.name,
                            "type": ai.category,
                            "sector": sector_list[0].name,
                            "capacity": ai.assets.capacity.sel(year=y).values[0],
                        },
                        index=[(y, ai.assets.technology.values[0])],
                    )
                    frame = concat([frame, data])
    print()
    assert (frame[columns].values == alldata[columns].values).all()


def test_aggregate_sectors():
    """Test for aggregate_sectors function."""
    from pandas import DataFrame, concat

    mca = examples.model("multiple-agents")
    year = [2020, 2025, 2030]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    agent_list = [list(a.agents) for a in sector_list]
    alldata = sectors_capacity(mca.sectors)
    alldatadict = alldata.to_dict("split")
    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (sorted(columns)) == sorted(alldatadict["columns"])
    frame = DataFrame()
    for a, ai in enumerate(agent_list):
        for ii in range(0, len(ai)):
            for y in year:
                if y in ai[ii].assets.year:
                    if ai[ii].assets.capacity.sel(year=y).values > 0.0:
                        data = DataFrame(
                            {
                                "region": ai[ii].region,
                                "agent": ai[ii].name,
                                "type": ai[ii].category,
                                "sector": sector_list[a].name,
                                "capacity": ai[ii]
                                .assets.capacity.sel(year=y)
                                .values[0],
                            },
                            index=[(y, ai[ii].assets.technology.values[0])],
                        )
                        frame = concat([frame, data])
                        print(frame, "frame, year)")

    assert (frame[columns].values == alldata[columns].values).all()


def test_aggregate_sector_manyregions():
    """Test for aggregate_sector function with two regions check colum titles, number of
    agents/region/technologies and assets capacities."""
    from pandas import DataFrame, concat

    mca = examples.model("multiple-agents")
    residential = next(
        (sector for sector in mca.sectors if sector.name == "residential")
    )
    agents = list(residential.agents)
    agents[0].assets["region"] = "BELARUS"
    agents[1].assets["region"] = "BELARUS"
    agents[0].region = "BELARUS"
    agents[1].region = "BELARUS"
    year = [2020, 2025, 2030]
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    agent_list = [list(a.agents) for a in sector_list]
    alldata = sectors_capacity(mca.sectors)
    alldatadict = alldata.to_dict("split")
    columns = ["region", "agent", "type", "sector", "capacity"]
    assert (sorted(columns)) == sorted(alldatadict["columns"])
    frame = DataFrame()
    for a, ai in enumerate(agent_list):
        for ii in range(0, len(ai)):
            for y in year:
                if y in ai[ii].assets.year:
                    if ai[ii].assets.capacity.sel(year=y).values > 0.0:
                        data = DataFrame(
                            {
                                "region": ai[ii].region,
                                "agent": ai[ii].name,
                                "type": ai[ii].category,
                                "sector": sector_list[a].name,
                                "capacity": ai[ii]
                                .assets.capacity.sel(year=y)
                                .values[0],
                            },
                            index=[(y, ai[ii].assets.technology.values[0])],
                        )
                        frame = concat([frame, data])

    assert (frame[columns].values == alldata[columns].values).all()
