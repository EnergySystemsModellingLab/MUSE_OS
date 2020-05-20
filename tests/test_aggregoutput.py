import numpy as np

from muse import examples
from muse.outputs.mca import sector_capacity, sectors_capacity


def test_aggregate_sector():
    """Test for aggregate_sector function check colum titles, number of
    agents/region/technologies and assets capacities."""
    from operator import attrgetter

    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "residential" == sector.name]
    capa = sector_capacity(sector_list[0])
    assert "region" in capa.coords
    assert "agent" in capa.coords
    assert "sector" in capa.coords
    assert "year" in capa.coords
    assert "technology" in capa.coords
    agent_names = [a.name for a in sector_list[0].agents]
    region_names = [a.region for a in sector_list[0].agents]
    technology_names = [a.assets.technology.values[0] for a in sector_list[0].agents]

    assert sorted(capa.agent.values) == sorted(agent_names)
    assert sorted(np.unique(capa.region.values)) == sorted(np.unique(region_names))
    assert sorted(np.unique(capa.technology.values)) == sorted(
        np.unique(technology_names)
    )
    expected_capacity = [
        [u.assets.capacity.sel(year=2020).sum(dim="asset").values]
        for u in sorted(sector_list[0].agents, key=attrgetter("name"))
    ]
    assert (expected_capacity == capa.sel(year=2020).values).all()


def test_aggregate_sectors():
    """Test for aggregate_sectors function."""
    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = sectors_capacity(mca.sectors)
    agent_names = [a.name for sector in sector_list for a in sector.agents]
    region_names = [a.region for sector in sector_list for a in sector.agents]
    technology_names = [
        a.assets.technology.values[0] for sector in sector_list for a in sector.agents
    ]

    assert sorted(alldata.agent.values) == sorted(agent_names)
    assert sorted(np.unique(alldata.region.values)) == sorted(np.unique(region_names))
    assert sorted(np.unique(alldata.technology.values)) == sorted(
        np.unique(technology_names)
    )
    expected_capacity = [
        u.assets.capacity.sel(year=2020).sum(dim="asset").values
        for sector in sector_list
        for u in sector.agents
    ]
    assert (expected_capacity == alldata.sel(year=2020).sum("technology").values).all()


def test_aggregate_sector_manyregions():
    """Test for aggregate_sector function with two regions check colum titles, number of
    agents/region/technologies and assets capacities."""
    from operator import attrgetter

    mca = examples.model("multiple-agents")
    residential = next(
        (sector for sector in mca.sectors if sector.name == "residential")
    )
    residential.agents[0].assets["region"] = "BELARUS"
    residential.agents[1].assets["region"] = "BELARUS"
    residential.agents[0].region = "BELARUS"
    residential.agents[1].region = "BELARUS"
    sector_list = [sector for sector in mca.sectors if "residential" == sector.name]
    capa = sector_capacity(sector_list[0])
    assert "region" in capa.coords
    assert "agent" in capa.coords
    assert "sector" in capa.coords
    assert "year" in capa.coords
    assert "technology" in capa.coords
    agent_names = [a.name for a in sector_list[0].agents]
    region_names = [a.region for a in sector_list[0].agents]
    technology_names = [a.assets.technology.values[0] for a in sector_list[0].agents]

    assert sorted(capa.agent.values) == sorted(agent_names)
    assert sorted(np.unique(capa.region.values)) == sorted(np.unique(region_names))
    assert sorted(np.unique(capa.technology.values)) == sorted(
        np.unique(technology_names)
    )

    expected_capacity = [
        [u.assets.capacity.sel(year=2020).sum(dim="asset").values]
        for u in sorted(sector_list[0].agents, key=attrgetter("name"))
    ]
    assert (expected_capacity == capa.sel(year=2020)).all()
