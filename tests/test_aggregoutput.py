import numpy as np
from xarray import DataArray
from muse import examples
from muse.outputs.sector import aggregate_sector, aggregate_sectors


def test_aggregate_sector():
    """Test for aggregate_sector function
    check colum titles, number of agents/region/technologies
    and assets capacities."""
    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "residential" == sector.name]
    capa = aggregate_sector(sector_list[0], 2020)
    assert "region" in capa.coords
    assert "agent" in capa.coords
    assert "sector" in capa.coords
    assert "year" in capa.coords
    assert "technology" in capa.coords
    agent_names = [a.name for sector in sector_list for a in sector.agents]
    region_names = [a.region for sector in sector_list for a in sector.agents]
    technology_names = [
        a.assets.technology.values[0] for sector in sector_list for a in sector.agents
    ]

    expected_capacity = [
        u.assets.capacity.sel(year=2020).sum(dim="asset").values
        for sector in sector_list
        for u in sector.agents
    ]
    assert sorted(capa.agent.values) == sorted(agent_names)
    assert sorted([str(region) for region in capa.region.values[None]]) == sorted(
        np.unique(region_names)
    )
    assert sorted(capa.technology.values) == sorted(technology_names)
    assert (expected_capacity == capa.values).all()


def test_aggregate_sectors() -> DataArray:
    """Test for aggregate_sectors function"""
    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = aggregate_sectors(sector_list, 2020)
    agent_names = [a.name for sector in sector_list for a in sector.agents]
    region_names = [a.region for sector in sector_list for a in sector.agents]
    technology_names = [
        a.assets.technology.values[0] for sector in sector_list for a in sector.agents
    ]

    expected_capacity = [
        u.assets.capacity.sel(year=2020).sum(dim="asset").values
        for sector in sector_list
        for u in sector.agents
    ]
    assert sorted(alldata.agent.values) == sorted(agent_names)
    assert sorted([str(region) for region in alldata.region.values[None]]) == sorted(
        np.unique(region_names)
    )
    assert sorted(alldata.technology.values) == sorted(technology_names)
    assert (expected_capacity == alldata.values).all()
