import numpy as np
import pandas as pd
from xarray import DataArray, concat
from muse.utilities import reduce_assets
from muse import examples
from muse.outputs.sector import aggregate_sector, aggregate_sectors
from typing import List


def test_aggregate_sector():
    """Test for aggregate_sector function
    check colum titles, number of agents/region/technologies
    and assets capacities."""
    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    capa = aggregate_sector (sector_list[0], 2020)
    assert "region" in capa.coords
    assert "agent" in capa.coords
    assert "sector" in capa.coords
    assert "year" in capa.coords
    assert "technology" in capa.coords
    agent_names = [[a.name for a in sector.agents] for sector in sector_list
    if "residential" in sector.name]
    region_names = [[a.region for a in sector.agents] for sector in sector_list
    if "residential" in sector.name]
    technology_names = [[a.assets.technology.values[0] for a in sector.agents] for sector in sector_list
    if "residential" in sector.name]

    expected_agent_names = list( [y for x in agent_names for y in x] )
    expected_region_names = list( set([y for x in region_names for y in x] ) )
    expected_technology_names = list(set( [y for x in technology_names for y in x]  ) )
    obtained_technology_names = list( set( list(capa.technology.values)) ) 
    expected_capacity= np.zeros(len(expected_agent_names))
    for ui,u in enumerate(sector_list[0].agents):
        expected_capacity[ui] = u.assets.capacity.sel(year=2020).sum(dim="asset")
    assert sorted(list(capa.agent.values)) == sorted(expected_agent_names)
    assert sorted([str(capa.region.values)]) == sorted(expected_region_names)
    assert sorted(obtained_technology_names) == sorted(expected_technology_names)
    assert expected_capacity.all() == capa.values.all()


def test_aggregate_sectors() -> DataArray:
    """Test for aggregate_sectors function"""
    mca = examples.model("multiple-agents")
    sector_list = [sector for sector in mca.sectors if "preset" not in sector.name]
    alldata = aggregate_sectors(sector_list, 2020)
    agent_names = [[a.name for a in sector.agents] for sector in sector_list]
    region_names = [[a.region for a in sector.agents] for sector in sector_list]
    technology_names = [[a.assets.technology.values[0] for a in sector.agents] for sector in sector_list]

    expected_agent_names = list( [y for x in agent_names for y in x] )
    expected_region_names = list( set([y for x in region_names for y in x] ) )
    expected_technology_names = list(set( [y for x in technology_names for y in x]  ) )
    obtained_technology_names = list( set( list(alldata.technology.values)) ) 
    expected_capacity= np.zeros(len(expected_agent_names))
    for ui,u in enumerate(sector_list[0].agents):
        expected_capacity[ui] = u.assets.capacity.sel(year=2020).sum(dim="asset")
    assert sorted(list(alldata.agent.values)) == sorted(expected_agent_names)
    assert sorted([str(alldata.region.values)]) == sorted(expected_region_names)
    assert sorted(obtained_technology_names) == sorted(expected_technology_names)
    assert expected_capacity.all() == alldata.values.all()

