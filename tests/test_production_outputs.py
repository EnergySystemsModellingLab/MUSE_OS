"""Test saving multiple outputs to file."""


import numpy as np
import pandas
from muse import examples
from muse.outputs.mca import sector_alcoe


def test_aggregate_alcoe_sector():
    """Test for aggregate_sector function check colum titles, number of
    agents/region/technologies and alcoe."""
    from pandas import DataFrame
    from muse.quantities import annual_levelized_cost_of_energy

    mca = examples.model("multiple-agents")
    year = [2020, 2025]
    residential = next(
        (sector for sector in mca.sectors if sector.name == "residential")
    )

    agent_list = residential.agents
    alldata = sector_alcoe(mca.market, residential)
    lcoe = annual_levelized_cost_of_energy(mca.market.prices, residential.technologies)

    for r, i in alldata.iterrows():
        for a in agent_list:
            if a.assets.technology == r[1]:
                assert np.unique(alldata.loc[[r], ["alcoe"]]) == lcoe.sel(
                    year=r[2], technology=r[1], timeslice=r[0]
                )
