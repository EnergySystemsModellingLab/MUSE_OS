"""Test saving multiple outputs to file."""
from pathlib import Path
from typing import Text

import numpy as np
import pandas
import xarray
from xarray import Dataset, DataArray, broadcast, concat
import os
from pytest import approx, importorskip, mark

from muse.outputs.sector import factory, register_output_quantity

from muse import examples
from muse.outputs.mca import (
    sector_capacity,
    sectors_capacity,
    sector_alcoe,
    sector_llcoe,
)


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


@register_output_quantity
def streetcred(*args, **kwargs):

    return DataArray(
        np.random.randint(0, 5, (3, 2)),
        coords={
            "year": [2010, 2015],
            "technology": ("asset", ["a", "b", "c"]),
            "installed": ("asset", [2010, 2011, 2011]),
        },
        dims=("asset", "year"),
    )
