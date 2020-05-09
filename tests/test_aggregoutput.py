import pandas as pd
from xarray import DataArray
from muse.utilities import reduce_assets
from muse.sectors import AbstractSector
from muse import examples
from muse.outputs import save_output, to_csv
from typing import List


def mca(loaded_residential_settings):
    """Initialized MCA with the default settings and the residential sector."""
    from muse.mca import MCA

    result = MCA.factory(loaded_residential_settings)

    return result


def aggregate_sector(sector: AbstractSector, year,
                     columns=None) -> pd.DataFrame:
    """Sector output to desired dimensions using reduce_assets"""
    capa_sector = []
    for u in sector.agents:
        capa_agent = u.assets.capacity.sel(year=year)
        capa_agent['agent'] = u.name
        capa_agent['sector'] = sector.name
        capa_sector.append(capa_agent)

    capa_reduced = reduce_assets(capa_sector, coords=(["agent","sector","region","year","technology"]))
    capa_reduced = capa_reduced.to_dataframe()#columns should be read here
    return capa_reduced


def aggregate_sectors(sectors: List[AbstractSector], year) -> pd.DataFrame:
    """Aggregate outputs from all sectors"""
    alldata = [aggregate_sector(sector, year)  for sector in sectors]
    return pd.concat(alldata)


path = """C:/Users/sg2410/Desktop/GitHub/StarMuse/src/muse/data/model/settings.toml"""


market = mca(path)

#
# market.run()
# for t in market.market.year:
sector_list = [sector for sector in market.sectors if "preset" not in sector.name]

capa = aggregate_sectors (sector_list, market.market.year[0])#year to be changed


# def aggregate_sectors_tocsv(Dataframe, **kwargs) -> pd.Dataframe:
#     """Returns output to csv"""

#     return Dataframe.to_csv
