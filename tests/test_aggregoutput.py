import pandas as pd
from xarray import DataArray
from muse.utilities import reduce_assets
from muse.sectors import AbstractSector
from muse import examples
from muse.outputs import save_output, to_csv


def mca(loaded_residential_settings):
    """Initialized MCA with the default settings and the residential sector."""
    from muse.mca import MCA

    result = MCA.factory(loaded_residential_settings)

    return result


def aggregate_sector(sector: AbstractSector, year, columns=None) -> pd.DataFrame:
    """Sector output to desired dimensions using reduce_assets"""

    capa =[u.assets.capacity.sel(year=year)
                            for u in sector.agents]

    capa_reduced = reduce_assets(capa, coords=(["region","year","technology"]))
    capa_reduced=capa_reduced.to_dataframe()
    return capa_reduced


# def aggregate_sectors(sectors: List[AbstractSectors], **kwargs)
#       -> pd.Dataframe:
#     """Aggregate outputs from all sectors"""
#     alldata = [aggregate_sector(sector, **kwargs) for sector in sectors]
#     return Dataframe.concatenate(alldata)


path = "C:/Users/sg2410/Desktop/GitHub/StarMuse/src/muse/data/model/settings.toml"

market = mca(path)

#market.run()
#for t in market.market.year:
for sector in market.sectors:
    if "agents" in sector.__dict__:
        capa = aggregate_sector(sector,market.market.year[0])
#        aggregate_sectors
#        result = save_output(config, None, None, None)
        print (capa)



# def aggregate_sectors_tocsv(Dataframe, **kwargs) -> pd.Dataframe:
#     """Returns output to csv"""

#     return Dataframe.to_csv
