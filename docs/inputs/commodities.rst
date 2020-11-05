.. _inputs-commodities:

=====================
Commodity Description
=====================

MUSE handles a configurable number and type of commodities which are primarily used to
represent energy, services, pollutants/emissions. The commodities for the simulation as
a whole are defined in a csv file with the following structure.

.. csv-table:: Global commodities
   :header: Commodity, CommodityType, CommodityName, CommodityEmissionFactor_CO2, HeatRate, Unit


   Coal, Energy, hardcoal, 94.6, 29, PJ
   Agricultural-residues, Energy, agrires, 112, 15.4, PJ

Commodity
   represents the extended name of a commodity

CommodityType
   defines the type of a commodity (i.e. energy, material or environmental)

CommodityName
   is the internal name used for a commodity inside the model. 

CommodityEmissionFactor_CO2
   is CO2 emission per unit of commodity flow 

HeatRate
   represents the lower heating value of an energy commodity 

Unit
   is the unit used as a basis for all the input data. More specifically the model allows
   a totally flexible way of defining the commodities. CommodityName is currently the
   only column used internally as it defines the names of commodities and needs to be
   used consistently across all the input data files. The remaining columns of the file
   are only relevant for the user internal reference for the original sets of
   assumptions used.
