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
   defines the type of a commodity (i.e. Energy, Service, Material, or Environmental).

   The "energy" type includes the energy commodities, such as biomass, electricity, gasoline, and hydrogen,
   which are either extracted, transformed from one to another, or used in the energy system.

   The "service" type includes commodities such as space heating or hot water which correspond to selected
   people's needs, and whose fulfillment requires energy uses.

   The "material" type represent non-energy inputs for energy technologies, such as limestone or oxygen.
   The "environmental" type refers to non-energy commodities whichare used to quantify an impact on the environment,
   such as greenhouse gases or CO2. They can be subjected to different types of environmental fees or taxes.

CommodityName
   is the internal name used for a commodity inside the model.

CommodityEmissionFactor_CO2
   is CO2 emission per unit of commodity flow.
   This commodity property is not directly used in the MUSE core set of equations, but can be further referred to
   for any subsequent development of the code.

HeatRate
   represents the lower heating value of an energy commodity
   This commodity property is not directly used in the MUSE core set of equations, but can be further referred to
   for any subsequent development of the code.

Unit
   is the unit used as a basis for all the input data. More specifically the model allows
   a totally flexible way of defining the commodities. CommodityName is currently the
   only column used internally as it defines the names of commodities and needs to be
   used consistently across all the input data files. The remaining columns of the file
   are only relevant for the user internal reference for the original sets of
   assumptions used.
