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

CommodityName
   the internal name used for a commodity inside the model (e.g. "heat" or "elec").
   Any references to commodities in other files must use these names.

Description
   an extended name/description of a commodity (e.g. "Heating" or "Electricity").

CommodityType
   defines the type of a commodity (i.e. Energy, Service, Material, or Environmental).

   The "energy" type includes energy commodities, such as biomass, electricity, gasoline, and hydrogen,
   which are either extracted, transformed from one to another, or used in the energy system.

   The "service" type includes commodities such as space heating or hot water which correspond to selected
   people's needs, and whose fulfillment requires energy uses.

   The "material" type represent non-energy inputs for energy technologies, such as limestone or oxygen.

   The "environmental" type refers to non-energy commodities whichare used to quantify an impact on the environment,
   such as greenhouse gases or CO2. They can be subjected to different types of environmental fees or taxes.

Unit (optional)
   is the unit used to represent quantities of the commodity (e.g "PJ").
   This parameter does not need to be included, as it isn't used in the model, but care should be taken to ensure that units are consistent across all input files.
