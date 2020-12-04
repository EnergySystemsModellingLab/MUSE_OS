.. _regional_data:

=============
Regional data
=============

MUSE requires the definition of the methodology used for investment and dispatch and alias
demand matching. The methodology has to be defined by region and subregion, meant as a
geographical subdivision in a region. Currently, the methodology definition is
important for the legacy sectors only, and can therefore be ignored for the open source version of MUSE.


Below the generic structure of the input commodity file for the electric
heater is shown:

.. csv-table:: Methodology used in investment and demand matching
   :header: SectorName, RegionName, Subregion, sMethodologyPlanning, sMethodologyDispatch
       
   Agriculture, region1, region1, NPV, DCF
   Bioenergy, region1, region1, NPV, DCF
   Industry, region1, region1, NPV, DCF
   Residential, region1, region1, EAC, EAC
   Commercial, region1, region1, EAC, EAC
   Transport, region1, region1, LCOE, LCOE
   Power, region1, region1, LCOE, LCOE
   Refinery, region1, region1, LCOE, LCOE
   Supply, region1, region1, LCOE, LCOE


SectorName
   represents the sector_ID and needs to be consistent across the data input files

RegionName
   represents the region ID and needs to be consistent across all the data inputs

Subregion
   represents the subregion ID and needs to be consistent across all the data inputs

sMethodologyPlanning
   reports the cost quantity used for making investments in new technologies in each
   sector (e.g. NPV stands for net present value, EAC stands for equivalent annual
   costs, LCOE stands for levelised cost of energy)

sMethodologyDispatch
   reports the cost quantity used for the demand matching using existing technologies in
   each sector (e.g. DCF stands for discounted cash flow, EAC stands for equivalent
   annual cost, LCOE stands for levelised cost of energy)