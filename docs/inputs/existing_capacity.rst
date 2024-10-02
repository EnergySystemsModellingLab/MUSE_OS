.. _inputs-existing-capacity:

==========================
Existing Sectoral Capacity
==========================

For each technology, the decommissioning profile should be given to MUSE.

The csv file which provides the installed capacity in base year and the decommissioning
profile in the future periods for each technology in a sector, in each region, should
follow the structure reported in the table.


.. csv-table:: Existing capacity of technologies: the residential boiler example
   :header: ProcessName, RegionName, Unit, 2010, 2020, 2030, 2040, 2050

   resBoilerElectric, region1, PJ/y, 5, 0.5, 0, 0, 0
   resBoilerElectric, region2, PJ/y, 39, 3.5, 1, 0.3, 0


ProcessName
   represents the technology ID and needs to be consistent across all the data inputs.

RegionName
   represents the region ID and needs to be consistent across all the data inputs.

Unit (optional)
   reports the unit of the technology capacity; it is for the user internal reference only.

2010,..., 2050
   represent the simulated periods.
