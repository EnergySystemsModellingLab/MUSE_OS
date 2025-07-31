.. _inputs-existing-capacity:

==========================
Existing Capacity
==========================

For each technology, the decommissioning profile should be given to MUSE.

The csv file which provides the installed capacity in base year and the decommissioning
profile in the future periods for each technology in a sector, in each region, should
follow the structure reported in the table.


.. csv-table:: Existing capacity of technologies: the residential boiler example
   :header: technology, region, 2010, 2020, 2030, 2040, 2050

   resBoilerElectric, region1, 5, 0.5, 0, 0, 0
   resBoilerElectric, region2, 39, 3.5, 1, 0.3, 0


``technology``
   represents the technology ID and needs to be consistent across all the data inputs.

``region``
   represents the region ID and needs to be consistent across all the data inputs.

Years (one column per year)
   represent the simulated periods.
