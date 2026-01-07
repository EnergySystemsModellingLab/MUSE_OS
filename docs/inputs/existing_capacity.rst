.. _inputs-existing-capacity:

==========================
Existing Capacity
==========================

Each sector must have an existing capacity file, which defines the total capacity of all
pre-defined assets in that sector. This data must be given in the form of a decommissioning
profile, which shows how much capacity of each technology exists in each region at the
start of the simulation, and how it is expected to decline over time as these assets are
decommissioned. Any assets installed *by MUSE* during the simulation will be added on
top of this existing capacity.

This file should follow the structure shown in the example table below, and be referenced from
the TOML settings file using the ``existing_capacity`` key. For example, in this case,
the file shows that in region1 there is 5 MW of existing residential electric boiler
capacity in 2010, of which 0.5 MW will remain active in 2020, and none will remain by 2030.

.. csv-table:: Existing capacity of technologies: the residential boiler example
   :header: technology, region, 2010, 2020, 2030, 2040, 2050

   resBoilerElectric, region1, 5, 0.5, 0, 0, 0
   resBoilerElectric, region2, 39, 3.5, 1, 0.3, 0

``technology``
   represents the technology ID, which must match a technology defined in the
   sector's technodata file.

``region``
   represents the region ID, which must match a region defined in the settings TOML.

Years (one column per year)
   represent the years in the simulation. The values in these columns represent
   the total installed capacity of the technology in the given year/region.
