.. _inputs-iocomms:

=================
Commodities
=================

Input

Input commodities are the commodities consumed by each
technology.  They are defined in a csv file which describes the commodity inputs to each
technology, calculated per unit of technology activity, Where the unit is defined by the user (e.g. petajoules).

Output


Output commodities are the commodities produced by each
technology.  They are defined in a csv file which describes the commodity outputs from
each technology, defined per unit of technology activity. Emissions, such as CO2
(produced from fuel combustion and reactions), CH4, N2O, F-gases, can also be accounted
for in this file.


General features


To illustrate the data required for a generic technology in MUSE, the *electric boiler
technology* is used as an example. The commodity flow for the electric boiler, capable
to cover space heating and water heating energy service demands.

.. figure:: commodities_io.png
   :width: 400
   :alt: Electric boilers schematic

   The table below shows the basic data requirements for a typical technology, the
   electric boiler.

.. image:: commodities_io_table.png
   :width: 400
   :alt: Electric boilers input output commodities


Below it is shown the generic structure of the input commodity file for the electric
heater.

.. csv-table:: Commodities used as consumables - Input commodities
   :header: ProcessName, RegionName, Time, Level, electricity

   Unit, -, Year, -, GWh/PJ
   resBoilerElectric, region1, 2010, fixed, 300
   resBoilerElectric, region1, 2030, fixed, 290


ProcessName
   represents the technology ID and needs to be consistent across all the data inputs.

RegionName
   represents the region ID and needs to be consistent across all the data inputs.

Time
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation.

Level
   characterises either a fixed or a flexible input type the following columns should
   contain the list of commodities the row.

Unit
   reports the unit in which the technology consumption is defined; it is for the user
   internal reference only.

The input data has to be provided for the base year. Additional years within the time
framework of the overall simulation can be defined. In this case, MUSE would interpolate
the values between the provided periods and assume a constant value afterwards. The additional
years at which input data for input/output commodities, are defined needs to equal for :ref:`inputs-technodata` and :ref:`inputs-technodata-ts`.

Interpolation is activated only if the feature *interpolation_mode = 'Active'* is defined in the TOML file.
