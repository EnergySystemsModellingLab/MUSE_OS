.. _inputs-projection:

=========================
Initial Market Projection
=========================

MUSE needs an initial projection of the market prices for each period of the simulation.

* The price trajectory is needed if the MCA works in *equilibrium* mode as an initial
  trajectory for the base year of the simulation. The market will override the
  calculated prices obtained from each commodity equilibrium for all the future periods
  following the base year (setting market keyword *equilibrium = true*)
* Similarly, if the market works in a *carbon budget* mode, the prices are used as a
  starting point. The only difference from the previous case is given by the fact that
  the MCA will be calculating an additional global market price for carbon dioxide (and
  additional pollutants if required)
* If the MCA works in an *exogenous* mode (setting market keyword *equilibrium = false*
  or *maximum_iterations = 1*), it will use the initial market projection as the projection
  for the the base year and
  all the future periods of the simulation

By default, MUSE works in *equilibrium* mode with a number of iterations greater than 1.
These are the typical conditions for a complete simulations with multiple sectors, belonging
to the supply and the demand. In this way the iterative methods for the market converges
can have a meaningful application. While global settings files, such as the initial price file
as well as the commodidity definition files need to refer to all the commodities present in all
the sectors, each sector allows a definition of the sector-specific attribute referring just to
commodities either consumed or produced in the same sector.

As MUSE allows 1 sector model simulations, the market convergence settings could be conveniently
modified into *equilibrium = false* and setting *maximum_iterations = 1*).

The forward price trajectory should follow the structure reported in the table below.


.. csv-table:: Initial market projections
   :header: RegionName, Attribute, Time, com1, com2, com3


   Unit, -, Year, MUS$2010/PJ, MUS$2010/PJ, MUS$2010/PJ
   region1, CommodityPrice, 2010, 20, 1.9583, 2
   region1, CommodityPrice, 2015, 20, 1.9583, 2
   region1, CommodityPrice, 2020, 20.38518042, 1.996014941, 2.038518042
   region1, CommodityPrice, 2025, 20.77777903, 2.034456234, 2.077777903
   region1, CommodityPrice, 2030, 21.17793872, 2.073637869, 2.117793872
   region1, CommodityPrice, 2035, 21.58580508, 2.113574105, 2.158580508
   region1, CommodityPrice, 2040, 22.00152655, 2.154279472, 2.200152655
   region1, CommodityPrice, 2045, 22.42525441, 2.195768786, 2.242525441
   region1, CommodityPrice, 2050, 22.85714286, 2.238057143, 2.285714286


RegionName
   represents the region ID and needs to be consistent across all the data inputs

Attribute
   defines the attribute type. In this case it refers to the CommodityPrice; it is
   relevant only for internal use

Time
   corresponds to the time periods of the simulation; the simulated time framework in
   the example goes from 2010 through to 2050 with a 5-year time step

com1, ..., comN
   Any further columns represent the commodities modelled, as defined in the global
   commodities the row Unit reports the unit in which the technology consumption is
   defined; it is for the user internal reference only. The names *comX* should be
   replaced with the names of the commodities.
