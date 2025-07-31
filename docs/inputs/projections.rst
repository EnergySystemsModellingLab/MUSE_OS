.. _inputs-projection:

=========================
Commodity Price Projections
=========================

This file can be used to supply pre-set prices for commodities.
The interpretation of these prices depends on the type of commodity:

* For **non-enviromental** commodities (**energy**, **service**, **material**), prices represent
  the monetary cost of purchasing the commodity. Whether a price projection is required
  or not depends on the processes in the model and the simulation settings:

  * For commodities that are **produced by processes in the model**, prices do not need to
    be provided as they are calculated endogenously by the model in all years after the
    base year, so long as the model is in equilibrium mode. Equilibrium mode is active
    as long as the ``maximum_iterations`` parameter in the settings file is greater than 1.
  * For commodities that are **not produced by processes in the model**, prices should be
    provided for all years as they cannot be endogenously calculated.

* For **environmental** commodities, prices represent levies on production (e.g. carbon tax).
  In most cases, these will not be calculated endogenously, so users should provide
  full price trajectories. The exception is when using the carbon budget
  mode, where the prices of environmental commodities may be updated throughout the simulation.
  Lack of a price trajectory will be interpreted as a price of 0 for all periods (i.e. no levy on production),
  again with the exception of the carbon budget mode.

The price trajectory should follow the structure shown in the table below.

.. csv-table:: Initial market projections
   :header: region, attribute, year, com1, com2, com3

   region1, CommodityPrice, 2010, 20, 1.9583, 2
   region1, CommodityPrice, 2015, 20, 1.9583, 2
   region1, CommodityPrice, 2020, 20.38518042, 1.996014941, 2.038518042
   region1, CommodityPrice, 2025, 20.77777903, 2.034456234, 2.077777903
   region1, CommodityPrice, 2030, 21.17793872, 2.073637869, 2.117793872
   region1, CommodityPrice, 2035, 21.58580508, 2.113574105, 2.158580508
   region1, CommodityPrice, 2040, 22.00152655, 2.154279472, 2.200152655
   region1, CommodityPrice, 2045, 22.42525441, 2.195768786, 2.242525441
   region1, CommodityPrice, 2050, 22.85714286, 2.238057143, 2.285714286


``region``
   represents the region ID and needs to be consistent across all the data inputs

``attribute``
   defines the attribute type. In this case it refers to the CommodityPrice; it is
   relevant only for internal use

``year``
   corresponds to the time periods of the simulation; the simulated time framework in
   the example goes from 2010 through to 2050 with a 5-year time step

Commodities (one column per commodity)
   Any further columns represent the commodities modelled, with names matching those
   defined in the global commodities file.
   Values in these columns represent the price of the commodity in the given year/region.

   **Note:** All commodity prices should be expressed in the currency specified in the
   settings file. For example, if the currency is set to "USD" and a commodity has units
   "PJ", then the prices for that commodity should be expressed as "USD/PJ".
