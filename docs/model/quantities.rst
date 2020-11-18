.. _conventions:

==========================
Quantities and Conventions
==========================

.. _indices:

Indices
-------

:math:`t`
   technologies
:math:`c`
   commodities
:math:`s`
   timeslices
:math:`r`
   regions
:math:`y`
   years
:math:`\iota`
   installation period
:math:`a`
   Composite *asset* index, generally made up of technologies :math:`t` and installation
   period :math:`\iota`


.. _quantities:

Quantities
----------

time period, :math:`\Delta y`
   Discretization of time in the simulation, in years.

Agent assets, :math:`A^{i, r}_{t, \iota}(y)`
   :math:`i` is an index or a compound index over agents in the same region.


.. _market:

Market, :math:`\mathcal{M}`
   * Demand, :math:`\mathcal{D}_{c, s, r}`
   * Production (Consumption), :math:`\mathcal{C}_{c, s, r}`
   * Prices, :math:`\mathcal{I}_{c, s, r}`

Technodata
   * Utilization factor by asset, :math:`T^{uf}`
