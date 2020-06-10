.. _model-agents:

======
Agents
======

.. currentmodule:: muse

The main role of the agent is to invest in order to fulfill a given future demand. This
demand is received from the sector. The investment procedure is meant to reflect the
preferences of agents characteristic of groups of investors in a region. To this end,
investment is articulated in the following way:

.. contents:: Pseudo-code
   :local:

------
Inputs
------

* technologies: a dataset describing the characteristics of the technologies in the
  sector
* market: MCA market, converted to the sector's timeslices

   * Demand, :math:`\mathcal{D}_{c, s, r}`, commodities consumed by sectors
   * Production (Consumption), :math:`\mathcal{C}_{c, s, r}`, commodities produced by
     sectors
   * Prices, :math:`\mathcal{I}_{c, s, r}`, current market prices

* demand: share of the market demand this agent will satisfy


-----------------------
1. Initial housekeeping
-----------------------

This step gives the code the opportunity to do some housekeeping, e.g. to limit the
growth in memory requirements. Currently, *retrofit* agents remove technologies which
no longer have capacity now or in the future. *New* agents do nothing.

.. warning::

   The :ref:`csv file defining agents <inputs-agents>` does not allow for modifying
   the initial housekeeping operation.

.. seealso::

   :py:func:`hooks.register_initial_asset_transform`, :py:func:`hooks.clean`.

----------------------
2. Filter search space
----------------------

In this step, the space of available replacement technologies is filtered down to those an
agent will consider. The search space is defined in terms of a matrix of boolean values:

The result ``search_space`` is a matrix defining the relationship between an existing
installed technology and the techologies an agent will consider to replace it:

.. math::

      \mathcal{S}_{t, u} \in {0, 1}

With :math:`t` and index over the technologies to be replaced, and :math:`u` an index
over replacement technologies.


In practice, the technologies to be replaced :math:`t` are determined by
``initial_filter``. ``initial_filter`` can be any function registered with
:py:func:`filters.register_initializer`. In general, ``initial_filter`` is
:py:func:`filters.initialize_from_technologies`, and hence :math:`t` is the set of
technologies included in ``demand_share``. In general, :math:`u` will start off as all
technologies in the sector.

.. code-block:: Python

   search_space = initial_filter(
      agent, demand_share, technologies=technologies, market=market
   )
   for afilter in filters:
      search_space = afilter(
         agent, search_space, technologies=technologies, market=market
      )
   return search_space

The initial filter takes as input the share of the demand which the agent will fulfill,
as well as the full market and the dataset characterising each technology. The demand
share acts as an initial filter in the sense that it helps determine for which assets
there is a demand. Subsequent filters take the search space as it exists, as well as the
full market and the data characterising the technologies in the sector.

.. note::

   In all cases, if ``initial_filter`` is not provided, it defaults to
   :py:func:`filters.initialize_from_technologies`.

   The *retrofit* agents always finish with a call to
   :py:func:`filters.with_asset_technology` followed by a call to
   :py:func:`filters.compress`. *New* agents finish always finish with a call to
   :py:func:`filters.compress`.

.. warning::

   The :ref:`csv file defining agents <inputs-agents>` only allows for specifying
   three filters. These filters need not include the filters added in by default.

.. seealso::

   :py:mod:`~muse.filters`, :py:func:`filters.factory`,
   :py:func:`filters.register_filter`, :py:func:`filters.register_initializer`

---------------------------------------------
3. Compute replacement-technology cost-matrix
---------------------------------------------

The cost matrix is defined as data array :math:`\mathcal{C}_{t, u}^o`, where :math:`t`
is an index running over the technologies to be replaced, :math:`u` runs over the
potential replacement technologies, and :math:`o` is an optional compound index which
could run over timeslices, regions, or any other suitable dimension.

It is computed in three steps:

#. Any number of objectives are computed for the enabled replacement technologies the
   search space
#. The objectives are combined via a decision method
#. The resulting scaling objectives are `ranked`__ along the
   :math:`u` dimension

__ http://xarray.pydata.org/en/stable/generated/xarray.DataArray.rank.html>

Objectives are any function registered with
:py:func:`@register_objective<objectives.register_objective>`. They take as input the
agent (which should not be modified during the call), the search space, the market and
the dataset of technology characteristics. They must return an array
:math:`\mathcal{O}_{t, u}^g` where :math:`g` is an optional composite index.

The decision method is any function registered with
:py:func:`@register_decision<decisions.register_decision>`. It takes as input an agent, a
dataset collecting the datarrays computed from the objective functions, and some
parameters, such as whether weights or whether an objective should be minimized or
maximized. The function should return a cost matrix, as defined above.

.. note::

   It is the users responsibility that any extra dimension (e.g. :math:`o` and
   :math:`g`) are consistent and meaningful throughout the calculation, from objectives
   to decision to investment below.

.. warning::

   The objective and decision method can be defined to accept any number of keyword
   arguments parameterizing their behaviour. However, the :ref:`standard csv input
   file<inputs-agents>` does not currently allow to specify them.

.. seealso::

   :py:mod:`~muse.objectives`, :py:func:`objectives.register_objective`,
   :py:mod:`~muse.decisions`, :py:func:`decisions.register_decision`
   

---------------------------------
4. Compute investment constraints
---------------------------------

Currently, there is only one kind of constraint, implementing maximum capacity
expansion. It is a constraint imposed separately on each agent limiting how much they
can invest into each technology. Let :math:`\Delta\mathcal{A}^{i,r}_t` bet the future
investments. Then the constraints are:

.. math::

   \Gamma_t^{i, r} \geq \Delta\mathcal{A}^{i,r}_t

   \Delta\mathcal{A}^{i,r}_t \geq 0

:math:`\Gamma_t^{i, r}` is an upper bound on investments for a given agent, replacement
technology :math:`r` and current technology :math:`t`. It is computed as a function of
the current asset profile in :py:meth:`~muse.constraints.max_capacity_expansion`. The
second constraints imposes that investments cannot be negative (e.g. they can only add
to the agent's assets, and never remove).

----------------------------
5. Determine new investments
----------------------------

The objective of this step is to satisfy the market demand for products
Given the maximum production for each replacement technology :math:`P_u`, the demand for
a product :math:`\mathcal{D}_{c, s}^r`, the search space determined above,
:math:`\mathcal{S}_{t, u}`, and the maximum capacity constraints :math:`\Gamma_t^{i,
r}`, we want to match new investments to the demand.

.. seealso::

   :py:func:`muse.demand_matching.demand_matching`

-----------------------------------
6. Fold new investments into assets
-----------------------------------

Given a set of assets representing new investments for the agents, it still needs to be
decided how they should be merged into the existing assets. Currently, *retrofit* agents
will merge old and new assets, whereas *new* agents will simply replace the old with the
new.

.. warning::

   The :ref:`csv file defining agents <inputs-agents>` does not allow for modifying
   the asset-merge operation.

.. seealso::

   :py:func:`hooks.register_final_asset_transform`, :py:func:`hooks.merge_assets`,
   :py:func:`hooks.new_assets_only`.

-------
Outputs
-------

No direct output. However, :py:attr:`agent.AgentBase.assets` is likely to have been
modified, reflecting new investments.
