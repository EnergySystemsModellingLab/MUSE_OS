=======
Sectors
=======

.. currentmodule:: muse

The main role of sectors is to match the future demand coming from the |mca| with each
agent, so that each agent can decide how to operate the assets available and invest in
new ones if needed. A secondary role is match the agents' assets to the forward demand
trajectory, in order to compute the consumption, production, and the cost of supply a
commodity. Consumption and production of energy commodities are passed back to the
|mca|; the cost of supply of a commodity is calculated and passed only by conversion and
supply sectors and it is used by the |mca| to update the commodity prices when it
operates in the "equilibrium" or "budget" mode.

The following pseudo-code shows how the sector operates:


.. contents:: Pseudo-code
   :local:

------
Inputs
------

The input market from the MCA. In general, it is the result of running sectors with
higher priorities. It contains three components:

* Demand (``consumption``), :math:`\mathcal{D}_{c, s, r}`, commodities consumed by
  sectors
* Production (``supply``), :math:`\mathcal{C}_{c, s, r}`, commodities produced by
  sectors
* Prices, :math:`\mathcal{I}_{c, s, r}`, current market prices

------------------------------------------
1. Convert MCA market to sector timeslices
------------------------------------------

- the conversion is called in in :py:meth:`sectors.Sector.next`

   .. literalinclude:: ../../src/muse/sectors/sector.py
      :lineno-match:
      :dedent: 8
      :start-after: # > to sector timeslice
      :end-before: # > agent interactions

- the conversion proper occurs in :py:meth:`sectors.Sector.convert_market_timeslice`, a
  helper function to distinguish between intensive and extensive quantities
- also see :py:mod:`muse.timeslices`


.. _model, agent interaction:

---------------------------
2. Agent-agent interactions
---------------------------

Performs agent-agent interactions. Canonically, MUSE agents interact in pairs where one
agent handles investment into _new_ demand, whereas the other handles retrofitting. In
that instance, agents in pair interact by exchanging assets (from *new* to *retrofit*
agents) and information about market makeup (from *retrofit* to *new* agents).


- the interactions are called in :py:meth:`sectors.Sector.next`

   .. literalinclude:: /../src/muse/sectors/sector.py
      :lineno-match:
      :start-after: # > agent interactions
      :end-before: # > investment
      :dedent: 8

- the inputs consists only of the list of all agents in a sector
- the agents are likely to be modified in-place by the call
- the interaction function above is created from the TOML inputs via
  :py:func:`interactions.factory`, called in :py:meth:`sectors.Sector.factory`
- interactions are defined in :py:mod:`~muse.interactions`
- :py:func:`~interactions.register_interaction_net` registers new ways to figure out
  Groups of interacting agents
- :py:func:`~interactions.register_agent_interaction` registers new interactions proper




.. _model, agent investment:

-------------------
3. Agent investment
-------------------

Agents are allowed to invest in order to service the demand for the forecast year.
In practice, there are two parts to the investment process:


#. The demand is split according to a user-defined function into seperate blocks to be
   serviced by each agent
#. A loop over the agents calls for each agent to invest the demand determined above

The method in the first step above is any method registered with
:py:func:`@register_demand_share <muse.demand_share.register_demand_share>`. In
practice, their is currently only one such method,
:py:func:`~muse.demand_share.new_and_retro`. In this particular setup, agents
come in pairs. Broadly, a pair services a share of the demand which corresponds to the
assets the pair owns. However, the *new* agent services only the deman corresponding to
an increase in consumption, whereas the *retrofit* agent services the demand which
corresponds to its newly decommisioned assets. A more detailled description can be found
:py:func:`here <muse.demand_share.new_and_retro>`.


The computation is called in in :py:meth:`sectors.Sector.next`, via
:py:meth:`sectors.Sector.investment`

.. literalinclude:: ../../src/muse/sectors/sector.py
    :lineno-match:
    :dedent: 8
    :start-after: # > investment
    :end-before: # > output to mca

.. seealso::
    :py:mod:`muse.demand_share`,
    :py:func:`demand_share.register_demand_share`,
    :py:func:`demand_share.new_and_retro`,

------------------------------------------------------
4. Compute market: production, consumption, and prices
------------------------------------------------------

- the computation is called in in :py:meth:`sectors.Sector.next`, via
  :py:meth:`~sectors.Sector.market_variables`

   .. literalinclude:: ../../src/muse/sectors/sector.py
      :lineno-match:
      :dedent: 8
      :start-after: # > output to mca
      :end-before: # < output to mca

- uses sector's timeslices
- pseudo-code:

   #. Compute aggregate asset capacity across agents. See
      :py:func:`~utilities.reduce_assets`.
   #. Compute supply from capacity and input MCA demand, using function registered with
      :py:func:`muse.production.register_production`.
   #. Compute consumption from supply and input market prices
   #. Compute prices from supply and
      :py:func:`LCOE<quantities.annual_levelized_cost_of_energy>`.

------------------------------------------
5. Convert output market to MCA timeslices
------------------------------------------

- the conversion is called in in :py:meth:`sectors.Sector.next`

   .. literalinclude:: ../../src/muse/sectors/sector.py
      :lineno-match:
      :dedent: 8
      :start-after: # > to mca timeslices
      :end-before: # < to mca timeslices

- it is mostly reverse operation of
  :ref:`model/sectors:1. Convert mca market to sector timeslices`. 
- the ``comm_usage`` coordinate helps the MCA determine which commodities are produced
  by the sector, and hence which commodity prices should be set from the sector.  See
  :py:mod:`muse.commodities`.

.. |mca| replace:: market-clearing algorithm


-------
Outputs
-------

A market containing the production, consumption and cost of supplying commodities
specific to this market. It is defined as follows:
  
* Production (``supply``), :math:`\mathcal{C}_{c, s, r}`, commodities produced by
  the sector
* Demand (``consumption``), :math:`\mathcal{D}_{c, s, r}`, commodities consumed by
  the sector
* Costs associated with the consumption, :math:`\mathcal{O}_{c, s, r}`
* A coordinate ``comm_usage`` defining each commodity according to its usage (see
  :py:mod:`muse.commodities`). More specifically, it allows the caller (the MCA)
  to know which commodity is produced by the sector (see
  :py:func:`muse.mca.single_year_iteration`).
