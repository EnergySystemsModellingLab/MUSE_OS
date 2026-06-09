.. _inputs-agents:

======
Agents
======

Agents are defined using a CSV file, with
one agent per row, using a format meant specifically for retrofit and new-capacity agent pairs.

Each sector should have an agents file, which should follow the structure
reported in the table below, and be referenced from the TOML settings file using the
``agents`` key.

This is an example of what an agents file could look like:

.. csv-table::
   :header: name, type, agent_share, region, objective1, search_rule, decision_method, ...

   A1, New, Agent1New, ASEAN, EAC, all->maturity, epsilonCon, ...
   A2, New, Agent2New, ASEAN, CapitalCosts, all->spend_limit, weightedSum, ...
   A1, Retrofit, Agent1Retro, ASEAN, efficiency, existing, epsilonCon, ...
   A2, Retrofit, Agent2Retro, ASEAN, Emissions, similar, weightedSum, ...

For simplicity, not all columns are included in the example above.

The columns have the following meaning:

.. _name:

``name``
   Name shared by a retrofit and new-capacity agent pair.

``type``
   One of **New** or **Retrofit**. **New** and **Retrofit** agents make up a pair with a given
   :ref:`name <name>`. The demand is split into two, with one part coming from
   decommissioned assets, and the other coming from everything else. "Retrofit" agents
   invest only to make up for decommissioned assets. They are often limited in the
   technologies they can consider (by :ref:`search_rule <search_rule>`). **New** agents
   invest on the rest of the demand, and can often consider more general sets of
   technologies. If only **New** agents are included, they will also invest to make up for
   decommissioned assets, but the end mix might be different than using a specialised
   **Retrofit** agent for that.
   **Note: Retrofit agents will be deprecated in a future release.**

``agent_share``
   Name used to assign a fraction of existing capacity to the agent in the :ref:`inputs-technodata` file.
   If using "New" and "Retrofit" agents, you should create a column with the name of each "Retrofit" agent share (e.g. "Agent1Retro", "Agent2Retro" etc.) in the :ref:`inputs-technodata` file,
   with values summing to 1 for each technology.
   If only using "New" agents, you should create a column with the name of each "New" agent share in the :ref:`inputs-technodata` file,
   with values summing to 1 for each technology.
   See documentation for the :ref:`inputs-technodata` file for more details.

``region``
   Region where an agent operates.

.. py:currentmodule:: muse.objectives

``objective1``
   Objective that an agent will try and maximize or minimize during investment.
   This objective should be one registered with
   :py:func:`@register_objective <register_objective>`. The following objectives are
   available with MUSE:

   - :py:func:`comfort <comfort>`: Comfort provided by a given technology. Comfort does
     not change during the simulation. It is obtained straightforwardly from
     :ref:`inputs-technodata`.

   - :py:func:`efficiency <efficiency>`: Efficiency of the technologies. Efficiency does
     not change during the simulation. It is obtained straightforwardly from
     :ref:`inputs-technodata`.

   - :py:func:`fixed_costs <fixed_costs>`: The fixed maintenance costs incurred by a
     technology. The costs are a function of the capacity required to fulfil the current
     demand.

   - :py:func:`capital_costs <capital_costs>`: The capital cost incurred by a
     technology. The capital cost does not change during the simulation. It is obtained
     as a function of parameters found in :ref:`inputs-technodata`.

   - :py:func:`emission_cost <emission_cost>`: The costs associated for emissions for a
     technology. The costs is a function both of the amount produced (equated to the
     total demand in this case) and of the prices associated with each pollutant.
     Aliased to "emission" for simplicity.

   - :py:func:`fuel_consumption_cost <fuel_consumption_cost>`: Costs of the fuels for
     each technology, where each technology is used to fulfil the whole demand.

   - :py:func:`lifetime_levelized_cost_of_energy <lifetime_levelized_cost_of_energy>`:
     LCOE over the lifetime of a technology. Aliased to "LCOE" for simplicity.

   - :py:func:`net_present_value <net_present_value>`: Present value of all the costs of
     installing and operating a technology, minus its revenues, of the course of its
     lifetime. Aliased to "NPV" for simplicity.

   - :py:func:`equivalent_annual_cost <equivalent_annual_cost>`: Annualized form of the
     net present value. Aliased to "EAC" for simplicity.

   The weight associated with this objective can be changed using ``obj_data1``.
   Whether the objective should be minimized or maximized depends on
   ``obj_sort1``. Multiple objectives are combined using the
   :ref:`decision method <decision_method>`


``obj_data1``
   A value associated with the objective.
   Its meaning, and whether it is used at all, will depend in large part on the :ref:`decision method <decision_method>`.


``obj_sort1``
   Determines whether the objective is maximized or minimized.
   This should be set to "True" for minimization and "False" for maximisation.

Additional objectives
   For certain decision methods you can use more than one objective.
   In this case, additional objectives can be specified with additional columns (e.g. ``objective2``, ``obj_data2``, ``obj_sort2`` etc.)
   For example, when using the weighted sum decision method, the ``obj_dataX`` column for each objective defines the weight of the objective in the weighted sum calculation.


.. py:currentmodule:: muse.filters

.. _search_rule:

``search_rule``
   The search rule allows users to par down the search space of technologies to those an
   agent is likely to consider.
   The search rule is any function with a given signature, and registered with MUSE via
   :py:func:`@register_filter <register_filter>`. The following search rules, defined
   in :py:mod:`~muse.filters`, are available with MUSE:

   - :py:func:`same_enduse <same_enduse>`: Only allow technologies that provide the same
     enduse as the current set of technologies owned by the agent.

   - :py:func:`identity <identity>`: Allows all current technologies. E.g. disables
     filtering. Aliased to "all".

   - :py:func:`similar_technology <similar_technology>`: Only allows technologies that
     have the same type as current crop of technologies in the agent, as determined by
     "Type" in :ref:`inputs-technodata`. Aliased to "similar".

   - :py:func:`same_fuels <same_fuels>`: Only allows technologies that consume the same
     fuels as the current crop of technologies in the agent. Aliased to
     "fueltype".

   - :py:func:`currently_existing_tech <currently_existing_tech>`: Only allows
     technologies that the agent already owns. Aliased to "existing".

   - :py:func:`currently_referenced_tech <currently_referenced_tech>`: Only allows
     technologies that are currently present in the market with non-zero capacity.

   - :py:func:`maturity <maturity>`: Only allows technologies that have achieved a given
     market share.

   - :py:func:`spend_limit <spend_limit>`: Only allows technologies with a unit capital cost (cap_par in :ref:`inputs-technodata`) lower than the spend limit.

   Filters can be combined by chaining them with "->". For example, "all->maturity->spend_limit".

.. py:currentmodule:: muse.decisions

.. _decision_method:

``decision_method``
   Decision methods reduce multiple objectives into a single scalar objective per
   replacement technology. They allow combining several objectives into a single metric
   through which replacement technologies can be ranked.

   Decision methods are any function which follow a given signature and are registered
   via the decorator :py:func:`@register_decision <register_decision>`. The following
   decision methods are available with MUSE, as implemented in
   :py:mod:`~muse.decisions`:

   - :py:func:`mean <mean>`: Computes the arithmetic mean across multiple objectives,
     treating all objectives as equally important.
   - :py:func:`weighted_sum <weighted_sum>`: Computes a weighted average across
     objectives using weights defined in ``obj_data``, where higher weights increase the
     importance of an objective in the final score.
   - :py:func:`lexical_comparion <lexical_comparison>`: Ranks technologies using a
     priority ordering of objectives. The first objective is considered within a
     tolerance defined by ``obj_data1`` (e.g. 0.1 corresponds to a 10% tolerance), and
     only distinguishes technologies outside this tolerance. Subsequent objectives
     (``obj_data2``, ``obj_data3``) are used to break ties. Aliased to "lexo".
   - :py:func:`retro_lexical_comparion <retro_lexical_comparison>`: A variant of lexical
     comparison where tolerances are adjusted so that existing technologies remain
     feasible for comparison. Otherwise, behaviour matches ``lexical_comparison``.
     Aliased to "retro_lexo".
   - :py:func:`epsilon_constraints <epsilon_constraints>`: Selects technologies by
     optimising the first objective while using all remaining objectives as feasibility
     filters. Each filter applies a threshold defined in ``obj_data2`` and
     ``obj_data3``, with direction (above or below the threshold) determined by
     ``obj_sort2`` and ``obj_sort3``. ``obj_data1`` is not used, but must be provided.
     Aliased to "epsilon" and "epsilon_con".
   - :py:func:`retro_epsilon_constraints <retro_epsilon_constraints>`: A variant of
     epsilon constraints where thresholds are adjusted so that existing technologies are
     never excluded by feasibility conditions. Otherwise behaves identically to
     epsilon_constraints. Aliased to "retro_epsilon".
   - :py:func:`single_objective <single_objective>`: Ranks technologies using a single
     objective without considering additional objectives or constraints.

   The functions allow for any number of objectives. However, the format described here
   allows only for three.

``quantity``
   Represents the fraction of new demand that is assigned to the agent
   (e.g. if 0.2, 20% of new demand in each year will be assigned to the agent).
   Must sum to 1 across all "New" agents.
   When using both "Retrofit" agents and "New" agents, this only applies to the "New" agents.

Additional optional columns
  Certain columns may also be required when using certain search rules. These are:

  ``maturity_threshold``
     Only applies when using the :py:func:`maturity <muse.filters.maturity>` search rule.
     Allows agents to only consider technologies that have achieved a certain market share
     (e.g. if 0.5, the agent will only invest in technologies that have a current market share of 50% or more).

  ``spend_limit``
     Only applies when using the :py:func:`spend_limit <muse.filters.spend_limit>` search rule.
     Allows agents to only consider technologies with a unit capital cost (`cap_par`) lower than the spend limit.
     (e.g. if 10, the agent will only invest in technologies with a `cap_par` of 10 or lower, as listed in the :ref:`inputs-technodata` file).
