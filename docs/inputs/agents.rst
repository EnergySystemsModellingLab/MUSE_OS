.. _inputs-agents:

======
Agents
======

In MUSE, an agent-based formulation was originally introduced for the residential and
commercial building sectors (`Sachs et al. (2019) <https://doi.org/10.1016/j.energy.2019.01.161>`_).  Agents are defined using a CSV file, with
one agent per row, using a format meant specifically for retrofit
and new-capacity agent pairs. This CSV file can be read using
:py:func:`~muse.readers.csv.read_csv_agent_parameters`. The data is also
interpreted to some degree in the factory functions
:py:func:`~muse.agent.create_retrofit_agent` and
:py:func:`~muse.agent.create_newcapa_agent`.

For instance, we have the following CSV table:

.. csv-table::
   :header: Name, Type, AgentShare, RegionName, Objective1, SearchRule, DecisionMethod, ...

   A1, New, Agent1New, ASEAN, EAC, all->maturity, epsilonCon, ...
   A2, New, Agent2New, ASEAN, CapitalCosts, all->spend_limit, weightedSum, ...
   A1, Retrofit, Agent1Retro, ASEAN, efficiency, existing, epsilonCon, ...
   A2, Retrofit, Agent2Retro, ASEAN, Emissions, similar, weightedSum, ...

For simplicity, not all columns are included in the example above.

The columns have the following meaning:

.. _name:

Name
   Name shared by a retrofit and new-capacity agent pair.

Type
   One of "New" or "Retrofit". "New" and "Retrofit" agents make up a pair with a given
   :ref:`name <Name>`. The demand is split into two, with one part coming from
   decommissioned assets, and the other coming from everything else. "Retrofit" agents
   invest only to make up for decommissioned assets. They are often limited in the
   technologies they can consider (by :ref:`SearchRule <SearchRule>`). "New" agents
   invest on the rest of the demand, and can often consider more general sets of
   technologies. If only "New" agents are included, they will also invest to make up for
   decommissioned assets, but the end mix might be different than using a specialised
   "Retrofit" agent for that.
   *Note: Retrofit agents will be deprecated in a future release.*

AgentShare
   Name used to assign a fraction of existing capacity to the agent in the :ref:`inputs-technodata` file.
   If using "New" and "Retrofit" agents, you should create a column with the name of each "Retrofit" agent share (e.g. "Agent1Retro", "Agent2Retro" etc.) in the :ref:`inputs-technodata` file,
   with values summing to 1 for each technology.
   If only using "New" agents, you should create a column with the name of each "New" agent share in the :ref:`inputs-technodata` file,
   with values summing to 1 for each technology.
   See documentation for the :ref:`inputs-technodata` file for more details.

RegionName
   Region where an agent operates.

.. py:currentmodule:: muse.objectives

.. _Objective1:

Objective1
   First objective that an agent will try and maximize or minimize during investment.
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

   The weight associated with this objective can be changed using :ref:`ObjData1
   <ObjData1>`.  Whether the objective should be minimized or maximized depends on
   :ref:`Objsort1 <Objsort1>`. Multiple objectives are combined using the
   :ref:`DecisionMethod <DecisionMethod>`

.. _Objective2:

Objective2 (optional)
   Second objective. See :ref:`Objective1 <Objective1>`.

.. _Objective3:

Objective3 (optional)
   Third objective. See :ref:`Objective1 <Objective1>`.

.. _ObjData1:

ObjData1
   A weight associated with the :ref:`first objective <Objective1>`. Whether it is used
   will depend in large part on the :ref:`decision method <DecisionMethod>`.

ObjData2 (optional)
   A weight associated with the :ref:`second objective <Objective2>`. See :ref:`ObjData1
   <ObjData1>`.

ObjData3 (optional)
   A weight associated with the :ref:`third objective <Objective3>`. See :ref:`ObjData1
   <ObjData1>`.

.. _Objsort1:

Objsort1
   Sets whether :ref:`first objective <Objective1>` is maximized or minimized. For both
   "adhoc" and "scipy" solvers this should be set to "True" for minimization and
   "False" for maximisation.

Objsort2 (optional)
   Objsort parameter for :ref:`second objective <Objective2>`. See :ref:`Objsort1
   <Objsort1>`.

Objsort3 (optional)
   Objsort parameter for :ref:`third objective <Objective3>`. See :ref:`Objsort1
   <Objsort1>`.

.. py:currentmodule:: muse.filters

.. _SearchRule:

SearchRule
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

   - :py:func:`spend_limit <spend_limit>`: Only allows technologies with a unit capital cost (cap_par in
      :ref:`inputs-technodata`) lower than the spend limit.

   Filters can be combined by chaining them with "->". For example, "all->maturity->spend_limit".

.. py:currentmodule:: muse.decisions

.. _DecisionMethod:

DecisionMethod
   Decision methods reduce multiple objectives into a single scalar objective per
   replacement technology. They allow combining several objectives into a single metric
   through which replacement technologies can be ranked.

   Decision methods are any function which follow a given signature and are registered
   via the decorator :py:func:`@register_decision <register_decision>`. The following
   decision methods are available with MUSE, as implemented in
   :py:mod:`~muse.decisions`:

   - :py:func:`mean <mean>`: Computes the average across several objectives.
   - :py:func:`weighted_sum <weighted_sum>`: Computes a weighted average across several
     objectives.
   - :py:func:`lexical_comparion <lexical_comparison>`: Compares objectives using a
     binned lexical comparison operator. Aliased to "lexo". This is a `lexicographic method <https://en.wikipedia.org/wiki/Lexicographic_order>`_ where objectives are compared in a specific order, for example first costs, then environmental emissions.
   - :py:func:`retro_lexical_comparion <retro_lexical_comparison>`: A binned lexical
     comparison function where the bin size is adjusted to ensure the current crop of
     technologies are competitive. Aliased to "retro_lexo".
   - :py:func:`epsilon_constraints <epsilon_constraints>`: A comparison method which
     ensures that first selects technologies following constraints on objectives 2 and
     higher, before actually ranking them using objective 1. Aliased to "epsilon" and
     "epsilon_con".
   - :py:func:`retro_epsilon_constraints <retro_epsilon_constraints>`: A variation on
     epsilon constraints which ensures that the current crop of technologies are not
     deselected by the constraints. Aliased to "retro_epsilon".
   - :py:func:`single_objective <single_objective>`: A decision method to allow
     ranking via a single objective.

   The functions allow for any number of objectives. However, the format described here
   allows only for three.

Quantity
   Represents the fraction of new demand that is assigned to the agent
   (e.g. if 0.2, 20% of new demand in each year will be assigned to the agent).
   Must sum to 1 across all "New" agents.
   When using both "Retrofit" agents and "New" agents, this only applies to the "New" agents.

MaturityThreshold (optional)
   Only applies when using the :py:func:`maturity <muse.filters.maturity>` search rule.
   Allows agents to only consider technologies that have achieved a certain market share
   (e.g. if 0.5, the agent will only invest in technologies that have a current market share of 50% or more).

SpendLimit (optional)
   Only applies when using the :py:func:`spend_limit <muse.filters.spend_limit>` search rule.
   Allows agents to only consider technologies with a unit capital cost (`cap_par`) lower than the spend limit.
   (e.g. if 10, the agent will only invest in technologies with a `cap_par` of 10 or lower, as listed in the :ref:`inputs-technodata` file).
