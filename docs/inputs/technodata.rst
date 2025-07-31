.. _inputs-technodata:

===========
Technodata
===========
The technodata includes the techno-economic characteristics of each technology such
as capital, fixed and variable cost, lifetime, utilization factor.
The technodata should follow the structure reported in the table below.
In this example, we show an electric boiler for a generic region, region1:

.. csv-table:: Technodata
   :header: technology, region, year, cap_par, cap_exp, fix_par, ...

   resBoilerElectric, region1, 2010, 3.81, 1.00, 0.38, ...
   resBoilerElectric, region1, 2030, 3.81, 1.00, 0.38, ...


``technology``
   represents the technology ID and needs to be consistent across all the data inputs

``region``
   represents the region ID (must match a region specified in the settings file)

``year``
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation

Capital costs (optional, default ``cap_par = 0`` and ``cap_exp = 1``)
   Two parameters are used in the capital cost estimation (``cap_par`` and ``cap_exp``),
   defined as:

   .. math::

      \text{CAPEX} = \text{cap$\_$par} * \text{(Capacity)}^\text{cap$\_$exp}

   The exponent allows the model to take into account economies of scale (ie. As `Capacity` increases, the capital cost of the technology decreases).

Fixed costs (optional, default ``fix_par = 0`` and ``fix_exp = 1``)
   Two parameters are used in the fixed cost estimation (``fix_par`` and ``fix_exp``),
   defined as:

   .. math::

      \text{FOM} = \text{fix$\_$par} * (\text{Capacity})^\text{fix$\_$exp}

   The exponent allows the model to take into account economies of scale (ie. As `Capacity` increases, the fixed cost of the technology decreases).

Variable costs (optional, default ``var_par = 0`` and ``var_exp = 1``)
   Two parameters are used in the variable cost estimation (``var_par`` and ``var_exp``),
   defined as:

   .. math::

      \text{VAREX} = \text{var$\_$par} * \text{(Production)}^{\text{var$\_$exp}}

   The exponent allows the model to take into account economies of scale (ie. As `Production` increases, the variable cost of the technology decreases).

Growth constraints (optional)
   ``max_capacity_addition``
      represents the maximum addition of installed capacity per technology, per year in a period, per region.

   ``max_capacity_growth``
      represents the fraction growth per year based on the available stock in a year, per region and technology.
      To allow growth to be initiated, a seed value must be specified (see ``growth_seed`` below).

   ``total_capacity_limit``
      represents the total capacity limit per technology, region and year.

   .. csv-table:: Techno-data: growth constraints
      :header: technology,	region,	max_capacity_addition,	max_capacity_growth,	total_capacity_limit

      resBoilerElectric, region1, 10,	0.2,	100

   In this example, ``max_capacity_addition``, ``max_capacity_growth`` and ``total_capacity_limit`` equal to 10 PJ, 0.2 (corresponding to 20 \%), and 100 PJ.
   Assuming a 5-year time step:

   * ``max_capacity_addition`` restricts new capacity which can be installed over the investment period to 10 * 5 = 50 PJ.
   * ``max_capacity_growth`` restricts capacity growth to 20 \% per year (:math:`\approx` 149 \% over 5 years).
     The investment limit will depend on the existing capacity and the decommissioning profile. Assuming that 7.7 PJ of resBoilerElectric is available in the current year, and that 4.9 PJ of
     resBoilerElectric is already commissioned for the investment year, then the constraint applies as follows: 7.7 * (1 + 0.2)\ :sup:`5` - 4.9 = 14.3 PJ.
     Also see the GrowthSeed parameter below.
   * ``total_capacity_limit`` will restrict new addition to 100 - 4.9 = 95.1 PJ (so that total capacity in the investment year will not exceed 100 PJ).
   * Overall, the most restrictive constraint will apply, which in this case is 14.3 PJ.

   Growth constraints are applied for each single agent in a multi-agent simulation. When only one agent is present, the growth constraints
   apply individually to the "New" and "Retrofit" agent, when present.

   If any of the three parameters are not provided in the technodata file, that particular constraint is not applied.

``growth_seed`` (optional, default = 1)
    applies a lower-bound on the initial capacity value used in the MaxCapacityGrowth calculation, allowing growth to initiate when capacity is low/zero.

    Taking the above example, if the GrowthSeed is set to 10 PJ (higher than the existing capacity of 7.7 PJ), the MaxCapacityGrowth constraint will be applied as follows:
    10 x (1 + 0.2)\ :sup:`5` - 4.9 = 19.9 PJ.

``technical_life``
   represents the number of years that a technology operates before it is decommissioned.

``utilization_factor`` (optional, default = 1)
   represents the *maximum* actual output of the technology in a year, divided by the theoretical maximum output if the technology were operating at full capacity for the whole year. Must be between 0 and 1.

``minimum_service_factor`` (optional, default = 0)
   Is the *minimum* output of the technology in a year, divided by the theoretical maximum output if the technology were operating at full capacity for the whole year. Must be between 0 and 1 and be smaller or equal than the `utilization_factor`. It is used to define the minimum service level that a technology must provide due to, typically, technical or efficiency constraints.

``interest_rate`` (optional, default = 0)
   is the technology interest rate (called hurdle rates in other models).
   This corresponds to the interest built when borrowing money for the capital costs of investment.

Agent shares (e.g. ``Agent0``, ..., ``AgentN``)
   represent the proportion of initial capacity allocated to each agent.
   Must match AgentShare names specified in the :ref:`inputs-agents` file.
   All agents must be represented in the table.
   If using "New" and "Retrofit" agents, you should create a column with the name of each "Retrofit" agent share.
   If only using "New" agents, you should create a column with the name of each "New" agent share.
   The value corresponds to the ownership of the initial stock, as defined in the :ref:`inputs-existing-capacity` for the starting year of the simulation.

   For example, in a one-agent simulation, you should specify the following to indicate full ownership of existing capacity by the agent (assuming an agent share name of "Agent1"):

   .. csv-table:: Techno-data: AgentShare - 1 agent
      :header: technology, region, year, ..., Agent1

      resBoilerElectric, region1, 2010, ..., 1
      resBoilerElectric, region1, 2030, ..., 1

   In a two-agent simulation, assuming a 30\% / 70\% split of initial capacity between the two agents, the table would be as follows:

   .. csv-table:: Techno-data: AgentShare - 2 agents
      :header: technology, region, year, ..., Agent1, Agent2

      resBoilerElectric, region1, 2010, ..., 0.3, 0.7
      resBoilerElectric, region1, 2030, ..., 0.3, 0.7

   Values must sum to 1 for each row of the table.

Additional optional columns
  Certain columns may also be required when using certain agent objectives or search spaces.
  These are:

  ``efficiency``
     a numerical value representing the technology efficiency, from 0 to 1.
     Required when using the "efficiency" agent objective, which ranks investment options according to their energy or material efficiency (see :ref:`inputs-agents`).
     Note: this has no impact on the commodity flows through the technology, it is merely intended as a customisable value that agents can use to rank technologies.

  ``comfort``
     a numerical value representing the comfort level of a technology, from 0 to 1.
     Required when using the "comfort" agent objective, which ranks investment options according to their comfort level (see :ref:`inputs-agents`).
     Like ``efficiency``, this is merely intended as a customisable value that agents can use to rank technologies.

  ``type``
     a string value that can be used to define the type of a technology (e.g. "nuclear", "electric vehicle").
     Required when using the "similar_technology" search space, which allows agents to filter for technologies of a similar type (see :ref:`inputs-agents`).


--------------------------------

The input data has to be provided for the base year, after which MUSE will assume
that values are constant for all subsequent years, if no further data is provided.
If users wish to vary parameters by year, they can provide rows for additional years.
In this case, MUSE would interpolate the values between the provided periods and assume
a constant value afterwards.

Note: if you wish to provide data for one technology in a different year, you must do
so for *all* technologies.
