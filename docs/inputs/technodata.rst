.. _inputs-technodata:

===========
Techno-data
===========
The techno-data includes the techno-economic characteristics of each technology such
as capital, fixed and variable cost, lifetime, utilisation factor.
The techno-data should follow the structure reported in the table. The column order
is not important and additional input data can alsobe read in this format. In the table,
the electric boiler used in households is taken as an example for a generic region, region1.


.. csv-table:: Techno-data: cost inputs
   :header: ProcessName, RegionName, Time, cap_par, cap_exp, fix_par, ...

   resBoilerElectric, region1, 2010, 3.81, 1.00, 0.38, ...
   resBoilerElectric, region1, 2030, 3.81, 1.00, 0.38, ...


ProcessName
   represents the technology ID and needs to be consistent across all the data inputs

RegionName
   represents the region ID and needs to be consistent across all the data inputs

Time
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation

cap_par, cap_exp
   are used in the capital cost estimation. Capital costs are calculated as:

   .. math::

      \text{CAPEX} = \text{cap$\_$par} * \text{(Capacity)}^\text{cap$\_$exp}

   where the parameter cap_par is estimated at a selected reference size (i.e. CapRef),
   such as:

   .. math::

      \text{cap$\_$par} = \left(
         \frac{\text{CAPEXref}}{\text{CapRef}}
      \right)^{\text{cap$\_$exp}}

   CapRef is a reference size for the cost estimate decided by the modeller before filling the input data files.

   This allows the model to take into account economies of scale. ie. As `Capacity` increases, the price of the technology decreases. This does not include technological learning parameters, where prices may come down due to learning.

fix_par, fix_exp
   are used in the fixed cost estimation. Fixed costs are calculated as:

   .. math::

      \text{FOM} = \text{fix$\_$par} * (\text{Capacity})^\text{fix$\_$exp}


   where the parameter fix_par is estimated at a selected reference capacity (i.e. CapRef),
   such as:

   .. math::

      \text{fix$\_$par}= \frac{\text{FOMref}}{(\text{CapRef})^\text{fix$\_$exp}}

   CapRef is a reference size for the cost estimate decided by the modeller before filling the input data files.

var_par, var_exp
   are used in the variable costs estimation. These variable costs are production
   dependent Variable costs are calculated as:

   .. math::

      \text{VAREX} = \text{var$\_$par} * \text{(Production)}^{\text{var$\_$exp}}

   where the parameter var_par is estimated at a selected reference size (i.e. CapRef),
   such as:

   .. math::

      \text{var$\_$par}= \frac{\text{VARref}}{(\text{ProductionRef})^\text{var$\_$exp}}

   ProductionRef is the production of a reference capacity (CapRef) for the cost estimate decided by the modeller before filling the input data files.

Growith constraints
   MaxCapacityAddition
      represents the maximum addition of installed capacity per technology, per year in a period, per region.

   MaxCapacityGrowth
      represents the fraction growth per year based on the available stock in a year, per region and technology.

   TotalCapacityLimit
      represents the total capacity limit per technology, region and year.

   .. csv-table:: Techno-data: growth constraints
      :header: ProcessName,	RegionName,	MaxCapacityAddition,	MaxCapacityGrowth,	TotalCapacityLimit

      resBoilerElectric, region1, 10,	0.2,	100

   In this example, MaxCapacityAddition,	MaxCapacityGrowth, and TotalCapacityLimit equal to 10 PJ, 0.2 (corresponding to 20 \%), and 100 PJ.
   Assuming a 5-year time step, the *MaxCapacityAddition* introduces a constraint on the maximum capacity which can be added in the investment year:
   it constrains the new capacity which can be installed in a modelled period as being equal to *10 * 5 = 50 PJ*.
   The *MaxCapacityGrowth* applies a constraint on the capacity which can be installed in a modelled period, which depends on the
   decommissioning profile. Assuming that 7.7 PJ of resBoilerElectric is available in the year when the decision is made (investment year), and that 4.9 PJ of
   resBoilerElectric is available in the year at which capacities invested in, will be online, then the constraint applies as follows *7.7 * ((1 + 0.2) ** 5) - 4.9 = 14.3 PJ*.
   The *TotalCapacityLimit* applies a constraint on the maximum capacity of a technology in the investment year; it depends on the decommissioning profile and equals *100 - 4.9 = 95.1 PJ*.

   Growth constraints are applied for each single agent in a multi-agent simulation. When only one agent is present, the growth constraints
   apply individually to the "New" and "Retrofit" agent, when present.


TechnicalLife
   represents the number of years that a technology operates before it is decommissioned.

UtilizationFactor
   represents the *maximum* actual output of the technology in a year, divided by the theoretical maximum output if the technology were operating at full capacity for the whole year. Must be between 0 and 1.

MinimumServiceFactor (optional, default = 0)
   Is the *minimum* output of the technology in a year, divided by the theoretical maximum output if the technology were operating at full capacity for the whole year. Must be between 0 and 1 and be smaller or equal than the `UtilizationFactor`. It is used to define the minimum service level that a technology must provide due to, typically, technical or efficiency constraints.

ScalingSize (optional)
   required when using the "capital_costs" agent objective. Represents the reference capacity at which capital costs are estimated when using this objective (see :ref:`inputs-agents`).

efficiency (optional)
   represents the technology efficiency. Required when using the "efficiency" agent objective, which ranks investment options according to their energy or material efficiency (see :ref:`inputs-agents`).

Type (optional)
   defines the type of a technology. Required when using the "similar_technology" search space, which allows agents to filter for technologies of a similar type (see :ref:`inputs-agents`).

Fuel (optional)
   defines the fuel used by a technology. Required when using the "fueltype" search space, which allows agents to filter for technologies using the same fuel (see :ref:`inputs-agents`).

EndUse (optional)
   defines the end use of a technology, defined to restrict the new investments of each agent to selected technologies using selected end uses (see :ref:`inputs-agents`).

InterestRate
   is the technology interest rate (called hurdle rates in other models).
   This is used for the interest used in the discount rate and corresponds to the interest built when borrowing money.

Agent0, ..., AgentN
   represent the proportion of initial capacity allocated to each agent.
   Must match AgentShare names specified in the :ref:`inputs-agents` file.
   All agents must be represented in the table.
   If using "New" and "Retrofit" agents, you should create a column with the name of each "Retrofit" agent share.
   If only using "New" agents, you should create a column with the name of each "New" agent share.
   The value corresponds to the ownership of the initial stock, as defined in the :ref:`inputs-existing-capacity` for the starting year of the simulation.

   For example, in a one-agent simulation, you should specify the following to indicate full ownership of existing capacity by the agent (assuming an agent share name of "Agent1"):

   .. csv-table:: Techno-data: AgentShare - 1 agent
      :header: ProcessName, RegionName, Time, ..., Agent1

      resBoilerElectric, region1, 2010, ..., 1
      resBoilerElectric, region1, 2030, ..., 1

   In a two-agent simulation, assuming a 30\% / 70\% split of initial capacity between the two agents, the table would be as follows:

   .. csv-table:: Techno-data: AgentShare - 2 agents
      :header: ProcessName, RegionName, Time, ..., Agent1, Agent2

      resBoilerElectric, region1, 2010, ..., 0.3, 0.7
      resBoilerElectric, region1, 2030, ..., 0.3, 0.7

   Values must sum to 1 for each row of the table.

The input data has to be provided for the base year. Additional years within the time
framework of the overall simulation can be defined. In this case, MUSE would interpolate
the values between the provided periods and assume a constant value afterwards. The additional
years at which input data for techno-data are defined need to equal for :ref:`inputs-iocomms` and :ref:`inputs-technodata-ts`.

Interpolation is activated only if the feature *interpolation_mode = 'Active'* is defined in the TOML file.
