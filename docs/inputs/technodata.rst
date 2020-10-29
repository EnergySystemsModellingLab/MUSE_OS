.. _inputs-technodata:

===========
Techno-data
===========
The techno-data includes the techno-economic characteristics of each technology such
as capital, fixed and variable cost, lifetime, utilisation factor.
The techno-data should to follow the structure reported in the table although the column order
is not important and additional input data could be also read through this format. In the table,
the electric boiler used in households is taken as an example for a generic region, region1.


.. csv-table:: Techno-data
   :header: ProcessName, RegionName, Time, Level, cap_par, cap_exp, fix_par, ...
       
   resBoilerElectric, region1, 2010, fixed, 3.81, 1.00, 0.38, ...
   resBoilerElectric, region1, 2030, fixed, 3.81, 1.00, 0.38, ...


ProcessName
   represents the technology ID and needs to be consistent across all the data inputs

RegionName
   represents the region ID and needs to be consistent across all the data inputs

Time
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation

Level
   characterises either a fixed or a flexible input type

cap_par, cap_exp
   are used in the capital cost estimation. Capital costs are calculated as
   
   .. math::
   
      \text{CAPEX} = \text{cap\_par} * \text{Capacity}^\text{cap\_exp}

   where the parameter cap_par is estimated at a selected reference size (i.e. Capref),
   such as that
   
   .. math::

      \text{cap\_par} = \left(
         \frac{\text{CAPEXref}}{\text{Capref}}
      \right)^{\text{cap\_exp}}

   Capref is decided by the modeller before filling the input data files

fix_par, fix_exp

   are used in the fixed cost estimation. Fixed costs are calculated as
   
   .. math::
   
      \text{FOM} = \text{fix\_par} * (\text{Capacity})^\text{fix\_exp}

   where the parameter fix_par is estimated at a selected reference size (i.e. Capref),
   such as that

   .. math::

      \text{fix\_par} = \left(
         \frac{\text{FOMref}}{\text{Capref}}
      \right)^{\text{fix\_exp}}

   Capref is decided by the modeller before filling the input data files

var_par, var_exp
   are used in the variable costs estimation. These variable costs are capacity
   dependent Variable costs are calculated as 

   .. math::
   
      \text{VAREX} = \text{cap\_par} * \text{Capacity}^{\text{cap\_exp}}

   where the parameter var_par is estimated at a selected reference size (i.e. Capref),
   such as that
   
   .. math::

      \text{var\_par} = \left(
         \frac{\text{VARref}}{\text{Capref}}
      \right)^{\text{var\_exp}}

   Capref is decided by the modeller before filling the input data files

MaxCapacityAddition
   represents the maximum addition to installed capacity per technology, region, year

MaxCapacityGrowth
   represents the maximum growth in capacity as a fraction of the installed capacity per
   technology, region, year

TotalCapacityLimit
   represents the total capacity limit per technology, region, year 

TechnicalLife
   represents the number of years when a technology operates before decommissioning

UtilizationFactor
   is the number of operating hours of a process over maximum number of hours in a year      

ScalingSize
   represents the minimum size of a technology to be installed

efficiency
   is calculated as the ration between the total output commodities and the input commodities 

AvailabiliyYear
   defines the starting year of a technology; for example the value equals 1/0 when a
   technology would or would not be available

Type
   defines the type of a technology       

Fuel
   defines the fuel used by a technology  

EndUse
   defines the end use of a technology  

InterestRate
   is the technology interest rate

Agent_0, ..., Agent_N
   represent the allocation of the initial capacity to the each agent
   
The input data has to be provided for the base year. Additional years within the time
framework of the overall simulation can be defined. In this case, MUSE would interpolate
ghe values between the provided periods and assume a constant value afterwards.