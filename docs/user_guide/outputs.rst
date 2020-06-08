.. _output-files:

==================
MUSE output files
==================

In addition to sector outputs, MUSE outputs are specified through the simulation *settings*, defined in the input TOML file. 

By default, the output is reported by variable type and is the concatenation of
the calculated output for each region, sector, technology, and agent
for each simulated year.

In the reported example, the output represents the capacity of each asset owned by each retrofit agent
, broken down by sector, for each milestone year, region,



.. csv-table:: Capacity of installed technologies, example output
   :header: Year,	Region,	Sector,	Technology,			Agent, Type,	Capacity
   
   2020,	R1,		residential,	gasboiler,	A1,		retrofit, 10
   2025,	R1,		residential,	gasboiler,	A1,		retrofit, 5
   2025,	R1,		residential,	heatpump,	A1,		retrofit, 1
   2030,	R1,		residential,	heatpump,	A1,		retrofit, 2
   2020,	R1,		supply,   		gassupply1, A1,		retrofit, 10
   2025,	R1,		supply,			gassupply1,	A1,		retrofit, 5
   2030,	R1,		supply,			gassupply1,	A1,		retrofit, 5


Year
   represents the modelled period reporting the milestone
   year of the simulation, 
   consistent with the *time_frame* in the *settings*

Region
   represents the modelled regions reporting the corresponding 
   region ID, consistent with the data inputs

Sector
   represents the modelled sector, reporting the corresponding 
   sector ID, consistent with the data inputs

Technology   
   represents the modelled technologies, reporting the technology ID,
   consistent with the data inputs
   
Agent
	represents the agent owning each technology, in each sector, and region.
	Agents ID is consistent with the assumptions of the agent name in the data input

Type
   represents the category of the agent owning an asset, by sector, and region.
   By construction, only retrofit agent would own asset capacity different from zero
   in a milestone year: any new agent would only invest in new asset which would become available
   only in subsequent years.

Capacity
   reports the technology capacity of each technology as owned by each agent.
   Units are implicitly equal to those of the data inputs.
   This means that if the data inputs refer to costs and efficiencies in a PJ unit of measure,
   the same convention is followed in the results, unless further conversions are implemennted by
   the user following the conventions.