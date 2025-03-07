.. _inputs-technodata-ts:

======================
Techno-data Timeslices
======================
The techno-data timeslices is an optional file which contains information on technologies, their region, timeslices, utilization factors and minimum service factor. The objective of this file is to link the utilization factor and minimum service factor to timeslices. For instance, if you were to model solar photovoltaics, you would probably want to specify that they can not produce any electricity at night, or if you're modelling a nuclear power plant, that they must generate a minimum amount of electricity. The techno-data timeslice file enables you to do that. Note, that if this file is not present, the utilization facto will be used from the technodata file.


.. csv-table:: Techno-data
   :header: ProcessName,RegionName,Time,month,day,hour,UtilizationFactor,MinimumServiceFactor

   Unit,-,Year,-,-,-,-,-
   gasCCGT,R1,2020,all-year,all-week,night,1,1
   gasCCGT,R1,2020,all-year,all-week,morning,1,2


ProcessName
   represents the technology ID and needs to be consistent across all the data inputs

RegionName
   represents the region ID and needs to be consistent across all the data inputs

Time
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation

month
   represents the first level of the timeslice. This input is dynamic and so does not necessarily need to be a month, but a season for instance. As long as it matches with the toml file.

day
   represents the second level of the timeslice. Again, this input is dynamic, and thus does not necessarily need to be a day - as long as it matches with the toml file.

hour
   represents the third level of the timeslice.

UtilizationFactor
   represents the maximum actual output of the technology in a timeslice, divided by the theoretical maximum output if the technology were operating at full capacity for the whole timeslice. Must be between 0 and 1. This overwrites the UtilizationFactor in the technodata file.

MinimumServiceFactor
   represents the minimum service that a technology can output. For instance, the minimum amount of electricity that can be output from a nuclear power plant at a particular timeslice.


The input data has to be provided for the base year. Additional years within the time
framework of the overall simulation can be defined. In this case, MUSE would interpolate
the values between the provided periods and assume a constant value afterwards. The additional
years at which input data for techno-data timeslices, need to equal those for :ref:`inputs-iocomms` and :ref:`inputs-technodata`.
