.. _inputs-technodata-ts:

======================
Technodata Timeslices
======================
The techno-data timeslices is an optional file which allows technology utilization factors and minimum service factors to be specified for each timeslice.
For instance, if you were to model solar photovoltaics, you would probably want to specify that they can not produce any electricity at night, or if you're modelling a nuclear power plant, that they must generate a minimum amount of electricity.

.. csv-table:: Techno-data
   :header: technology,region,year,month,day,hour,utilization_factor,minimum_service_factor

   gasCCGT,R1,2020,all-year,all-week,night,1,1
   gasCCGT,R1,2020,all-year,all-week,morning,1,2


``technology``
   represents the technology ID and needs to be consistent across all the data inputs

``region``
   represents the region ID and needs to be consistent across all the data inputs

``year``
   represents the period of the simulation to which the value applies; it needs to
   contain at least the base year of the simulation

Timeslice levels (e.g. ``month``, ``day``, ``hour``)
    One column for each timeslice level specified in the toml file.
    Together, you should specify one row for each timeslice defined in the model.
    e.g. the above table would be valid for a model with two timeslices:
    - ``all-year,all-week,night``
    - ``all-year,all-week,morning``

``utilization_factor``
   represents the maximum actual output of the technology in a timeslice, divided by the theoretical maximum output if the technology were operating at full capacity for the whole timeslice. Must be between 0 and 1 (default = 1).

``minimum_service_factor``
   represents the minimum service that a technology can output. For instance, the minimum amount of electricity that can be output from a nuclear power plant at a particular timeslice. Must be between 0 and 1 (default = 0).

--------------------------------

The input data has to be provided for the base year, after which MUSE will assume
that values are constant for all subsequent years, if no further data is provided.
If users wish to vary parameters by year, they can provide rows for additional years.
In this case, MUSE would interpolate the values between the provided periods and assume
a constant value afterwards.

Utilization factors and minimum service factors defined in this file will override any values defined in the technodata file for all years of the simulation.
If data for a particular region/process is not defined in this file, then the values defined in the technodata file will be used for all timeslices.
