Correlation Demand Files
========================

It is possible to use macrodrivers as a way to infer the service demand. For example, one can use the expected GDP based on purchasing power parity (GDP PPP) and population in the future per region to infer the service demand using a regressor.

To do this, a minimum of three files are required:

#. A macrodrivers file

#. A file which states the regression parameters

#. A file which dictates how the demand per benchmark year is split across the timeslices.

We will go into the details of each of these files below.

Macrodrivers
------------

An example of a shortened macrodriver file is shown below. This file contains the data for each of the years you are interested in. For example, in the file below, it contains GDP PPP in region `R1`, in the unit `millionUS$2015` for each year. It also contains the data for the population.

.. list-table:: Macrodrivers
   :widths: 50 50 50 25 25 25
   :header-rows: 1

   * - variable
     - region
     - unit
     - 2010
     - 2011
     - ...
   * - GDP|PPP
     - R1
     - millionUS$2015
     - 1206919
     - 1220599
     - ...
   * - Population
     - R1
     - million
     - 80.0042
     - 81.82599
     - ...

``variable``
    This is the variable that you would like to use in the regression for the service demand.

``region``
    This is the region that the data applies to. This must correlate with the regions set in the rest of your input files, as well as the toml file.

``unit``
    This unit can be whatever you like, however they must be consistent across all input files.

Years (one column per year)
    This is the quantity of the variable per year of the simulation.


Regression Parameters
---------------------

In the regression parameters file, it is necessary to state the parameters of the regression. This can be obtained from your own dataset, where you regress the service demand against GDP PPP and populaiton, for example.

An example file is shown below:

.. csv-table:: Regression Parameters File
   :header: sector,function_type,coeff,region,electricity,gas,heat,CO2f

   Residential,logistic-sigmoid,GDPexp,R1,0,0,9.94E-02,0
   Residential,logistic-sigmoid,constant,R1,0,0,0.0000434,0
   Residential,logistic-sigmoid,GDPscaleLess,R1,0,0,753.1068725,0
   Residential,logistic-sigmoid,GDPscaleGreater,R1,0,0,672.9316672,0

``sector``
    This is the sector name in which these parameters apply to.

``function_type``
    This is the function type you would like to MUSE to use. MUSE allows these to be:

        - Exponential
        - ExponentialAdj
        - Logistic
        - Loglog
        - LogisticSigmoid
        - Linear
        - endogenous_demand

    Your own functions can be created using the `@register_regression` hook, from the `regressions.py` file.

``coeff``
    This is the coefficient for the respective function type. These are explicitly defined within the `regressions.py` file, as they differ between functions.

``region``
    This is the region in which these parameters apply to.

Commodities (one column per commodity)
    Here you can specify the coefficients for the expected demand for the respective commodity.


Timeslice share
---------------

In this file, you are able to split the energy service proportionally by timeslice.

An example file is shown below:

.. csv-table:: Timeslice share
   :header: timeslice,region,electricity,gas,heat,CO2f,wind

    1,R1,0,0,0.034835,0,0
    2,R1,0,0,0.064546,0,0
    3,R1,0,0,0.044569,0,0
    4,R1,0,0,0.011161,0,0
    5,R1,0,0,0.014145,0,0
    6,R1,0,0,0.085783,0,0

``timeslice``
    This is the timeslice index.

``region``
    This is the region in question for this data.

Commodities (one column per commodity)
    Here you specify the proportion of each energy service for each timeslice.
