
Customising MUSE Tutorials
==========================

Next, we show you how to customise MUSE to create your own scenarios. 

We recommend following the tutorials step by step, as the files build on the previous example. If you prefer to jump straight in, your results may be different to the ones presented. To help you, we have provided the code to generate the various examples, in case you want to compare your code to ours. Links to this code can be found in the table below, in the `Tutorial Information`_ section.


.. toctree::
   :maxdepth: 1
   :caption: Tutorial contents:
   :numbered:

   add-solar
   add-agent
   add-region
   modify-timing-data
   addition-service-demand
   add-gdp-correlation-demand


Tutorial Information
--------------------


Below is a table to help you keep up with the different scenarios. The "Tutorial" header refers to the tutorial number as per the contents table. For example, Tutorial 1 is equal to "1. Adding a new technology". The "Agents", "Regions", "Residential Technologies", "Power Technologies" and "Gas Technologies" headers contains a tuple which contains the respective technologies present in the tutorial. The "Code" header provides a link to the files required to generate the tutorials.

In the table below, you will notice a x2 after some of the technology tuples. This refers to the fact that there are two regions, and the technologies within each region are the same for both regions.

.. list-table:: Scenarios
   :widths: 5 25 25 25 25 25 5
   :header-rows: 1

   *  - Tutorial
      - Agents
      - Regions
      - Residential Technologies
      - Power Technologies
      - Gas Technologies
      - Code
   *  - 1
      - [A1]
      - [R1]
      - [gasboiler, heatpump]
      - [gasCCGT, windturbine, solarPV]
      - [gassuply1]
      - [1]
   *  - 2
      - [A1, A2]
      - [R1]
      - [gasboiler, heatpump]
      - [gasCCGT, windturbine, solarPV]
      - [gassuply1]
      - [2]
   *  - 3
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - [3]
   *  - 4
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - [4]
   *  - 5
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump, electric_stove, gas_stove] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - [5]


