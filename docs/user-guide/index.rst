.. _customising-muse:

Customising MUSE Tutorials
==========================

Next, we show you how to customise MUSE to create your own scenarios.

We recommend following the tutorials step by step, as the files build on the previous examples. If you prefer to jump straight in, your results may be different to the ones presented. To help you, we have provided the code to generate the various examples in case you want to compare your code to ours. Links to this code can be found in the table below, in the `Tutorial Information`_ section.


.. toctree::
   :maxdepth: 1
   :caption: Tutorial contents:

   add-solar
   add-agent
   add-region
   modify-timing-data
   additional-service-demand
   add-gdp-correlation-demand
   min-max-timeslice-constraints


Tutorial Information
--------------------


Below is a table to help you keep up with the different scenarios. The "Tutorial" header refers to the tutorial number as per the contents table. For example, Tutorial 1 is equal to "1. Adding a new technology". The "Agents", "Regions", "Residential Technologies", "Power Technologies" and "Gas Technologies" headers contain a tuple which contains the respective technologies present in the tutorial. The "Code" header provides a link to the files required to generate the tutorials.

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
      - `[1] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/1-add-new-technology>`_
   *  - 2
      - [A1, A2]
      - [R1]
      - [gasboiler, heatpump]
      - [gasCCGT, windturbine, solarPV]
      - [gassuply1]
      - `[2] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/2-add-agent>`_
   *  - 3
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - `[3] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/3-add-region>`_
   *  - 4
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - `[4] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/4-modify-timing-data>`_
   *  - 5
      - [A1, A2] x2
      - [R1, R2]
      - [gasboiler, heatpump, electric_stove, gas_stove] x2
      - [gasCCGT, windturbine, solarPV] x2
      - [gassuply1] x2
      - `[5] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/5-add-service-demand>`_
   *  - 6
      - [A1]
      - [R1]
      - [gasboiler, heatpump]
      - [gasCCGT, windturbine, solarPV]
      - [gassuply1]
      - `[6] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/6-add-correlation-demand>`_
   *  - 7
      - [A1]
      - [R1]
      - [gasboiler, heatpump]
      - [gasCCGT, windturbine]
      - [gassuply1]
      - `[7] <https://github.com/SGIModel/MUSE_OS/tree/main/docs/tutorial-code/7-min-max-timeslice-constraints>`_
