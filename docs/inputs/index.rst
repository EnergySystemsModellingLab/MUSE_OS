
.. _input-files:

Input Files
===========

In this section we detail each of the files required to run MUSE.
We include information based on how these files should be used, as well as the data that populates them.

All MUSE simulations require a settings file in TOML format (see :ref:`toml-primer` and :ref:`simulation-settings`), as well as a set of CSV files that provide the simulation data.

Whilst file names and paths are flexible and can be configured via the TOML,
a typical minimal file layout might look something like this:

example_model/
    - :ref:`settings.toml <simulation-settings>`
    - :ref:`GlobalCommodities.csv <inputs-commodities>`
    - :ref:`Projections.csv <inputs-projection>`
    - :ref:`Agents.csv <inputs-agents>`
    - sector1/
        - :ref:`Technodata <inputs-technodata>`
        - :ref:`CommoditiesIn.csv <inputs-iocomms>`
        - :ref:`CommoditiesOut.csv <inputs-iocomms>`
        - :ref:`ExistingCapacity.csv <inputs-existing-capacity>`
    - presets/
        - :ref:`Consumption2020.csv <preset-consumption>`
        - :ref:`Consumption2025.csv <preset-consumption>`

More complex models may use a different file structure. See full documentation below for more details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   toml_primer
   toml
   inputs_csv


Indices and tables
------------------


- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
