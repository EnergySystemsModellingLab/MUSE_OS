
.. _input-files:

Input Files
===========

In this section we detail each of the files required to run MUSE.
We include information based on how these files should be used, as well as the data that populates them.

All MUSE simulations require a settings file in TOML format (see :ref:`toml-primer` and :ref:`simulation-settings`), as well as a set of CSV files that provide the simulation data.

Whilst file names and paths are fully flexible and can be configured via the settings TOML,
a typical minimal file layout might look something like this:

model_name/
    - :ref:`settings.toml <simulation-settings>`
    - :ref:`GlobalCommodities.csv <inputs-commodities>`
    - :ref:`Projections.csv <inputs-projection>`
    - sector1/
        - :ref:`Technodata <inputs-technodata>`
        - :ref:`CommoditiesIn.csv <inputs-iocomms>`
        - :ref:`CommoditiesOut.csv <inputs-iocomms>`
        - :ref:`ExistingCapacity.csv <inputs-existing-capacity>`
        - :ref:`Agents.csv <inputs-agents>`
    - presets/
        - :ref:`Consumption2020.csv <preset-consumption-file>`
        - etc.

Note, however, that this is just a convention, and more complex models may benefit from or require a different file structure.
See full documentation below for more details on the settings TOML and all the different types of data file.

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
