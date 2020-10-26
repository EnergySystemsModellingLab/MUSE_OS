.. _input-files:

================
MUSE input files
================

Broadly, MUSE consumes two kinds of inputs. Simulation *settings* specify how to
run the simulation, e.g. which sectors to run, for how many years, and what to
output. Simulation *data* describe the technologies involved in simulation, or
the number and kind of agents.

Simulation *settings* are specified in a TOML file. `TOML`_ is a simple, extensible,
and intuitive file format well suited for specifying small sets of fairly complex data.

Simulation *data* is specified in `CSV`_, a tabular data format not too dissimilar from
excel tables, at least conceptually.

MUSE requires at least:

* a single :ref:`general input file<simulation-settings>` for the simulation as a whole
* a file indicating initial market price :ref:`projections<inputs-projection>`
* a file describing the :ref:`commodities in the simulation<inputs-commodities>`
* for generalized sectors:
   * a file descring the :ref:`agents<inputs-agents>`
   * a file descring the :ref:`technologies<inputs-technodata>`
   * a file descring the :ref:`input commodities<inputs-icomms>` for each technology
   * a file descring the :ref:`output commodities<inputs-ocomms>` for each technology
   * a file descring the :ref:`existing capacity<inputs-existing-capacity>` of a given
     sector
* for each preset sector, which is used to model energy service demand:
   * a csv file describing consumption for the duration of the simulation
   * alternatively, a user can model a correlation of macroeconomic inputs (such as GDP and 
     Population) to describe how services consumprion can vary in time. Two files need to be 
     specified: a file describing the type of correlation function and the correlation parameters.
     By default, MUSE comes with an implementation of the following correlation functions between 
     energy ervice and macroeconomic driver: linear, logistic, log-log, and exponential 
     functions. Additional functions should be implemented by the user.
   

.. _TOML: https://github.com/toml-lang/toml
.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   inputs/toml_primer
   inputs/toml
   inputs/csv
