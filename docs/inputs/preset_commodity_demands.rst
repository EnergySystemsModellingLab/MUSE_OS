=============================
Preset Commodity Demands
=============================

.. _preset-consumption-file:

This document describes the CSV files used to supply pre-set commodity consumption
profiles to MUSE. These files are referenced from the TOML setting ``consumption_path``
and are typically provided one file per year (file names must include the year, e.g.
``Consumption2015.csv``). Wildcards are supported in the path (for example
``{cwd}/Consumption*.csv``).

The CSV format should follow the structure shown in the example below.

.. csv-table:: Consumption
   :header: "RegionName", "Timeslice", "electricity", "diesel", "algae"
   :stub-columns: 2

   USA,1,1.9,0,0
   USA,2,1.8,0,0

``RegionName``
   The region identifier. Must match region IDs used across other inputs.

``Timeslice``
   Index of the timeslice, according to the timeslice definition in the settings TOML.
   Indexing starts at 1 (i.e. the first timeslice defined in
   the global timeslices definition is 1, the second is 2, etc).

Commodities (one column per commodity)
   Any additional columns represent commodities. Column names must match the
   commodity identifiers defined in the global commodities file. Values are the
   consumption quantities for that timeslice and region.
