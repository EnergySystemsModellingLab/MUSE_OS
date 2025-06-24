.. _toml-primer:

===========
TOML primer
===========

The full specification for TOML files can be found
`here <https://github.com/toml-lang/toml>`_.
A TOML file is separated into sections, with each section except the topmost
introduced by a name in square brackets. Sections can hold key-value pairs,
e.g. a name associated with a value. For instance:

.. code-block:: TOML

   general_string_attribute = "x"

   [some_section]
   section_attribute = 12

   [some_section.subsection]
   subsetion_attribute = true

TOML is quite flexible in how one can define sections and attributes. The following
three examples are equivalent:

.. code-block:: TOML

   [sectors.residential.production]
   name = "match"
   costing = "prices"


.. code-block:: TOML

   [sectors.residential]
   production = {"name": "match", "costing": "prices"}

.. code-block:: TOML

   [sectors.residential]
   production.name = "match"
   production.costing = "prices"


.. _toml-array:

Additionally, TOML files can contain tabular data, specified row-by-row using double
square bracket. For instance, below we define a table with two rows and a single
*column* called `some_table_of_data` (though column is not quite the right term, TOML tables are made more
flexible than most tabular formats. Rather, each row can be considered a
dictionary).

.. code-block:: TOML

   [[some_table_of_data]]
   a_key = "a value"

   [[some_table_of_data]]
   a_key = "another value"

.. Since MUSE requires a number of data files, paths to file can be formatted quite
.. flexibly. A `path` any key-value where the value ends with `.csv` or `.toml`,
.. as well any key which ends in `_path`, `_file`, or `_dir`, e.g. `data_path` or
.. `sector_dir`.  Paths can be formatted with shorthands for specific directories.
.. Shorth-hands are specified by curly-brackets:

As MUSE requires a number of data file, paths to files can be formatted in a flexible manner. Paths can be formatted with shorthands for specific directories and are defined with curly-brackets. For example:


.. code-block:: TOML

   projection = '{path}/inputs/projection.csv'
   timeslices_path = '{cwd}/technodata/timeslices.csv'

path
   refers to the directory where the TOML file is located

cwd
   refers to the directory from which the muse simulation is launched
