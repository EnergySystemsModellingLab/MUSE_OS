===
API
===


.. automodule:: muse

-------------------------
Market Clearing Algorithm
-------------------------

Main MCA
~~~~~~~~

.. automodule:: muse.mca
   :members:


Carbon Budget
~~~~~~~~~~~~~

.. automodule:: muse.carbon_budget
   :members:


------------------------------------
Sectors and associated functionality
------------------------------------

.. automodule:: muse.sectors

AbstractSector
~~~~~~~~~~~~~~

.. autodecorator:: muse.sectors.register_sector

.. autoclass:: muse.sectors.AbstractSector
   :members:


Sector
~~~~~~

.. autoclass:: muse.sectors.Sector
   :members:

PresetSector
~~~~~~~~~~~~

.. autoclass:: muse.sectors.PresetSector
   :members:

LegacySector
~~~~~~~~~~~~

.. autoclass:: muse.sectors.LegacySector
   :members:

Production
~~~~~~~~~~

.. automodule:: muse.production
   :members:


Agent Interactions
~~~~~~~~~~~~~~~~~~

.. automodule:: muse.interactions
   :members:


-------------------------------------
Agents and associated functionalities
-------------------------------------

.. automodule:: muse.agent
   :members: agents_factory, create_agent, create_retrofit_agent, create_newcapa_agent


.. autoclass:: muse.agent.AgentBase
   :members:

.. autoclass:: muse.agent.Agent
   :members:
   :private-members:


Objectives
~~~~~~~~~~

.. automodule:: muse.objectives
   :members:


Search Rules
~~~~~~~~~~~~

.. automodule:: muse.filters
   :members:


Decision Methods
~~~~~~~~~~~~~~~~

.. automodule:: muse.decisions
   :members:


Investment Methods
~~~~~~~~~~~~~~~~~~

.. automodule:: muse.investments
   :members:


Demand Share
~~~~~~~~~~~~

.. automodule:: muse.demand_share
   :members:

Constraints:
~~~~~~~~~~~~

.. automodule:: muse.constraints
    :members:


Initial and Final Asset Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: muse.hooks
   :members:


------------------
Reading the inputs
------------------

.. automodule:: muse.readers.toml
   :members:

.. automodule:: muse.readers.csv
   :members:

.. automodule:: muse.decorators
   :members:

---------------
Writing Outputs
---------------

.. automodule:: muse.outputs
   :members:

Sinks
~~~~~

.. automodule:: muse.outputs.sinks
   :members:


Sectorial Outputs
~~~~~~~~~~~~~~~~~

.. automodule:: muse.outputs.sector
   :members:


----------
Quantities
----------

.. automodule:: muse.quantities
   :members:

-------------------------
Demand Matching Algorithm
-------------------------

.. automodule:: muse.demand_matching
   :members:

-------------
Miscellaneous
-------------

Timeslices
~~~~~~~~~~

.. automodule:: muse.timeslices
   :members:

Commodities
~~~~~~~~~~~

.. automodule:: muse.commodities
   :members:


Regression functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: muse.regressions
   :members:


Functionality Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: muse.registration
   :members:

Utilities
~~~~~~~~~

.. automodule:: muse.utilities
   :members:


Examples
~~~~~~~~

.. automodule:: muse.examples
   :members:
