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

.. autodecorator:: muse.sectors.register.register_sector

AbstractSector
~~~~~~~~~~~~~~

.. autoclass:: muse.sectors.AbstractSector
   :members:


Sector
~~~~~~

.. autoclass:: muse.sectors.sector.Sector
   :members:

Subsector
~~~~~~~~~

.. autoclass:: muse.sectors.subsector.Subsector

PresetSector
~~~~~~~~~~~~

.. autoclass:: muse.sectors.preset_sector.PresetSector
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

.. automodule:: muse.agents.factories
   :members: agents_factory, create_agent, create_retrofit_agent, create_newcapa_agent,
       factory


.. autoclass:: muse.agents.agent.AbstractAgent
   :members:

.. autoclass:: muse.agents.agent.Agent
   :members:
   :private-members:

.. autoclass:: muse.agents.agent.InvestingAgent
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
    :members: demand, factory, max_capacity_expansion, max_production, lp_costs,
        lp_constraint, lp_constraint_matrix, register_constraints, search_space,
        ScipyAdapter

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
