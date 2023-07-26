Application Flow
================

While not essential to be able to use MUSE, it will be useful to know the sequence of events that a run of MUSE will follow in a bit more detail that the brief overview of the :ref:`MUSE Overview` section. Let's start with the big picture.

High level sequence
-------------------

Any MUSE simulation follows the steps outlined in the following graph:

.. graphviz::
    :align: center
    :alt: High level sequence of steps in a MUSE simulation

    digraph top_level {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            settings [label="Read\nsettings"]
            market [label="Create\nmarket"]
            sectors [label="Create\nsectors"]
            mca [label="Create\nMCA"]
            run [label="Last year?", shape=diamond, style=""]
            equilibrium [label="Find\nequilibrium"]
            propagate [label="Propagate\nprices"]
            outputs [label="Produce\noutputs"]

            subgraph cluster_1 {
                settings -> market -> sectors -> mca
                label="Initialization"
                color=lightgrey
            }

            subgraph cluster_2 {
                equilibrium -> propagate -> outputs -> run
                run -> equilibrium [label="No", constraint=false]
                label="Run"
                color=lightgrey
            }

            start -> settings
            mca -> equilibrium [label="year=1"]
            run -> end [label="Yes"]
        }

It has two main components, the **Initialisation** phase when the input settings file is read and, based on it, all the components needed for the simulation are created, and the **Run** phase when the actual simulation takes place and intermediate outputs are produced along the way.

Initialization
--------------

The initialization phase is where all the parameters of the simulation are pulled from the :ref:`input-files` and the relevant objects required to run the simulation are created. If there is any configuration that does not make sense, it should be spotted during this phase and the execution of MUSE interrupted (with a meaningful error message) so no time is wasted in running a simulation that is wrong.

Each of the steps above can be further split into smaller steps, as described in the following chart: