Application Flow
================

While not essential to be able to use MUSE, it is useful to know the sequence of events that a run of MUSE will follow in a bit more detail that the brief overview of the :ref:`MUSE Overview` section. Let's start with the big picture.

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

Each of the steps above can be further split into smaller steps, described individually in the following sections:

Read settings
~~~~~~~~~~~~~

The settings TOML file is where the higher level configuration of the simulation is defined. See :ref:`simulation-settings` for details of its content and structure. During the initialisation, the file is read, merged with default settings included in MUSE to get those parameters that are required and not provided by the user and, finally, the resulting settings are validated.

The validation step covers a wide range of checks (and more that can be added by the user via plugins) that not only asses if the relevant information is correct but that, in some cases, also create the relevant Python objects or normalizes it to some defined format.

.. graphviz::
    :align: center
    :alt: Read settings detailed flow chart

    digraph read_settings {
        fontname="Helvetica,Arial,sans-serif"
        node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
        edge [fontname="Helvetica,Arial,sans-serif", len=2]
        rankdir=TB
        labelloc=t

        usettings [label="Read user\nsettings"]
        dsettings [label="Read default\nsettings"]
        settings [label="Merge settings"]
        plugins [label="Check user plugins"]
        validate [label="Validate\nsettings"]

        usettings -> settings
        dsettings -> settings
        settings -> plugins -> validate

        validate -> {
            "Log level"
            "Interpolation\nmode"
            "Carbon budget\nparameters"
            "Foresight"
            "Iteration control"
            "Timeslices"
            "Global\ndata files"
            "Sector files"
            "... others"
        } [dir=both color="red:blue"]
    }
