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
            run [label="Last\nyear?", shape=diamond, style=""]
            equilibrium [label="Find market\nequilibrium"]
            propagate [label="Update\nprices"]
            outputs [label="Produce\noutputs"]
            check_c [label="Carbon\nbudget?", shape=diamond, style=""]
            c_prices [label="Update carbon\nprices"]
            next_year [label="Next year", shape=""]
            {node [style="invis"]; int1; int2;}


            subgraph cluster_1 {
                settings -> market -> sectors -> mca
                label="Initialization"
                color=lightgrey
            }

            subgraph cluster_2 {
                next_year -> check_c
                check_c -> equilibrium [label="No"]
                check_c -> c_prices [label="Yes", constraint=false]
                c_prices -> equilibrium [constraint=false]
                equilibrium -> propagate -> outputs -> run
                run -> next_year [label="No", constraint=false]
                label="Run"
                color=lightgrey
                {rank=same; check_c; c_prices}
            }

            start -> settings
            mca -> int1
            int2 -> next_year
            run -> end [label="Yes"]
            start -> int2 [style="invis", constraint=false]
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
        rankdir=LR
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

Create initial market
~~~~~~~~~~~~~~~~~~~~~

As described in :ref:`inputs-projection`, MUSE needs an initial market with prices and potential imports and exports of commodities to kick-off the simulation. These prices will be updated as the simulation progresses, or used as a static market throughout the whole timeline of the simulation.

This market object (an xarray Dataset, internally) will be instrumental throughout the simulation and regularly updated with supply, consumption and new prices, if relevant.

.. graphviz::
    :align: center
    :alt: Steps when creating the initial market

    digraph create_market {
        fontname="Helvetica,Arial,sans-serif"
        node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
        edge [fontname="Helvetica,Arial,sans-serif", len=2]
        rankdir=LR
        labelloc=t

        // nodes
        projections [label="Read projections"]
        exports [label="Read exports\n(optional)\nor exports=0"]
        imports [label="Read imports\n(optional)\nor imports=0"]
        interpolate [label="Interpolate\nto time framework"]
        initial [label="Set initial\nsupply=0\nconsumption=0"]

        projections -> interpolate
        exports -> interpolate
        imports -> interpolate
        interpolate -> initial
    }

Create sectors
~~~~~~~~~~~~~~

The sectors manage all the actors that will drive the evolution of the simulation: the technologies available, the commodities consumed and produced and the agents that will invest in the different technologies to ensure that the supply of commodities meets the demand. Sections :ref:`muse-components` and :ref:`input-files` provide more information on the different factors that influence sectors and their components.

During the initialisation step, all input files relevant to a sector are loaded, their consistency validated and the agents that will be investing in this sector created. A broad description of the steps involved in the creation of **each sector defined in the input file** are included in the following chart (there might be other validation and data reformatting steps).

.. graphviz::
    :align: center
    :alt: Simplified process of the creation of the sectors

    digraph sectors {

        fontname="Helvetica,Arial,sans-serif"
        node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
        edge [fontname="Helvetica,Arial,sans-serif"]
        rankdir=LR
        clusterrank=local
        newrank=true

        technodata [label="Read\ntechnodata"]
        coms_in [label="Read\ncommodities\nIN"]
        coms_out [label="Read\ncommodities\nOUT"]
        validate [label="Validate\ntechnologies"]
        outputs [label="Setup\noutputs"]
        interaction [label="Setup agents\ninteractions"]

        subgraph cluster_1 {
            label="For each subsector within sector"
            agents [label="Create\nagents"]
            share [label="Setup\ndemand share"]
            constraints [label="Setup\nconstraints"]
            investment [label="Setup\ninvestment"]
            capacity [label="Read initial\ncapacity"]
            agents -> share -> constraints -> investment
            agents -> capacity [color="red", constraint=false]
            capacity -> agents [color="blue", constraint=false]
            {rank=same; agents;capacity}
        }

        {
            technodata
            coms_in
            coms_out
        } -> validate
        validate -> agents
        investment -> outputs -> interaction
    }

Create the MCA
~~~~~~~~~~~~~~

The last step of the initialization is also the simplest one. The MCA (market clearing algorithm) is initialized with all the objects created in the previous sections and, specifically, the global simulation parameters, the handling of the carbon budget and the global outputs. Once the MCA is initialized, the simulation is ready to run!

.. graphviz::
    :align: center
    :alt: Steps of the creation of the MCA

    digraph mca {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            simulation [label="Setup global\nsimulation parameters"]
            outputs [label="Setup global\noutputs"]
            budget [label="Setup carbon\nbudget"]

            simulation -> budget -> outputs
        }

Simulation run
--------------

If the initialization is successful, the execution of the simulation will start. Depending on the configuration of the carbon budget and what to do with it, the steps will be slightly different, but in all cases the main part will be the steps for reaching the equilibrium between the demand and the supply based on the investment.

Update carbon prices
~~~~~~~~~~~~~~~~~~~~

One of MUSE core features is to (optionally) consider carbon emission as a constrain for the investment in future technologies. A carbon budget can be defined in the :ref:`simulation-settings` across all years of the simulation and this will result in an increase of prices for those technologies that are less green.

The sequence of steps related to the carbon budget control are as follows:

.. graphviz::
    :align: center
    :alt: Description of the carbon budget cycle

    digraph carbon_budget {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            single_year [label="Single year\niteration"]
            emissions [label="Calculate emissions\nof carbon comodities"]
            comparison [label="Emissions\n> budget\n", shape=diamond, style=""]
            new_price [label="Calculate new\ncarbon price"]


            subgraph cluster_1 {
                label="Initial estimate of carbon emissions in Future year"
                single_year -> emissions -> comparison
            }

            start -> single_year
            comparison -> end [label="No", constraint=false]
            comparison -> new_price [label="Yes"]
            new_price -> end
        }

The method used to calculate the new carbon price can be selected by the user. The only option built-in in MUSE at the moment is ``fitting``, however this can be expanded by the user with the ``@register_carbon_budget_method`` hook in ``muse.carbon_budget``. The ``fitting`` method is based in the following algorithm:

.. graphviz::
    :align: center
    :alt: Fitting method to calculate the new carbon price in the future year

    digraph carbon_budget_method {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            emissions [label="Calculate emissions\nof carbon comodities"];
            sample_prices [label="Sample\ncarbon prices"]
            all_samples [label="All samples\ndone?", shape=diamond, style=""]
            find_equilibrium [label="Find market\nequilibrium"]
            fit [label="Regression\nprices-emissions\nfor emissions=0"]
            refine [label="Refine\ncarbon price"]

            start -> sample_prices -> find_equilibrium -> emissions -> all_samples
            all_samples -> find_equilibrium [label="No", constraint=false]
            all_samples -> fit [label="Yes"]
            fit -> refine -> end
        }

As it can be seen, this method will run the ``Find market equilibrium`` algorithm multiple times and, as a result, the simulation will take significantly longer to complete.

Find market equilibrium
~~~~~~~~~~~~~~~~~~~~~~~

This is the main part of MUSE, in which the agents in the different sectors will invest in new - or old - technologies to make sure that the supply of commodities matches their demand in years to come across all the regions of the simulation.

An overall picture of this process can be seen in the following chart, but there are many fine-grained steps related to specific objectives and criteria that heavily influence the results of the calculation. These steps are described in other parts of the documentation.

.. graphviz::
    :align: center
    :alt: Main loop to find the market equilibrium

    digraph find_equilibrium {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            exclude [label="Exclude\ncommodities\nfrom market"];
            single_year [label="Single year\niteration"]
            maxiter [label="Maxium iter?", shape=diamond, style=""]
            converged [label="Converged?", shape=diamond, style=""]
            prices [label="Update with\nconverged prices"]
            {node [label="Update with not\nconverged prices"]; prices1; prices2;}
            check_demand [label="Check demand\nfulfilment"]
            equilibrium [label="Check\nequilibrium"]

            start -> exclude -> single_year -> equilibrium -> check_demand -> converged
            converged -> prices [label="Yes"]
            prices -> end
            converged -> maxiter [label="No"]
            maxiter -> prices2 [label="No"]
            maxiter -> prices1 [label="Yes"]
            prices1 -> end
            prices2 -> single_year [constraint=false]
            {rank=same; prices2; single_year}
            {rank=same; maxiter; converged;}
            {rank=same; prices1; prices;}
        }