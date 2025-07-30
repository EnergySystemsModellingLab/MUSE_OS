Application Flow
================

While not essential to be able to use MUSE, it is useful to know the sequence of events that a run of MUSE will follow in a bit more detail that the brief overview of the :ref:`MUSE Overview` section. Let's start with the big picture.

.. note::

    Throughout this section, greyed nodes will be further described in a more detailed chart, so keep reading to find out more about those, probably unclear steps.

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
            settings [label="Read\nsettings", fillcolor="lightgrey", style="rounded,filled"]
            market [label="Create\nmarket", fillcolor="lightgrey", style="rounded,filled"]
            sectors [label="Create\nsectors", fillcolor="lightgrey", style="rounded,filled"]
            mca [label="Create\nMCA", fillcolor="lightgrey", style="rounded,filled"]
            run [label="Last\nyear?", shape=diamond, style=""]
            equilibrium [label="Find market\nequilibrium", fillcolor="lightgrey", style="rounded,filled"]
            propagate [label="Update\nprices"]
            outputs [label="Produce\noutputs"]
            check_c [label="Carbon\nbudget?", shape=diamond, style=""]
            c_prices [label="Update carbon\nprices", fillcolor="lightgrey", style="rounded,filled"]
            next_year [label="Next year", shape=""]
            {node [style="invis"]; int1; int2;}


            subgraph cluster_1 {
                settings -> market -> sectors -> mca
                label="initialisation"
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

initialisation
--------------

The initialisation phase is where all the parameters of the simulation are pulled from the :ref:`input-files` and the relevant objects required to run the simulation are created. If there is any configuration that does not make sense, it should be spotted during this phase and the execution of MUSE interrupted (with a meaningful error message) so no time is wasted in running a simulation that is wrong.

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

The last step of the initialisation is also the simplest one. The MCA (market clearing algorithm) is initialized with all the objects created in the previous sections and, specifically, the global simulation parameters, the handling of the carbon budget and the global outputs. Once the MCA is initialized, the simulation is ready to run!

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

If the initialisation is successful, the execution of the simulation will start. Depending on the configuration of the carbon budget and what to do with it, the steps will be slightly different, but in all cases the main part will be the steps for reaching the equilibrium between the demand and the supply based on the investment.

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
            single_year [label="Single year\niteration", fillcolor="lightgrey", style="rounded,filled"]
            emissions [label="Calculate emissions\nof carbon comodities"]
            comparison [label="Emissions\n> budget\n", shape=diamond, style=""]
            new_price [label="Calculate new\ncarbon price", fillcolor="lightgrey", style="rounded,filled"]


            subgraph cluster_1 {
                label="Initial estimate of carbon emissions in Future year"
                single_year -> emissions -> comparison
            }

            start -> single_year
            comparison -> end [label="No", constraint=false]
            comparison -> new_price [label="Yes"]
            new_price -> end
        }

The **method used to calculate the new carbon price** can be selected by the user. There are currently only two options for this method, ``fitting`` and ``bisection``, however this can be expanded by the user with the ``@register_carbon_budget_method`` hook in ``muse.carbon_budget``.

The ``fitting`` method is based in the following algorithm:

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

The ``bisection`` method is a custom implementation of the `well known bisection algorithm <https://en.wikipedia.org/wiki/Bisection_method>`_ on the carbon price to minimize the difference between the carbon budget and the carbon emissions.

Both methods will run the ``Find market equilibrium`` algorithm multiple times and, as a result, the simulation will take significantly longer to complete than if no carbon budget is considered.

.. _find-equilibrium:

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
            single_year [label="Single year\niteration", fillcolor="lightgrey", style="rounded,filled"]
            maxiter [label="Maximum iter?", shape=diamond, style=""]
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

Single year iteration
~~~~~~~~~~~~~~~~~~~~~

Both in the carbon budget and in the equilibrium calculation, a single year iteration step is involved. It is in this step where MUSE will go through each sector and use the agents to appropriately invest in different technologies, aiming to match these two factors.

**As sectors have different priorities, sectors with lower priorities (larger numbers) will run last and see a market updated by the higher priority sectors**. In general, demand sectors should run before conversion sectors and these before supply sectors, such that the later can see the real demand. Running each sector will update their commodities, consumption and production. Balancing them is the purpose of the :ref:`find-equilibrium` loop described above, where the prices of the commodities are updated due to the change in their demand occurring during the single year iteration.

A chart summarising this process is depicted below:

.. graphviz::
    :align: center
    :alt: Steps of a single year iteration

    digraph single_year {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            next [label="Next\nsector", shape=""]
            run_sector [label="Run sector", fillcolor="lightgrey", style="rounded,filled"]
            consumption [label="Update market\nconsumption"];
            supply [label="Update market\nsupply"];
            all_done [label="All sectors\ndone?", shape=diamond, style=""]


            start -> next -> run_sector -> consumption -> supply -> all_done
            all_done -> end [label="Yes"]
            all_done -> next [label="No", constraint=false]
        }

With the run of each sector involving the following steps:

.. graphviz::
    :align: center
    :alt: Steps of one period step in a sector

    digraph sector_step {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            next [label="Next\nsub-sector", shape=""]
            interactions [label="Run agents\ninteractions"]
            invest [label="Investment", fillcolor="lightgrey", style="rounded,filled"]
            update_agents [label="Update agents'\nassets"]
            all_subsectors [label="Sub-sectors\ndone?", shape=diamond, style=""]
            dispatch [label="Dispatch", fillcolor="lightgrey", style="rounded,filled"]
            input_net[label="net\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]
            input_interaction[label="interaction\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]

            subgraph cluster {

            }

            start  -> interactions -> next -> invest -> update_agents -> all_subsectors
            all_subsectors -> next [label="No", constraint=false]
            input_net -> interactions [constraint=false]
            input_interaction -> interactions [constraint=false]
            all_subsectors -> dispatch [label="Yes"]
            dispatch -> end
        }

This deeper level of the process is where most of the input options of MUSE are put in use to decide how the agents behave, in what sort of technologies they invest, what metrics are used to make these decisions and how the dispatch of commodities takes place in order to fulfil the demand.


Investment
~~~~~~~~~~

In the investment step is where new capacity is added to the different assets managed by the agents. This investment might be needed to cover an increase in demand (between now and the investment year) or to match decommissioned assets, typically to do both.

The following graph summarises the process.

.. graphviz::
    :align: center
    :alt: Investment stage of a subsector calculation

    digraph dispatch {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            demand_share [label="Calculate\ndemand share"]
            search [label="Find technology\nsearch space"];
            objectives [label="Calculate\nobjectives"];
            decision [label="Calculate\ndecision"];
            constrains [label="Calculate\nconstrains"]
            invest [label="Solve\ninvestment"]
            input_demand[label="demand_share\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]
            input_search[label="SearchRule\n(Agents.csv)", fillcolor="#ffb3b3", style="rounded,filled"]
            input_objectives[label="Objective\n(Agents.csv)s", fillcolor="#ffb3b3", style="rounded,filled"]
            input_decision[label="DecisionMethod\n(Agents.csv)", fillcolor="#ffb3b3", style="rounded,filled"]
            input_constrains[label="Constrains\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]
            input_solver[label="lpsolver\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]

            start ->  demand_share -> search -> objectives -> decision -> constrains -> invest -> end
            input_demand -> demand_share
            input_search -> search
            input_objectives -> objectives
            input_decision -> decision
            input_constrains -> constrains
            input_solver -> invest
        }

First the demand is distributed among the available agents as requested by the ``demand_share`` argument of each ``subsector`` in the ``settings.toml`` file. This distribution can be done based on any attribute or property of the agents, as included in the ``Agents.csv`` file. Demand can also be shared across multiple agents, depending on the "quantity" attribute (defined in ``Agents.csv``). The two built-in options in MUSE are:

- `standard_demand` (default): The demand is split only amongst *new* agents (indeed there will be an error if a *retro* agent is found for this subsector). *New* agents get a share of the increase in demand over the investment period as well as the demand that occurs from decommissioned assets.
- `new_and_retro`: The input demand is split amongst both *new* and *retro* agents. *New* agents get a share of the increase in demand for the investment period, whereas *retrofit* agents are assigned a share of the demand that occurs from decommissioned assets.

Then, each agent select the technologies it can invest in based on what is needed and the **search rules** defined for it in the ``Agents.csv`` file. The possible search rules are described in :py:mod:`muse.filters`. These determine the search rules for each replacement technology.

For those selected replacement technologies, an objective function is computed. This value is a well defined economic concept, like LCOE or NPV, or a combination of them, and will be used to prioritise the investment of some technologies over others. As above, these objectives are defined in the ``Agents.csv`` file for each of the agents. Available objectives are described in :py:mod:`muse.objectives`.

Then, a decision is computed. Decision methods reduce multiple objectives into a single scalar objective per replacement technology. The decision method to use is selected in the ``Agents.csv`` file. They allow combining several objectives into a single metric through which replacement technologies can be ranked. See :py:mod:`muse.decisions`.

The final step of preparing the investment process is to compute the constrains, e.g. factors that will determine how much a technology could be invested in and include things like matching the demand, the search rules calculated above, the maximum production of a technology for a given capacity or the maximum capacity expansion for a given time period. Available constrains are set in the subsector section of the ``settings.toml`` file and described in :py:mod:`muse.constrains`. By default, all of them are applied. Note that these constrains might result in unfeasible situations if they do not allow the production to grow enough to match the demand. This is one of the common reasons for a MUSE simulation not converging.

With all this information, the investment process can proceed. This is done per sector using the method described by the ``lpsolver`` in the ``settings.toml`` file. Available solvers are described in :py:mod:`muse.investments`

If the investment succeeds, the new installed capacity will become part of the agents' assets.


Dispatch
~~~~~~~~

The dispatch stage when running a sector can be described by the following graph:

.. graphviz::
    :align: center
    :alt: Dispatch stage of a sector calculation

    digraph dispatch {
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif", shape=box, style=rounded]
            edge [fontname="Helvetica,Arial,sans-serif"]
            rankdir=LR
            clusterrank=local
            newrank=true

            {node [shape=""]; start; end;}
            capacity [label="Aggregate capacity\nfrom all agents"]
            supply [label="Calculate supply"];
            consumption [label="Calculate consumption"];
            cost [label="Calculate cost"];
            market [label="Create sector\nmarket"]
            dispatch[label="dispatch_production\n(settings.toml)", fillcolor="#ffb3b3", style="rounded,filled"]


            start ->  capacity -> supply -> consumption -> cost -> market -> end
            dispatch -> supply
        }

After the investment stage is completed, then the new capacity of the sector is obtained by aggregating the assets of all agents of the sector. Then, the supply of commodities is calculated as requested by the ``dispatch_production`` argument defined for each sector in the ``settings.toml`` file.

There are two possible options for ``dispatch_production`` built into MUSE:
- ``share``: assets each supply a proportion of demand based on their share of total capacity.
- ``maximum``: all the assets dispatch their maximum production, regardless of the demand.

Once the supply is obtained, the consumed commodities required to achieve that production level are calculated. The cheapest fuel for flexible technologies is used.

Finally, the cost associated with that supply is calculated as the weighted average *annual LCOE* over assets, where the weights are the supply. This is later used to set the new prices.
