.. _simulation-settings:

=====================
Simulation settings TOML file
=====================

.. currentmodule:: muse

This section details the TOML input for MUSE. The format for TOML files is
described in this :ref:`previous section <toml-primer>`. Here, however, we focus on sections and
attributes that are specific to MUSE.

The TOML file can be read using :py:func:`~readers.toml.read_settings`. The resulting
data is used to construct the market clearing algorithm directly in the :py:meth:`MCA's
factory function <mca.MCA.factory>`.

------------
Main section
------------

This is the topmost section. It contains settings relevant to the simulation as
a whole.

.. code-block:: TOML

   time_framework = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
   regions = ["USA"]
   interpolation_mode = 'linear'
   log_level = 'info'
   currency = 'USD'

   equilibrium_variable = 'demand'
   maximum_iterations = 100
   tolerance = 0.1
   tolerance_unmet_demand = -0.1

``time_framework``
   Required. List of years for which the simulation will run.

``regions``
   List of regions in the model.

``interpolation_mode`` (optional, default = **linear**)
   interpolation when reading the initial market. Options are:

   * **linear** interpolates using a `linear function <https://en.wikipedia.org/wiki/Linear_interpolation>`_.

   * **nearest** uses `nearest-neighbour interpolation <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>`_. That is, the closest known data point is used as the prediction for the data point to interpolate.

   * **zero** assumes that the data point to interpolate is zero.

   * **slinear** uses a spline of order 1. This is a similar method to `linear` interpolation.

   * **quadratic** uses a quadratic equation for interpolation. This should be used if you expect your data to take a quadratic form.

   * **cubic** interpolation is similar to `quadratic` interpolation, but uses a cubic function for interpolation.


``log_level`` (optional, default = **info**)
   verbosity of the output. Valid options, from the highest to the lowest level of verbosity, are: **debug**, **info**, **warning**, **error**, **critical**.

``currency`` (optional)
   The currency used for prices (e.g. "USD", "EUR", "MUS$2010"). All prices in all input
   files should be expressed in this currency. For example, if the currency is "USD",
   then capital costs should be expressed as "USD" per capacity unit, and price
   projections for a commodity with units "PJ" should be expressed as "USD/PJ". No
   currency conversions are performed in the model, and the choice of currency has no
   impact on the simulation - it is purely for documentation purposes.

``equilibirum_variable`` (optional, default = **demand**)
   whether equilibrium of **demand** or **prices** should be sought.

``maximum_iterations`` (optional, default = 100)
   Maximum number of iterations when searching for equilibrium.

``tolerance`` (optional, default = 0.1)
   Tolerance criteria when checking for equilibrium.
   e.g. 0.1 signifies that 10% of a deviation is allowed among the iterative value of either demand or price over a year per region.

``tolerance_unmet_demand`` (optional, default = -0.1)
   Criteria checking whether the demand has been met.

``excluded_commodities`` (optional)
   List of commodities excluded from the equilibrium considerations.

``plugins`` (optional)
    Path or list of paths to extra python plugins, i.e. files with registered functions
    such as :py:meth:`~muse.outputs.register_output_quantity`.


------------------
Global input files
------------------

Defines the paths specific simulation data files. The paths can be formatted as
explained in the :ref:`toml-primer`.

.. code-block:: TOML

   [global_input_files]
   projections = '{path}/Projections.csv'
   global_commodities = '{path}/GlobalCommodities.csv'

``projections``
   Path to a csv file giving initial market projection. See :ref:`inputs-projection`.

``global_commodities``
   Path to a csv file describing the comodities in the simulation. See
   :ref:`inputs-commodities`.

.. _timeslices_toml:

----------
Timeslices
----------

Time-slices represent a sub-year disaggregation of commodity demand. Generally,
timeslices are expected to introduce several levels, e.g. season, day, or hour.
For example:

.. code-block:: TOML

    [timeslices]
    level_names = ["month", "day", "hour"]
    winter.weekday.night = 396
    winter.weekday.morning = 396
    winter.weekday.afternoon = 264
    winter.weekday.early-peak = 66
    winter.weekday.late-peak = 66
    winter.weekday.evening = 396
    winter.weekend.night = 156
    winter.weekend.morning = 156
    winter.weekend.afternoon = 156
    winter.weekend.evening = 156
    spring-autumn.weekday.night = 792
    spring-autumn.weekday.morning = 792
    spring-autumn.weekday.afternoon = 528
    spring-autumn.weekday.early-peak = 132
    spring-autumn.weekday.late-peak = 132
    spring-autumn.weekday.evening = 792
    spring-autumn.weekend.night = 300
    spring-autumn.weekend.morning = 300
    spring-autumn.weekend.afternoon = 300
    spring-autumn.weekend.evening = 300
    summer.weekday.night = 396
    summer.weekday.morning  = 396
    summer.weekday.afternoon = 264
    summer.weekday.early-peak = 66
    summer.weekday.late-peak = 66
    summer.weekday.evening = 396
    summer.weekend.night = 150
    summer.weekend.morning = 150
    summer.weekend.afternoon = 150
    summer.weekend.evening = 150

This input introduces three levels, via ``level_names``: **month**, **day**, **hour**.
Other simulations may want fewer or more levels.  The **month** level is split into
three points of data, *winter*, *spring-autumn*, *summer*. Then **day** splits out
weekdays from weekends, and so on. Each line indicates the number of hours for the
relevant slice. It should be noted that the slices are not a cartesian products of each
levels. For instance, there no *peak* periods during weekends. All that matters is
that the relative weights (i.e. the number of hours) are consistent and sum up to a
year.


----------------
Standard sectors
----------------

A MUSE model requires at least one sector.
Sectors are declared in the TOML file by adding a subsection to the ``sectors`` section:

.. code-block:: TOML

   [sectors.residential]
   type = 'default'
   [sectors.power]
   type = 'default'

Above, we've added two sectors, residential and power. The name of the subsection is
only used for identification. In other words, it should be chosen to be meaningful to
the user, since it will not affect the model itself.

A sector accepts these attributes:

.. _sector-type:

``type``
   Defines the kind of sector this is. There are two options:

   * **default**: defines a standard sector
   * **presets**: defines a preset sector (see below)

.. _sector-priority:

``priority``
   An integer denoting which sectors runs when. Lower values imply the sector will run
   earlier. Later sectors can depend on earlier
   sectors for the their input. If two sectors share the same priority, then their
   order is not defined. Indeed, it should indicate that they can run in parallel.
   For simplicity, the keyword also accepts standard values:

   - **preset**: 0
   - **demand**: 10
   - **conversion**: 20
   - **supply**: 30
   - **last**: 100

   Defaults to **last**.

``interpolation`` (optional, default = **linear**)
   Interpolation method used to fill missing years in the *technodata*.
   Available interpolation methods depend on the underlying `scipy method's kind attribute`_.
   Years outside the data range will always be back/forward filled with the closest available data.

   .. _scipy method's kind attribute: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

``dispatch_production`` (optional, default = **share**)
   The method used to calculate supply of commodities after investments have been made.

   MUSE provides two methods in :py:mod:`muse.production`:

   * **share**: assets each supply a proportion of demand based on their share of total capacity.
   * **maximum**: the production is the maximum production for the existing capacity and the technology's utilization factor. See :py:func:`muse.production.maximum_production`.

   Additional methods can be registered with
   :py:func:`muse.production.register_production`

``technodata``
   Path to a csv file containing the characterization of the technologies involved in
   the sector, e.g. lifetime, capital costs, etc... See :ref:`inputs-technodata`.

``technodata_timeslices`` (optional)
    Path to a csv file describing the utilization factor and minimum service
    factor of each technology in each timeslice.
    See :ref:`user_guide/inputs/technodata_timeslices`.

``commodities_in``
   Path to a csv file describing the inputs of each technology involved in the sector.
   See :ref:`inputs-iocomms`.

``commodities_out``
   Path to a csv file describing the outputs of each technology involved in the sector.
   See :ref:`inputs-iocomms`.

``timeslice_level`` (optional)
   This represents the level of timeslice granularity over which commodity
   flows out of the sector are balanced (e.g. if "day", the sector will aim to meet
   commodity demands on a daily basis, rather than an hourly basis).
   If not given, defaults to the finest level defined in the global ``timeslices`` section.
   Note: If ``technodata_timeslices`` is used, the data in this file must match the timeslice
   level of the sector (e.g. with global timeslice levels "month", "day" and "hour", if a sector has "day" as
   the timeslice level, then ``technodata_timeslices`` must have columns "month" and "day", but not "hour")

Sectors contain a number of subsections:

``subsectors``
    Subsectors group together agents into separate groups servicing the demand for
    different commodities. There must be at least one subsector, and there can be as
    many as required. For instance, a one-subsector setup would look like:

    .. code-block:: toml

        [sectors.gas.subsectors.all]
        agents = '{path}/gas/Agents.csv'
        existing_capacity = '{path}/gas/ExistingCapacity.csv'

    A two-subsector could look like:

    .. code-block:: toml

        [sectors.gas.subsectors.methane_and_ethanol]
        agents = '{path}/gas/me_agents.csv'
        existing_capacity = '{path}/gas/me_existing.csv'
        commodities = ["methane", "ethanol"]

        [sectors.gas.subsectors.natural]
        agents = '{path}/gas/nat_agents.csv'
        existing_capacity = '{path}/gas/nat_existing.csv'
        commodities = ["refined", "crude"]

    In the case of multiple subsectors, it is important to specify disjoint sets of
    commodities so that each subsector can service a separate demand.
    The subsectors accept the following keywords:

    ``agents``
        Path to a csv file describing the agents in the sector.
        See :ref:`user_guide/inputs/agents:agents`.

    ``existing_capacity``
       Path to a csv file describing the initial capacity of the sector.
       See :ref:`user_guide/inputs/existing_capacity:existing sectoral capacity`.

    ``lpsolver`` (optional, default = **scipy**)
        The solver for linear problems to use when figuring out investments. The solvers
        are registered via :py:func:`~muse.investments.register_investment`. At time of
        writing, three are available:

        - **scipy** solver (default from v1.3): Formulates investment as a true LP problem and solves it using
          the `scipy solver`_.

        - **adhoc** solver: Simple in-house solver that ranks the technologies
          according to cost and service the demand incrementally.

    ``demand_share`` (optional, default = **standard_demand**)
        A method used to split the MCA demand into separate parts to be serviced by
        specific agents. The appropriate choice depends on the type of agents being used
        in the simulation. There are currently two options:

        - :py:func:`~muse.demand_share.standard_demand` (default): The input demand is
          split amongst *new* agents. *New* agents get a share of the increase in demand
          over the investment period, as well as the demand that occurs from decommissioned
          assets.
        - :py:func:`~muse.demand_share.new_and_retro`: The input demand is split amongst
          both *new* and *retrofit* agents. *New* agents get a share of the increase in
          demand over the investment period, whereas *retrofit* agents are assigned a share
          of the demand that occurs from decommissioned assets.

    ``constraints`` (optional, defaults to full list)
        The list of constraints to apply to the LP problem solved by the sector. By
        default all of the following are included:

        - :py:func:`~muse.constraints.demand`: a lower-bound of the production decision
          variables specifying the target demand.
        - :py:func:`~muse.constraints.max_production`: an upper bound limiting how much
          can be produced for a given capacity.
        - :py:func:`~muse.constraints.max_capacity_expansion`: an upper bound limiting
          how much the capacity can grow during each investment event.
        - :py:func:`~muse.constraints.search_space`: a binary (on-off) constraint
          specifying which technologies are considered for investment.
        - :py:func:`~muse.constraints.minimum_service`: a lower constraint for
          production for those technologies that need to keep a minimum production.
        - :py:func:`~muse.constraints.demand_limiting_capacity`: limits the combined
          capacity to be installed to the demand of the peak timeslice.


``output``
   Outputs are made up of several components. MUSE is designed to allow users to
   mix-and-match both how and what to save.

   ``output`` is specified as a TOML array, e.g. with double brackets. Each sector can
   specify an arbitrary number of outputs, simply by adding an extra output row.

   A single row looks like this:

   .. code-block:: TOML

      [[sectors.commercial.outputs]]
      filename = '{cwd}/Results/{Sector}/{Quantity}/{year}{suffix}'
      quantity = "capacity"
      sink = 'csv'
      overwrite = true

   The following attributes are available:

   ``quantity``
      Name of the quantity to save.
      The options are capacity, consumption, supply and costs.
      Users can also customize and create further output quantities by registering with MUSE via
      :py:func:`muse.outputs.register_output_quantity`. See :py:mod:`muse.outputs` for more details.

   ``sink``
      the sink is the place (disk, cloud, database, etc...) and format with which
      the computed quantity is saved. Currently only sinks that save to files are
      implemented.
      The following sinks are available: "csv", "netcfd", "excel" and "aggregate".
      Additional sinks can be added by interested users, and registered with MUSE via
      :py:func:`muse.outputs.register_output_sink`. See :py:mod:`muse.outputs` for more details.

   ``filename``
      defines the format of the file where to save the data. There are several
      standard values that are automatically substituted:

      - cwd: current working directory, where MUSE was started
      - path: directory where the TOML file resides
      - sector: name of the current sector (.e.g. "commercial" above)
      - Sector: capitalized name of the current sector
      - quantity: name of the quantity to save (as given by the quantity attribute)
      - Quantity: capitablized name of the quantity to save
      - year: current year
      - suffix: standard suffix/file extension of the sink

      Defaults to `{cwd}/{default_output_dir}/{Sector}/{Quantity}/{year}{suffix}`.

   ``overwrite`` (optional, default = False)
      If `False` MUSE will issue an error and abort, instead of
      overwriting an existing file. This prevents important output files from being overwritten.

   Additional sink parameters
      You can pass additional parameters that will be forwarded to the underlying save function.
      For example, when using the "csv" sink, you could use `float_format = "%.6f"` to increase the precision of floating point numbers in the output file (default is 4 decimal places).
      For a complete list of available parameters, see the documentation for the respective save function (e.g., `pandas.to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_ for CSV outputs).


   For example, the following would save supply data for the commercial sector as a separate file for each year:

   .. code-block:: TOML

      [[sectors.commercial.outputs]]
      quantity = "supply"
      sink = "csv"
      filename = "{cwd}/{default_output_dir}/{Sector}/{Quantity}/{year}{suffix}"
      overwrite = true

   There is a special output sink for aggregating over years (i.e. a single output file for all years). It can be invoked as
   follows:

   .. code-block:: TOML

      [[sectors.commercial.outputs]]
      quantity = "supply"
      sink = "aggregate"
      filename = "{cwd}/{default_output_dir}/{Sector}/{Quantity}.csv"

   Note that the aggregate sink always overwrites the final file, since it will overwrite itself.


``interactions`` (optional)
   Defines interactions between agents. These interactions take place right before new
   investments are computed. The interactions can be anything. They are expected to
   modify the agents and their assets. MUSE provides a default set of interactions that
   have *new* agents pass on their assets to the corresponding *retro* agent, and the
   *retro* agents pass on the make-up of their assets to the corresponding *new*
   agents.

   ``interactions`` are specified as a :ref:`TOML array<toml-array>`, e.g. with double
   brackets. Each sector can specify an arbitrary number of interactaction, simply by
   adding an extra interaction row.

   There are two orthogonal concepts to interactions:

   - a *net* defines the set of agents that interact. A set can contain any
     number of agents, whether zero, two, or all agents in a sector. See
     :py:func:`muse.interactions.register_interaction_net`.
   - an *interaction* defines how the net actually interacts.  See
     :py:func:`muse.interactions.register_agent_interaction`.

   In practice, we always consider sequences of nets (i.e. more than one net) that
   interact using the same interaction function.

   Hence, the input looks something like the following:

   .. code-block:: TOML

      [[sectors.commercial.interactions]]
      net = 'new_to_retro'
      interaction = 'transfer'

   **new_to_retro** is a function that figures out all "new/retro" pairs of agents.
   Whereas **transfer** is a function that performs the transfer of assets and
   information between each pair.

   Furthermore, it is possible to pass parameters to either the net of the interaction
   as follows:

   .. code-block:: TOML

      [[sectors.commercial.interactions]]
      net = {"name": "some_net", "param": "some value"}
      interaction = {"name": "some_interaction", "param": "some other value"}

   The parameters will depend on the net and interaction functions. Neither
   "new_to_retro" nor "transfer" take any arguments at this point. MUSE interaction
   facilities are defined in :py:mod:`muse.interactions`.

   An error is raised if and empty network is found. This is the case, for example, if a
   "new_to_retro" type of network has been defined but no retro agents are included in
   the sector.

--------------
Preset sectors
--------------

The commodity production, commodity consumption and product prices of preset sectors are determined
exogeneously. They are know from the start of the simulation and are not affected by the
simulation.

A common example would be the following, where commodity consumption is defined exogeneously:

.. code-block:: TOML

    [sectors.commercial_presets]
    type = 'presets'
    priority = 0
    consumption_path = "{path}/commercial_presets/*Consumption.csv"

Alternatively, you may define consumption as a function of macro-economic data, i.e. population and GDP:

.. code-block:: TOML

    [sectors.commercial_presets]
    type = 'presets'
    priority = 0
    timeslice_shares_path = '{path}/commercial_presets/TimesliceShareCommercial.csv'
    macrodrivers_path = '{path}/commercial_presets/Macrodrivers.csv'
    regression_path = '{path}/commercial_presets/regressionparameters.csv'

The following attributes are accepted:

``type`` (required)
   See the attribute in the standard mode, :ref:`type<sector-type>`. *Preset* sectors
   are those with type "presets".

``priority`` (required)
   See the attribute in the standard mode, :ref:`priority<sector-priority>`.

.. _preset-consumption:

``consumption_path``
   CSV files, one per year. This attribute can include wild cards, i.e. '*',
   which can match anything. For instance: `consumption_path = "{cwd}/Consumption*.csv"` will match any csv file starting with "Consumption" in the
   current working directory. The file names must include the year for which it defines
   the consumption, e.g. `Consumption2015.csv`.

   The CSV format should follow the following format:

   .. csv-table:: Consumption
      :header: "RegionName", "Timeslice", "electricity", "diesel", "algae"
      :stub-columns: 2

      USA,1,1.9,0,0
      USA,2,1.8,0,0

   The "RegionName" and "Timeslice" columns must be present.
   Further columns are reserved for commodities. "Timeslice" refers to the
   index of the timeslice. Timeslices should be defined consistently to the sectoral
   level timeslices.

``supply_path``
   CSV file, one per year, indicating the amount of commodities produced. It follows
   the same format as :ref:`consumption_path <preset-consumption>`.

.. _preset-demand:

``demand_path``
   Incompatible with :ref:`consumption_path<preset-consumption>` or
   :ref:`macrodrivers_path<preset-macro>`. A CSV file containing the consumption in the
   same format as :ref:`inputs-projection`.

.. _preset-macro:

``macrodrivers_path``
   Incompatible with :ref:`consumption_path<preset-consumption>` or
   :ref:`demand_path<preset-demand>`. Path to a CSV file giving the profile of the
   macrodrivers. Also requires :ref:`regression_path<preset-regression>`.

.. _preset-regression:

``regression_path``
   Incompatible with :ref:`consumption_path<preset-consumption>`.
   Path to a CSV file giving the regression
   parameters with respect to the macrodrivers.
   Also requires :ref:`macrodrivers_path<preset-macro>`.

``timeslice_shares_path``
   Incompatible with :ref:`consumption_path<preset-consumption>` or
   :ref:`demand_path<preset-demand>`. Optional csv file giving shares per timeslice.
   The timeslice share definition needs to have a consistent number of timeslices as the
   sectoral level time slices.
   Requires
   :ref:`macrodrivers_path<preset-consumption>`.


-------------
Carbon market (optional)
-------------

This section contains the settings related to the modelling of the carbon market.
If omitted, it defaults to not including the carbon market in the simulation.

For example

.. code-block:: TOML

   [carbon_budget_control]
   budget = [1000, 800, 600, 400, 200, 0]
   commodities = ["CO2"]

``budget``
   Yearly budget. There should be one item for each year the simulation will run. In
   other words, if given and not empty, this is a list with the same length as
   `time_framework` from the main section. If not given or an empty list, then the
   carbon market feature is disabled. Defaults to an empty list.

``commodities``
   Commodities that make up the carbon market.

``control_undershoot`` (optional, default = False)
   Whether to control carbon budget undershoots. This parameter allows for carbon tax credit from one year to be passed to the next in the case of less carbon being emitted than the budget.

``control_overshoot`` (optional, default = False)
   Whether to control carbon budget overshoots. If the amount of carbon emitted is above the carbon budget, this parameter specifies whether this deficit is carried over to the next year.

``method`` (optional, default = **bisection**)
   Method used to equilibrate the carbon market. Available options are **fitting** and **bisection**, however this can be expanded with the `@register_carbon_budget_method` hook in `muse.carbon_budget`.

   These methods solve the market with a number of different carbon prices, aiming to find the carbon price at which emissions (pooled across all regions) are equal to the carbon budget.
   The obtained carbon price applies to all regions.

   The **fitting** method samples a number of different carbon prices to build a regression model (linear or exponential) of emissions as a function of carbon price.
   This regression model is then used to estimate the carbon price at which the carbon budget is met.

   The **bisection** method uses an iterative approach to settle on a carbon price.
   Starting with a lower and upper-bound carbon price, it iteratively halves this price interval until the carbon budget is met to within a user-defined tolerance, or until the maximum number of iterations is reached.
   Generally, this method is more robust for markets with a complex, nonlinear relationship between emissions and carbon price, but may be slower to converge than the `fitting` method.


``method_options``
   Additional options for the specified carbon method.

   Parameters for the **bisection** method:

   - ``max_iterations`` (optional, default = 5): maximum number of iterations.
   - ``tolerance`` (optional, default = 0.1): tolerance for convergence. E.g. 0.1 means that the algorithm will terminate when emissions are within 10% of the carbon budget.
   - ``early_termination_count`` (optional, default = 5): number of iterations with no change in the carbon price before the algorithm will terminate.
   - ``price_penalty`` (optional, default = 0.1): penalty factor applied to carbon price when selecting optimal solution when convergence isn't reached. For example, if the carbon price is measured in units of MUSD/kt, a price penalty of 1000 means that a price increase of 1 MUSD/kt will only be accepted if it reduces emissions by at least 1000 kt.

   Parameters for the **fitting** method:

   - ``fitter`` (optional, default = **linear**): the regression model used to approximate model emissions. Predefined options are **linear** (default) and **exponential**. Further options can be defined using the `@register_carbon_budget_fitter` hook in `muse.carbon_budget`.
   - ``sample_size`` (optional, default = 5): number of price samples used.

   Shared parameters:

   - ``refine_price`` (optional, default = False): If True, applies an upper limit on the carbon price.
   - ``price_too_high_threshold`` (optional, default = 10): upper limit on the carbon price.
   - ``resolution`` (optional, default = 2): Number of decimal places to solve the carbon price to. When using the bisection method, increasing this value may increase the time taken to solve the carbon market.



-------------
Output cache (for advanced users)
-------------

``outputs_cache``
   This option behaves exactly like `outputs` for sectors and accepts the same options but
   controls the output of cached quantities instead. This option is NOT available for
   sectors themselves (i.e using `[[sector.commercial.outputs_cache]]` will have no effect). See
   :py:mod:`muse.outputs.cache` for more details.

   A single row looks like this:

   .. code-block:: TOML

      [[outputs_cache]]
      quantity = "production"
      sink = "aggregate"
      filename = "{cwd}/{default_output_dir}/Cache{Quantity}.csv"
