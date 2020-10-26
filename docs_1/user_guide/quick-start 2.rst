===========================
Running the demo simulation
===========================

A simulation can be created from scratch, defining sector and input data,
or using either the standard or the example models as a reference.

Once a model has been created, the simulation can be started in a
standard way with the command "python -m muse" followed by the path
to the input toml file, as shown below.

.. code-block:: Bash

    python -m muse settings.toml

A description of the input file can be found in :ref:`input-files`.

There are a few standard models available directly with MUSE:

- the default model with 3 sectors (residential, power, gas supply)
  and 1 agent is available in the folder src/muse/data/model
- a multi-agent model with 1 sector (residential) and two agents
  is available in the folder folder src/muse/data/model/example

The standard model can run as follows:

.. code-block:: Bash

    python -m muse --model default

The output will be located in a sub-folder ``Results`` of the folder from which the
command is run. There may be other models available, other than default. To list them,
do:

.. code-block:: Bash

    python -m muse --help

The standard model can be copied to a path of your choosing with

.. code-block:: Bash

    python -m muse --model default --copy XXX

Once a model has been copied to folder, run it with:

.. code-block:: Bash

    python -m muse XXX/settings.toml

The additional models available in the "example" subfolder can be instanciated and run as
follows:

.. code-block:: Python

    from muse import examples
    model = examples.model("default")
    model.run()
