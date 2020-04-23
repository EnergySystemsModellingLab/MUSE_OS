===========================
Running the demo simulation
===========================

Assuming you have an existing model in the current folder (if not, keep reading), then
the MUSE simulation can be started with:

.. code-block:: Bash

    python -m muse settings.toml

A description of the input file can be found in :ref:`input-files`.

There are a few standard models available directly with MUSE. They can be run as
follows:

.. code-block:: Bash

    python -m muse --model default

The output will be located in a sub-folder ``Results`` of the folder from which the
command is run. There may be other models available, other than default. To list them,
do:

.. code-block:: Bash

    python -m muse --help

The full model can be copied to a path of your choosing with

.. code-block:: Bash

    python -m muse --model default --copy XXX

Once a model has been copied to folder, run it with:

.. code-block:: Bash

    python -m muse XXX/settings.toml

The models are also available directly in python. They can be instanciated and run as
follows:

.. code-block:: Python

    from muse import examples
    model = examples.model("default")
    model.run()
