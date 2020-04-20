===========================
Running the demo simulation
===========================

All the input files for the demonstration simulation can be found in **[INSERT LINK HERE]**. Click
on the link and expand the zip file. Assuming the installation process worked, then
running a simulation should be as simple as opening an anaconda prompt and navigating to
the input files. For instance, if the files were unzipped on the desktop:

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

The models are also available directly in python. They can be instanciated and run as
follows:

.. code-block:: Python

    from muse import examples
    model = examples.model("default")
    model.run()
