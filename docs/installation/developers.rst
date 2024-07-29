.. _developers:

Installation for developers
---------------------------

Developers of MUSE will need to have the version control tool ``git`` installed in their system and be familiar with its usage.
The `Introduction to Git and GitHub for software development <https://imperialcollegelondon.github.io/introductory_grad_school_git_course/>`_ and `Further Git and GitHub for effective collaboration <https://imperialcollegelondon.github.io/intermediate_grad_school_git_course/index.html/>`_ courses by the `Imperial RSE Team <https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/service-offering/research-software-engineering/>`_ are excellent starting points.


Installing MUSE source code in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have ``git`` in your system, clone the MUSE repository with:

.. code-block:: bash

    git clone https://github.com/EnergySystemsModellingLab/MUSE_OS.git

For developing MUSE, **it is highly recommended to do so in a virtual Python environment**. This is to isolate dependencies and to avoid version conflicts on other Python libraries.
Moreover, this allows for unconstrained experimentation with the code without breaking system dependencies, which, as least as far as ``Linux`` is concerned, is an issue.

Also, any operating system allows for multiple Python versions being installed in parallel for different purposes like creating new features and bug fixing.
You can either use ``conda`` or ``pyenv`` in conjunction with ``venv`` as explained in :ref:`virtual-environments`.

Once you have your environment created, **activate it** and install MUSE within the environment:

.. code-block:: bash

    cd MUSE_OS
    # 1 - Create a virtual environment
    # 2 - Activate that virtual environment
    # 3 - Install MUSE in editable mode with: python -m pip install -e .[dev,doc]

.. note::

    Depending on your system, you might need to add quotation marks around ``[dev,doc]`` as in ``"[dev,doc]"``.

    For example on ``Windows``, the command will read `python -m pip install -e .[dev,doc]`. On Ubuntu Linux, it will be `python -m pip install -e ."[dev,doc]"`.
    This will install MUSE including its dependencies for development. The downloaded code can be modified and the changes will be automatically reflected in the virtual environment.

.. note::

    The source-code installation will only be accessible whilst the virtual environment is active, and can be called from the command line using the ``muse`` command with the relevant input arguments.

    If you have also installed MUSE globally using ``pipx``, this can still be used outside the virtual environment (using the same ``muse`` command), but the source-code installation will take precedence when the virtual environment is active.

Installing pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure the consistency of the code with other developers, install the pre-commit hooks, which will run a series of checks whenever there is a new commit:

.. code-block:: bash

    python -m pip install pre-commit
    pre-commit install

Running tests
~~~~~~~~~~~~~

If you followed the **Installation for developers** guide, you are likely developing your own tests for MUSE.
These tests ensure that a model produces reproducible results for a defined set of input data.
Tests can be run with the command [pytest](https://docs.pytest.org/en/latest/), using the testing framework of the same name.

To run tests, within the ``MUSE-OS`` directory, activate the virtual environment where you installed ``MUSE`` and run:

.. code-block:: bash

    python -m pytest

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation can be built with Sphinx:

.. code-block:: bash

    python -m sphinx -b html docs docs/build

This command will use ``pandoc`` under the hood, which might not be available in your system. If that were the case, install it `following the instructions in the official webpage <https://pandoc.org/installing.html>`_. It will also build the graphs and flow diagrams with ``graphviz``, which also needs to be installed separately from the `official webpage <https://graphviz.org/download/>`_.

The main page for the documentation can then be found at ``docs/build/html/index.html`` and the file can viewed from any web browser.

Create the standalone version of MUSE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use `pyinstaller <https://pyinstaller.org/en/stable/>`_ to create a standalone version of MUSE, a version that bundles together MUSE and all its dependencies (including Python) in a way that can be easily distributed and used in any compatible system without having to install anything. In :ref:`standalone-muse` we describe how to get and use this standalone version.

If you want to create such a version yourself during the development process, just run:

.. code-block:: bash

    pyinstaller muse_dir.spec

This will start the (potentially long) process of collecting all the dependencies and MUSE itself and put them into a ``dist`` sub-folder, in binary form.

Configuring VSCode
~~~~~~~~~~~~~~~~~~

`VSCode <https://code.visualstudio.com/>`_ users will find that the repository is setup with default settings file.  Users will still need to `choose the virtual environment <https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment>`_, or conda environment where to run the code. This will change the ``.vscode/settings.json`` file and add a user-specific path to it. Users should try and avoid committing changes to ``.vscode/settings.json`` indiscriminately.
