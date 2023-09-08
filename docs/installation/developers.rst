Instructions for development
----------------------------

Developers of MUSE will need to have the version control tool ``git`` installed in their system and be familiar with its usage. The `Introduction to Git and GitHub for software development <https://imperialcollegelondon.github.io/introductory_grad_school_git_course/>`_ course created by `Imperial RSE Team <https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/service-offering/research-software-engineering/>`_ can be a good place to start.

Installing MUSE source code in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have ``git`` in your system, clone the MUSE repository with:

.. code-block:: bash

    git clone https://github.com/SGIModel/MUSE_OS.git

Then, we will create a virtual environment, either using ``conda`` or using ``venv`` as explained in `virtual-environments`_, and install MUSE within the environment:

.. code-block:: bash

    cd MUSE_OS
    # 1- Create virtual environment
    # 2- Activate virtual environment
    # 3- Finally, install MUSE in editable mode with:
    python -m pip install -e .[dev,doc]

Depending on your system, you might need to add quotation marks around ``[dev,doc]`` as in ``"[dev,doc]"``. This will install MUSE and all the dependencies required for its development. The downloaded code can be modified and the changes will be automatically reflected in the environment.

Installing pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure the consistency of the code with other developers, install the pre-commit hooks, which will run a series of checks whenever there is a new commit:

.. code-block:: bash

    python -m pip install pre-commit
    pre-commit install

Running tests
~~~~~~~~~~~~~

In the developing phase, MUSE can also be used to run test cases to check that the model would reproduce expected results from a defined set of input data. Tests can be run with the command [pytest](https://docs.pytest.org/en/latest/), from the testing framework of the same name.

Within the ``MUSE-OS`` directory, just run:

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
