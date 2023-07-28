
.. contents::

Installing MUSE
===============

Recommended installation
------------------------

To help you installing MUSE in your system we will follow these steps:

- `Launching a terminal`_: Needed to both install and run MUSE.
- `Installing a compatible Python version`_: At the moment, MUSE works with Python 3.8 and 3.9.
- `Installing pipx`_: A Python application manager that facilitates installing, keeping applications updated and run them in their own isolated environments.
- `Installing MUSE itself`_

In the following sections, we will guide you step by step in configuring your system so it can run MUSE. At the end, we include alternative ways of installing MUSE if this method does not work for any reason, for advanced users and for developers of MUSE.

.. note::

    The next sections will explain in detail the appropriate steps as well as commenting on possible caveats. **We strongly encourage you to read through the sections below** to understand what these steps entitle, but in the end, what we are going to do to install MUSE is the following:

    - Open a terminal
    - Install `pyenv <https://github.com/pyenv/pyenv>`_ (Linux and MacOS) or `pyenv-win <https://pyenv-win.github.io/pyenv-win/>`_ (Windows) and make sure it works.
    - Run the following commands in the terminal:

        .. code-block::

            pyenv install 3.9.13
            pyenv shell 3.9.13
            python -m pip install pipx
            python -m pipx ensurepath
            python -m pipx install muse-os

    - After this, MUSE will be available to use system wide simply by invoking ``muse`` in the terminal, for example ``muse --model default``.



Launching a terminal
~~~~~~~~~~~~~~~~~~~~

All operative systems have a Terminal application that let you run commands. You will need to use it extensively when using MUSE, so we strongly suggest you get familiar with it. For now, let's just figure out how to launch it:

- **Linux**: Depending on the distribution, you might have a shortcut in your tasks bar already or it should be easily found in the menu. Look for ``Console`` or ``Terminal`` to lunch the application.
- **MacOS**: Press ``Super key + Space`` to open the search box. There, type ``Terminal`` and press ``Enter``.
- **Windows**: Windows comes with a couple of options. We will be using ``Windows PowerShell``. Press the ``Windows key`` and start typing ``PowerShell``. When the application shows up, click to on it.

    .. image:: figures/launch_power_shell.png
       :width: 500
       :align: center
       :alt: Launching Windows PowerShell from the menu

Once you have launched the Terminal, the window that opens will show the command prompt, where we will input all the commands form now on. The following are a couple of examples of what it looks like, typically, but it might be a bit different in your system depending on how it is configured:

- Linux and MacOS Terminal:

.. code-block:: bash

    your_user_name@computer_name:~$

- Windows PowerShell:

.. code-block:: powershell

    PS C:\Users\your_user_name>

.. note::

    For simplicity, we will be excluding the command prompt in the following sections whenever we indicate that a command should be executed in the terminal.

Installing a compatible Python version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MUSE needs Python to run but, for now, it only works with versions 3.8 and 3.9, so the next step is to install a suitable version of Python.

.. note::

    Windows users should disable the alias for Python that comes by default with Windows and that will try to install Python from the Microsoft Store everytime we write ``python`` in the terminal. To do so, press the ``Windows key`` and start typing ``alias``, when it shows up, click in ``Manage app execution aliases``. In the window that opens, disable all the entries related to Python, like in the image.

    .. image:: figures/disable_python_alias.png
        :width: 400
        :align: center
        :alt: Screen to disable the Python aliases defined by Windows.

The first thing will be to check if you already have a suitable python version installed. Open a terminal and run:

.. code-block:: bash

    python --version

If the output is ``Python 3.8.X`` or ``Python 3.9.X``, where ``X`` is any number, then you have a version of Python compatible with MUSE and you can skip this section altogether. Move to `Installing pipx`_. In any other case, keep reading.

There are multiple ways of installing Python, as well as multiple distributions. Here we have opted for the one that we believe is simplest, requires the smallest downloads and gives the maximum flexibility: using ``pyenv``.

.. note::

    If you have Anaconda Python installed, then you can use it instead of ``pyenv`` to create an environment with a suitable Python version. Go to section `Creating a conda virtual environment`_ and jump to `Installing pipx`_ when it is completed.

Installing ``pyenv``
^^^^^^^^^^^^^^^^^^^^

`pyenv <https://github.com/pyenv/pyenv>`_ (`pyenv-win <https://pyenv-win.github.io/pyenv-win/>`_ for Windows) is a tool that lets you install and manage different python versions. It is small, unobtrusive and self-contained, and it is available for the three operative systems.

To install it, follow these steps:

- **Linux**: In this case, you will need to clone the GitHub repository using ``git``. Most Linux distributions come with ``git`` installed, so this should work out of the box:

    .. code-block:: bash

        git clone https://github.com/pyenv/pyenv.git ~/.pyenv

    Then, complete the setup by adding ``pyenv`` to your profile, so the executable can be found. `Check the instructions in the official webpage <https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv>`_.

- **MacOS**: The simplest option is to use Homebrew:

    .. code-block:: bash

        brew update
        brew install pyenv

    Then, complete the setup by adding ``pyenv`` to your profile, so the executable can be found. `Check the instructions in the official webpage <https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv>`_.

- **Windows**: ``pyenv-win`` is a separate project but it has the same functionality and it is also simpler to setup. Just run the following command and you should be ready to go:

    .. code-block:: powershell

        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

    .. note::

        If you are getting any ``UnauthorizedAccess`` error, then start Windows PowerShell with the “Run as administrator” option (see figure above) and run:

        .. code-block:: powershell

            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine

        Finally open a normal PowerShell and re-run the above installation command.

After completing the above steps, you will need to close the terminal and re-open it again. After that, to check if things work run:

.. code-block:: bash

    pyenv --version

You should get something similar to:

.. code-block:: output

    pyenv 3.1.1

Actually installing Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``pyenv`` installed and correctly configured, it is now easy to install any Python version we want. To see the versions available run:

.. code-block:: bash

    pyenv install -l

You should see a very long list of versions to choose from. Let's install the latest version of the 3.9 family:

.. code-block:: bash

    pyenv install 3.9.13

The command will take a minute or two to complete, depending on your internet connection, and show an output similar to the following (this is just an example for Windows):

.. code-block:: output

    :: [Info] ::  Mirror: https://www.python.org/ftp/python
    :: [Downloading] ::  3.9.13 ...
    :: [Downloading] ::  From https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe
    :: [Downloading] ::  To   C:\Users\your_username\.pyenv\pyenv-win\install_cache\python-3.9.13-amd64.exe
    :: [Installing] ::  3.9.13 ...
    :: [Info] :: completed! 3.9.13

Now, we have a new Python version in our system, but it is still not available (if you run ``python --version`` you will get the same result as before). There are two options moving forward:

- If you want to set it as the global python version, available system wide (only do this if you really want to set is as your main Python!) run:

    .. code-block:: bash

        pyenv global 3.9.13

- If you just want it momentarily to install MUSE run instead the following command:

    .. code-block:: bash

        pyenv shell 3.9.13

In both cases, if you run ``python --version`` afterwards, you should get ``Python 3.9.13``.

Installing ``pipx``
~~~~~~~~~~~~~~~~~~~

Next we need to install ``pipx``, a Python application manager that facilitates installing, keeping applications updated and run them in their own isolated environments. We could skip this step and install MUSE directly, but that will risk to have conflicting dependencies in the future if you install any other application, breaking your MUSE installation, and we do not want that to happen.

The installation instructions for ``pipx`` can be found in the `official webpage <https://pypa.github.io/pipx/installation/>`_ specific for the three operative systems. The following instructions, however, should work for the three cases:

.. code-block:: bash

    python -m pip install pipx
    python -m pipx ensurepath

Make sure you run these commands with a compatible Python version, as described in the previous section. If for whatever reason, this does not work, follow the system specific instructions in the webpage.

Installing MUSE itself
~~~~~~~~~~~~~~~~~~~~~~

With all the system prepared, installing MUSE is the easiest part:

.. code-block:: bash

    python -m pipx install muse-os

As above, make sure you run this command with the appropriate Python version.

And that is all! Now, MUSE should be available system wide simply by running ``muse`` in the terminal. For example, open a new terminal and run:

.. code-block:: bash

    muse --model default

This will run a default, example model, completing after reaching year 2050. The following are the last few lines of the simulation:

.. code-block::

    ...
    -- 2023-07-20 13:45:25 - muse.demand_share - INFO
    Computing demand_share: default

    -- 2023-07-20 13:45:25 - muse.production - INFO
    Computing production: share

    -- 2023-07-20 13:45:25 - muse.mca - WARNING
    Check growth constraints for wind.

    -- 2023-07-20 13:45:25 - muse.mca - INFO
    Finish simulation year 2050!


Alternative installation instructions
-------------------------------------

If you don't want to use ``pyenv`` or ``pipx``, or if you are having trouble with those tools, there are a couple of alternatives.

Installing Anaconda Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

We have chosen ``pyenv`` above because it is extremely lightweight and unobtrusive with your operative system. However, you might want to consider a more fully fledged Python distribution like Anaconda, specially if your work involved non-python packages or a lot of data science and machine learning tools.

Regardless of the reason, if you want to follow this route just go to the official `Anaconda webpage <https://www.anaconda.com/>`_ and download and install a version appropriate for your operative system. Do not worry about the Python version as ``conda`` will let you choose that when creating a virtual environment.

The installer should guide you step by step on the process of installing Anaconda and configuring your system to use it as your Python installation.

Creating virtual environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although not strictly necessary, **creating a virtual environment is highly recommended** regardless of how you installed Python. It will isolate users and developers from changes occurring on their operating system, and from conflicts between python packages and it ensures reproducibility from day to day.

Using ``pipx`` ensures that each application it installs has its own virtual environment, running it under the hood. However, you can explicitly create and manage the virtual environment if you prefer.

Creating a ``conda`` virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is available only if you installed Anaconda Python. Depending on the settings you used when installing Anaconda and your operative system, you might have ``conda`` available in your normal terminal or you might need to use the Anaconda Prompt.

``conda`` not only lets you create a virtual environment but also selecting which python version to use within, independently of the version of Anaconda Python installed, which means it can be an alternative to ``pyenv`` if it happens that you already have Anaconda installed in your system.

To create an environment called ``muse_env`` run:

.. code-block:: bash

    conda create -n muse_env python=3.9

Now, you can activate the environment with:

.. code-block:: bash

    conda activate muse_env

Later, to recover the system-wide "normal" python, deactivate the environment with:

.. code-block:: bash

    conda deactivate

Creating a virtual environment with ``venv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modern Python versions, regardless of their origin, come with a built in tool to create virtual environments, ``venv``. However, contrary to ``conda`` it does not let you select the version of Python that will be used - it will be the same one you are using to create the environment. Therefore, you still need to make sure your version of Python is compatible with MUSE. You can check it with ``python --version``.

Another caveat is that the virtual environment will be created in a specific folder, so whenever you want to use it in the future, you will need to remember in what folder it was created and activate the environment from there.

You can create a virtual environment with:

.. code-block:: bash

    python -m venv venv

And then you activate the environment with:

- Linux:

    .. code-block:: bash

        source venv/bin/activate

- MacOS:

    .. code-block:: zsh

        . venv/bin/activate

- Windows:

    .. code-block:: powershell

        venv\Scripts\Activate.ps1

Later, to recover the system-wide "normal" python, deactivate the environment with:

.. code-block:: bash

    deactivate

Installing MUSE in a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the method used, **once it has been created and activated**, you can install ``MUSE`` within using:

.. code-block:: bash

    python -m pip install muse-os

And then use it by invoking ``muse`` with the relevant input arguments. Keep in mind that, contrary to using ``pipx``, in this case **you will need to manually activate the environment every time you want to use MUSE**.

Instructions for development
----------------------------

Developers of MUSE will need to have the version control tool ``git`` installed in their system and be familiar with its usage. The `Introduction to Git and GitHub for software development <https://imperialcollegelondon.github.io/introductory_grad_school_git_course/>`_ course created by `Imperial RSE Team <https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/service-offering/research-software-engineering/>`_ can be a good place to start.

Installing MUSE source code in editable mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have ``git`` in your system, clone MUSE repository with:

.. code-block:: bash

    git clone https://github.com/SGIModel/MUSE_OS.git

Then, we will create a virtual environment, either using ``conda`` or using ``venv`` as explained above, and install MUSE within the environment:

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

This command will use ``pandoc`` under the hood, which might not be available in your system. If that were the case, install it `following the instructions in the official webpage <https://pandoc.org/installing.html>`_.

The main page for the documentation can then be found at ``docs/build/html/index.html`` and the file can viewed from any web browser.

Configuring VSCode
~~~~~~~~~~~~~~~~~~

`VSCode <https://code.visualstudio.com/>`_ users will find that the repository is setup with default settings file.  Users will still need to `choose the virtual environment <https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment>`_, or conda environment where to run the code. This will change the ``.vscode/settings.json`` file and add a user-specific path to it. Users should try and avoid committing changes to ``.vscode/settings.json`` indiscriminately.
