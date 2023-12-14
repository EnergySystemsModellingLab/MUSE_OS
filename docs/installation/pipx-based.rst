.. _pipx-based:

pipx-based installation
-----------------------

To help you installing MUSE in your system we will follow these steps:

- `Launching a terminal`_: Needed to both install and run MUSE.
- `Installing a compatible Python version`_: At the moment, MUSE works with Python 3.8 and 3.9.
- `Installing pipx`_: A Python application manager that facilitates installing, keeping applications updated and run them in their own isolated environments.
- `Installing MUSE itself`_

In the following sections, we will guide you step by step in configuring your system so it can run MUSE. At the end, we include alternative ways of installing MUSE if this method does not work for any reason and for advanced users.

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

.. _launch-terminal:

Launching a terminal
~~~~~~~~~~~~~~~~~~~~

All operative systems have a Terminal application that let you run commands. You will need to use it extensively when using MUSE, so we strongly suggest you get familiar with it. For now, let's just figure out how to launch it:

- **Linux**: Depending on the distribution, you might have a shortcut in your tasks bar already or it should be easily found in the menu. Look for ``Console`` or ``Terminal`` to lunch the application.
- **MacOS**: Press ``Super key + Space`` to open the search box. There, type ``Terminal`` and press ``Enter``.
- **Windows**: Windows comes with a couple of options. We will be using ``Windows PowerShell``. Press the ``Windows key`` and start typing ``PowerShell``. When the application shows up, click to on it.

    .. image:: ../figures/launch_power_shell.png
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

    .. image:: ../figures/disable_python_alias.png
        :width: 400
        :align: center
        :alt: Screen to disable the Python aliases defined by Windows.

.. note::

    If you already have a Python version installed from the Microsoft Store, you might have a ``py`` command that will launch Python in the terminal. That Python Launcher will use a Microsoft Stored-installed version of Python. Unless any of those versions, under the control of Microsoft and their autoupdating schedule, happen to be compatible with MUSE, we advise against using the launcher and follow the instructions below, which will give you more control on what is exactly being run and how MUSE is being installed.

The first thing will be to check if you already have a suitable python version installed. Open a terminal and run:

.. code-block:: bash

    python --version

If the output is ``Python 3.8.X`` or ``Python 3.9.X``, where ``X`` is any number, then **you have a version of Python compatible with MUSE and you can skip this section altogether**. Move to `Installing pipx`_. In any other case, keep reading.

There are multiple ways of installing Python, as well as multiple distributions. Here we have opted for the one that we believe is simplest, requires the smallest downloads and gives the maximum flexibility: using ``pyenv``.

.. note::

    If you have Anaconda Python installed, then you can use it instead of ``pyenv`` to create an environment with a suitable Python version. Go to section :ref:`conda-venvs` and jump to `Installing pipx`_ when it is completed.

Installing ``pyenv``
^^^^^^^^^^^^^^^^^^^^

`pyenv <https://github.com/pyenv/pyenv>`_ (`pyenv-win <https://pyenv-win.github.io/pyenv-win/>`_ for Windows) is a tool that lets you install and manage different python versions. It is small, unobtrusive and self-contained, and it is available for the three operative systems. However, you might want to consider a more fully fledged Python distribution like Anaconda, specially if your work involved non-python packages or a lot of data science and machine learning tools. If that is the case, go to the :ref:`virtual-env-based` section.

To install ``pyenv``, follow these steps:

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
    -- 2023-08-02 09:11:50 - muse.sectors.sector - INFO
    Running gas for year 2050

    -- 2023-08-02 09:11:50 - muse.interactions - INFO
    Net new_to_retro of 1 interactions interacting via transfer

    -- 2023-08-02 09:11:50 - muse.hooks - INFO
    Computing initial_asset_transform: default

    -- 2023-08-02 09:11:50 - muse.hooks - INFO
    Computing initial_asset_transform: clean

    -- 2023-08-02 09:11:50 - muse.demand_share - INFO
    Computing demand_share: default

    -- 2023-08-02 09:11:51 - muse.production - INFO
    Computing production: max

    -- 2023-08-02 09:11:51 - muse.production - INFO
    Computing production: max

    -- 2023-08-02 09:11:51 - muse.production - INFO
    Computing production: share

    -- 2023-08-02 09:11:51 - muse.mca - INFO
    Finish simulation year 2050!
