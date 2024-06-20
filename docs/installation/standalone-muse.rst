.. _standalone-muse:

Standalone MUSE
---------------

MUSE is available as a standalone application for Windows that bundles together Python itself and all its dependencies. There is nothing to install upfront, just downloading the bundle and putting it in a known location in your hard drive, so you can access the executables. This approach has the advantage that it comes with no extra effort of configuring your system with a Python version compatible with MUSE (see :ref:`pipx-based`), but it gives you the least flexibility when new versions of MUSE are release, as you need to download the whole bundle again and manually replace the previous one.

To get and install the standalone version of MUSE simply:

- Go to the `releases page in GitHub <https://github.com/EnergySystemsModellingLab/MUSE_OS/releases>`_ and download the standalone bundle for the release you are interested. Note that **there are bundles only from MUSE v1.1.0 onwards**, and it might be that not all releases have a bundle.
- Unzip it to a known location in your drive, eg. ``Documents/Programs/MUSE``

Inside the resulting directory you will see many files and documents related to Python and MUSE dependencies. Look for ``muse.exe`` and ``muse_gui.exe``, which are the two executables of interest, one to run MUSE in the terminal (and therefore only makes sense to execute from the terminal!) and another to run the graphical user interface (GUI) of MUSE. Just double click in ``muse_gui.exe`` and you should see the main MUSE GUI, like the following:

    .. image:: ../figures/muse_gui.png
       :width: 500
       :align: center
       :alt: Window of the GUI version of MUSE


Access MUSE from the desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the above works totally fine and you do not need anything else, navigating to the appropriate folder to run MUSE every time can be tiresome. To avoid that, you could:

- Create a shortcut of ``muse_gui.exe`` in the desktop.
- Add MUSE to the taskbar.
- Pin MUSE to the Start menu.

You can do all of the above by doing right click in ``muse_gui.exe`` and choosing the option you prefer.

    .. image:: ../figures/pin_muse_gui.png
       :width: 500
       :align: center
       :alt: Options to pin MUSE_GUI to the taskbar, Start menu or create a shortcut

Running MUSE from the terminal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run ``muse.exe`` from the terminal (see the section on :ref:`launch-terminal`), you will need first to navigate (in the terminal) to the folder where you unzip the bundle, which is certainly inconvenient. To avoid having to do that, you can add that folder to the PATH, so it is accessible anywhere in your system. To do that, follow these steps:

1. Launch the "Edit the system environment variables" tool. You can search for it in the Start menu.

    .. image:: ../figures/add_MUSE_to_path_1.png
       :width: 500
       :align: center
       :alt: Search for the environment variables editor in the Start menu

2. Click on the "Environment variables" button

    .. image:: ../figures/add_MUSE_to_path_2.png
       :width: 500
       :align: center
       :alt: Advanced system properties window

3. Select the variable "path" from the top list and click in "Edit"

    .. image:: ../figures/add_MUSE_to_path_3.png
       :width: 500
       :align: center
       :alt: Environment variables window

4. Click in "Browse" and in the dialog that opens select the directory containing ``muse.exe`` and ``muse_gui.exe``.

    .. image:: ../figures/add_MUSE_to_path_4.png
       :width: 500
       :align: center
       :alt: Window showing the value of a specific environment variable

After following this steps you should be able to open a **new** terminal and run MUSE from anywhere in your system using ``muse.exe``, eg.:

.. code-block:: powershell

    muse.exe --model default
