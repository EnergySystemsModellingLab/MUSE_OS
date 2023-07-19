Installation
============

There are two ways to install MUSE: one for users who do not wish to modify the source code of MUSE, and another for developers who do.

.. note::

   Windows users and developers may need to install Windows Build Tools. 
   To download any Visual Studio product, users should first log in with their Visual Studio account. 
   Either a Visual Studio Subscription, or a free account can be used by clicking on “Create a new Microsoft account” on the login page.
   
   After login on the Visual Studio account, https://visualstudio.microsoft.com/downloads/ the VisualStudioSetup.exe could be downloaded, selecting the latest

   These tools include C/C++ compilers which are needed to build some python dependencies.
   
   MacOS includes compilers by default, hence no action is needed for Mac users.
   
   Linux users may need to install a C compiler, whether GNU gcc or Clang, as well python development packages, depending on their distribution.

   #. Install latest Visual Studio from the following link: <https://visualstudio.microsoft.com/downloads/>__
   
   #. Select the “Visual Studio Community” version. Click on “Download” and save the executable vs_Commmunity.exe.
  
   #. Install Visual Studio by selecting the default options. You may be asked to reboot your computer to complete the installation.

   #. Download the Microsoft Visual C++ Build Tools from the following link: <https://visualstudio.microsoft.com/downloads/>__.

   #. Please select the “Build Tools for Visual Studio 2019 (version 16.9)”. Click on download. Save the vs_BuildTools.exe.

   #. Run the installer

   #. Select: Workloads → Desktop development with C++.

   #. Install options: select only the “Windows 10 SDK” (assuming the computer is Windows 10)]. This will come up on the right hand side of the screen.


   Earlier versions of Visual Studio BuildTools than the latest can be downloaded at <https://visualstudio.microsoft.com/vs/older-downloads/>__ selecting the desired version of the VisualStudio installer and the corresponding VuildTools version.
   
   For the 2019 version, the installation screen should look similar to the following:

   .. image:: figures/visual-studio-installation.png

   For further information, see this link: <https://www.scivision.dev/python-windows-visual-c-14-required>__



   .. __: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

For users
---------

MUSE is developed using python, an open-source programming language, which means that there are two steps to the installation process. First, python should be installed. Then so should MUSE.

The simplest method to install python is by downloading the `Anaconda distribution <https://docs.anaconda.com/free/anaconda/index.html>`_. Make sure to choose the appropriate operating system (e.g. windows), python version 3.8, and the 64 bit installer. Once this has been done follow the steps for the anaconda installer, as prompted.

After python is installed we can install MUSE. MUSE can be installed via the anaconda prompt (or any terminal on Mac and Linux). This is a command-line interface to python and the python eco-system. In the anaconda prompt, run:

.. code-block:: bash

   python -m pip install --user git+https://github.com/SGIModel/MUSE_OS

It should now be possible to run muse. Again, this can be done in the anaconda prompt as follows:

.. code-block:: bash

   python -m muse --help

.. note::

   Although not strictly necessary, users are encouraged to create an `Anaconda virtual environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ and install MUSE there, as shown in :ref:`installation-devs`.

.. _installation-devs:


For developers
--------------

Although not strictly necessary, creating an `Anaconda virtual environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ is highly
recommended. Anaconda will isolate users and developers from changes occuring on their
operating system, and from conflicts between python packages. It also ensures reproducibility
from day to day.

Create a virtual env including python with:

.. code-block:: bash

   conda create -n muse python=3.8

Activate the environment with:

.. code-block:: bash

   conda activate muse

Later, to recover the system-wide "normal" python, deactivate the environment with:

.. code-block:: bash

   conda deactivate

The simplest approach is to first download the muse code with `git`_:

.. code-block:: bash

   git clone https://github.com/SGIModel/MUSE_OS.git muse

For interested users, there are plenty of `good`__ tutorials for `git`_. 
Next, it is possible to install the working directory into the conda environment:

.. code-block:: bash

   # On Linux and Mac
   cd muse
   conda activate muse
   python -m pip install -e ".[dev,docs]"

   # On Windows
   dir muse
   conda activate muse
   python -m pip install -e ".[dev,docs]"

The quotation marks are needed on some systems or shells, and do not hurt on any. The
downloaded code can then be modified. The changes will be automatically reflected in the
conda environment.

Tests can be run with the command `pytest <https://docs.pytest.org/en/latest/>`_, from the testing framework of the same name.

The documentation can be built with:

.. code-block:: bash

   python setup.py docs

The main page for the documentation can then be found at
`build\\sphinx\\html\\index.html` (or `build/sphinx/html/index.html` on Mac and Linux).
The file can viewed from any web browser.

The source files to create the documentation can be found in the `docs/` folder from within the main MUSE directory.

.. _anaconda distribution: https://www.anaconda.com/distribution/#download-section

.. _anaconda prompt:
   https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal

.. _anaconda virtual environment: https://www.anaconda.com/what-is-anaconda/

.. _pytest: https://docs.pytest.org/en/latest/

.. _git: https://git-scm.com/

.. __: http://try.github.io/


