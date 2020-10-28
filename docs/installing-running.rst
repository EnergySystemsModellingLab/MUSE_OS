Installation
============

There are two ways to install MUSE, for users who do not wish to modify the source code of MUSE, and developers who do.

.. note::

   Windows users and developers may need to install `Windows Build Tools`__. These tools include C/C++ compilers which are needed to build some python dependencies.
   
   MacOS includes compilers by default, hence no action is needed for Mac users.
   
   Linux users may need to install a C compiler, whether GNU gcc or Clang, as well python development packages, depending on their distribution.


   .. __: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

For users
---------

MUSE is developed with python, an open-source programming language. Therefore, there are two steps to the installation process. First, python should be installed. Then, MUSE should be installed.

The simplest alternative is to install python by downloading the `Anaconda distribution`_. Choose the appropriate operating system (e.g. windows), python version 3.7, and the 64 bit installer, then follow the steps for the anaconda installer, as prompted.

Once python is installed, we can now install MUSE. MUSE can be installed via the `Anaconda Prompt`_ (or any terminal on Mac and Linux). This is a command-line interface to python and the python eco-system. In the anaconda prompt, run:

.. code-block:: bash

   python -m pip install --user git+https://github.com/SGIModel/StarMuse

It should now be possible to run muse, also in the anaconda prompt:

.. code-block:: bash

   python -m muse --help

.. note::

   Although not strictly necessary, users are encouraged to create an `Anaconda virtual environment <https://www.anaconda.com/what-is-anaconda/>`_ and install MUSE there, as shown in :ref:`installation-devs`.

.. _installation-devs:


For developers
--------------

Although not strictly necessary, creating an `Anaconda virtual environment <https://www.anaconda.com/what-is-anaconda/>`_ is highly
recommended: it will isolate users and developers from changes occuring on their
operating system, and from conflicts between python packages. It ensures reproducibility
from day to day.

Create a virtual env including python with:

.. code-block:: bash

   conda create -n muse python=3.7

Activate the environment with:

.. code-block:: bash

   conda activate muse

Later, to recover the system-wide "normal" python, deactivate the environment with:

.. code-block:: bash

   conda deactivate

The simplest approach is to first download the muse code with `git`_:

.. code-block:: bash

   git clone https://github.com/SGIModel/StarMuse.git muse

For interested users, there are plenty of `good`__ tutorials for `git`_. 
And then install the working directory into the conda environment:

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

.. _anaconda distribution: https://www.anaconda.com/distribution/#download-section

.. _anaconda prompt:
   https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal

.. _anaconda virtual environment: https://www.anaconda.com/what-is-anaconda/

.. _pytest: https://docs.pytest.org/en/latest/

.. _git: https://git-scm.com/

.. __: http://try.github.io/