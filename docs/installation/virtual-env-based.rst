.. _virtual-env-based:

Virtual environment-based installation
--------------------------------------

If the :ref:`pipx-based` does not work for you, you don't want to use ``pyenv`` or ``pipx``, or if you are having trouble with those tools, there are a couple of alternatives.

Installing Anaconda Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

To install Anaconda, just go to the official `Anaconda webpage <https://www.anaconda.com/>`_ and download and install a version appropriate for your operating system. Do not worry about the Python version as ``conda`` will let you choose that when creating a virtual environment.

The installer should guide you step by step on the process of installing Anaconda and configuring your system to use it as your Python installation.

.. _virtual-environments:

Creating virtual environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although not strictly necessary, **creating a virtual environment is highly recommended** regardless of how you installed Python. It will isolate users and developers from changes occurring on their operating system, and from conflicts between python packages and it ensures reproducibility from day to day.

Using ``pipx`` ensures that each application it installs has its own virtual environment, running it under the hood. However, you can explicitly create and manage the virtual environment if you prefer.

.. _conda-venvs:

Creating a ``conda`` virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option is available only if you installed Anaconda Python. Depending on the settings you used when installing Anaconda and your operating system, you might have ``conda`` available in your normal terminal or you might need to use the Anaconda Prompt.

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


.. _python_venv:

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

Creating a virtual environment with ``pyenv + venv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively to creating virtual environments in ``conda``, you can also make use of two well-tested and maintained libraries.
We met the first one, ``pyenv``, already in the :ref:`pipx-based <pipx-based>` under the section :ref:`Installing pyenv <pipx-based-installing-pyenv>` and the installation procedure is exactly the same.
If you go down that route, please follow the steps outlined there and chose a recent version ``Python``, say 3.9.

The second package we need to create virtual environments for any specific ``Python`` version is called
``venv``, and it ships with ``Python`` by default. To create such an environment, we first need to ensure that the
``Python`` version we wish to use is indeed the one we want. To do this, open the terminal and invoke ``pyenv versions``.
To install different versions or set them to local or global scope, please refer again to
:ref:`Installing pyenv <pipx-based-installing-pyenv>`.

We can now create virtual environments using ``Python`` directly as explained in :ref:`python_venv`.


Installing standalone MUSE in a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the method used, **once it has been created and activated**, you can install ``MUSE`` within using:

.. code-block:: bash

    python -m pip install muse-os

And then use it by invoking ``muse`` with the relevant input arguments. Keep in mind that, contrary to using ``pipx``, in this case **you will need to manually activate the environment every time you want to use MUSE**.
