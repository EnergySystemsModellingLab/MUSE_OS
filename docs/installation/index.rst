Installing MUSE
===============

There are several ways of installing MUSE depending on your level of proficiency, your operating system and, specially, what you want to do with it (only using it or also developing it). In this section, you will find appropriate instructions for all of these use cases.

The following table summarises the different options to help you decide on the best one for you:

+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
|                             | **Standalone**           | **pipx-based**         | **Virtual env. based**  | **Developers**        |
+=============================+==========================+========================+=========================+=======================+
| **Installation complexity** | Easy                     | Medium                 | High                    | Highest               |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Update**                  | Full re-install          | Only MUSE              | Only MUSE               | Only MUSE             |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Needs environment to run**| No                       | No                     | Yes                     | Yes                   |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Custom plugins**          | If no extra dependency   | If no extra dependency | Yes                     | Yes                   |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Call MUSE from python**   | No                       | No                     | Yes                     | Yes                   |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Access to source code**   | No                       | No                     | No                      | Yes                   |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+
| **Operating system**        | Windows only             | Win, Linux, MacOS      | Win, Linux, MacOS       | Win, Linux, MacOS     |
+-----------------------------+--------------------------+------------------------+-------------------------+-----------------------+

The :ref:`standalone-muse` and :ref:`pipx-based` installation methods should be the preferred ones for most users who just want to use MUSE *as is* or using custom plugins that only need dependencies already used by MUSE (eg. ``pandas``).

The :ref:`virtual-env-based` gives your more flexibility in using and expanding MUSE, enable its use programmatically, i.e. within a python script or Jupyter notebook, as well as creating plugins with arbitrary complexity and dependencies. This comes at the cost of a more complex installation and running process that requires creating and activating a virtual environment manually.

The :ref:`developers` method should only by used by developers, i.e. those who want to actively contribute to the MUSE code base. It involves the installation of other tools, like git or pandoc, as well as the need to follow some instructions on how the new code is formatted, the documentation created, etc.

.. toctree::
   :maxdepth: 1

   standalone-muse
   pipx-based
   virtual-env-based
   developers
