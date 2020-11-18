# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
<<<<<<< HEAD
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/muse"))


# -- Project information -----------------------------------------------------

project = "MUSE Documentation"
copyright = "2020, Sustainable Gas Institute"
author = "Sustainable Gas Institute"

# The full version, including alpha/beta/rc tags
release = "0.8"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
=======
# http://www.sphinx-doc.org/en/master/config

# -- Project information -----------------------------------------------------

project = "MUSE"
copyright = "2019, Sustainable Gas Institute"
author = "Imperial College London"
release = "0.7"

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "recommonmark",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]
<<<<<<< HEAD

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
=======
source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["build", "**.ipynb_checkpoints", "**/ResidentialBracket*.txt"]

modindex_common_prefix = ["muse"]
autodoc_typehints = "none"
add_module_names = False

autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------

html_theme = "classic"
html_static_path = ["_static"]


import recommonmark  # noqa


def setup(app):
    from recommonmark.transform import AutoStructify

    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
