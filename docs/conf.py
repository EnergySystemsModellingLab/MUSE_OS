# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Project information -----------------------------------------------------

project = "MUSE"
copyright = "2019, Sustainable Gas Institute"
author = "Imperial College London"
release = "0.9"

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "recommonmark",
    "nbsphinx",
    "ipykernel",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]
source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = [
    "build",
    "**.ipynb_checkpoints",
    "**/ResidentialBracket*.txt",
    "_sources/*",
    "_build/*",
    "tutorial-code/*",
]

modindex_common_prefix = ["muse"]
autodoc_typehints = "none"
add_module_names = False

autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

bibtex_bibfiles = []

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
