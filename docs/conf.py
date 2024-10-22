# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# -- Project information -----------------------------------------------------

project = "MUSE"
copyright = "2024, Imperial College London"
author = "Imperial College London"
release = "1.3.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx.ext.graphviz",
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
nbsphinx_allow_errors = True
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

bibtex_bibfiles: list[str] = []

# -- GraphViz configuration ----------------------------------
graphviz_output_format = "svg"

# -- Options for HTML output -------------------------------------------------

html_theme = "classic"
