# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# -- Project information -----------------------------------------------------

from importlib.metadata import version as get_version

from docutils import nodes
from docutils.parsers.rst import roles

project = "MUSE"
copyright = "2024, Imperial College London"
author = "Imperial College London"
release = get_version("MUSE_OS")
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

html_theme = "sphinx_rtd_theme"

# -- Render GitHub links -------------------------------------------------


def github_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Creates links to issues/pull requests on the MUSE_OS GitHub site.

    To use in markdown:
    {github}`ISSUE_NUMBER` (e.g. {github}`123`)

    To use in rst:
    :github:`ISSUE_NUMBER` (e.g. :github:`123`)

    In both cases this will create a clickable link (visible as #123) to the relevant
    GitHub page (i.e. https://github.com/EnergySystemsModellingLab/MUSE_OS/issues/123)

    The base URL is for the issues page, but this will also work for pull requests and
    discussions, as GitHub will automatically redirect to the appropriate page.
    """
    base_url = "https://github.com/EnergySystemsModellingLab/MUSE_OS/issues/"
    url = f"{base_url}{text}"
    node = nodes.reference(rawtext, f"#{text}", refuri=url, **(options or {}))
    return [node], []


roles.register_canonical_role("github", github_role)
