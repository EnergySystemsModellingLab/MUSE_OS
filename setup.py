#!/usr/bin/env python
"""Installation Script."""

import sys
from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

try:
    from sphinx.setup_command import BuildDoc

    docs_args = {"cmdclass": {"docs": BuildDoc}}
except ImportError:
    docs_args = {}

if sys.version_info < (3, 7):
    raise ImportError("MUSE requires Python>=3.7")

tests_require = [
    "pytest>4.0.2",
    "flake8!=3.8.1,!=3.8.0",
    "black",
    "pytest-flake8",
    "IPython",
    "jupyter",
    "nbconvert",
    "nbformat",
    "mypy",
    "numpy>=1.17",
]
docs_require = ["sphinx", "recommonmark", "nbsphinx", "sphinxcontrib-bibtex", "ipython"]


def find_data_files(directory, suffixes=(".toml", ".csv")):
    result = []
    for child in Path(directory).iterdir():
        if child.is_dir():
            result.extend(find_data_files(child, suffixes))
        elif child.suffix in suffixes:
            result.append(str(child.absolute()))
    return result


def pattern(package, repo, sha):
    return f"{package}@git+https://github.com/SGIModel/{repo}.git@{sha}"


sgidata = pattern(package="SGIModelData", repo="SGIModelData", sha="master")
muse_legacy = pattern(package="StarMUSELegacy", repo="StarMuse", sha="archive/legacy")

setup(
    name="StarMUSE",
    version="0.9",
    description="Energy System Model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science",
        "Intended Audience :: Research",
        "Intended Audience :: Economists",
        "Intended Audience :: Energy Experts",
        "Intended Audience :: Climate Mitigation Experts",
        "Intended Audience :: Industry",
    ],
    keywords=["energy", "modelling"],
    author="Sustainable Gas Institute",
    author_email="sgi@imperial.ac.uk",
    url="http://www.sustainablegasinstitute.org/home/muse-energy-model/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "muse": find_data_files(Path(__file__).parent / "src" / "muse" / "data")
    },
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "pandas>=0.21",
        "click",
        "xarray>0.14" and "xarray<0.17",
        "bottleneck",
        "coloredlogs",
        "toml",
        "xlrd==1.2.0",
    ],
    extras_require={
        "dev": tests_require,
        "excel": ["openpyxl"],
        "docs": docs_require,
        "private_sgi_model": [sgidata, muse_legacy],
    },
    tests_require=tests_require,
    **docs_args,
)
