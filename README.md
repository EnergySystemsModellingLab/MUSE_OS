[![PyPI version shields.io](https://img.shields.io/pypi/v/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI status](https://img.shields.io/pypi/status/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI license](https://img.shields.io/pypi/l/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![QA, tests and publishing](https://github.com/EnergySystemsModellingLab/MUSE_OS/actions/workflows/ci.yml/badge.svg)](https://github.com/EnergySystemsModellingLab/MUSE_OS/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/EnergySystemsModellingLab/MUSE_OS/graph/badge.svg?token=KJFHOHMKWK)](https://codecov.io/gh/EnergySystemsModellingLab/MUSE_OS)

# ModUlar energy system Simulation Environment: MUSE

## Installation instructions

MUSE is available in `PyPI` and therefore it can be installed easily with `pip`. Detailed instructions on how to do that ensuring the right version of Python is used can be found in the Documentation.

- Recommended installation instructions
- Instructions for developers

## Usage

Once installed, users can:

- activate the virtual environment (needed only once per session) as explained
  above
- run `muse --model default` to run the default example model
- run `muse --model default --copy XXX` to copy the model to subfolder `XXX`.
- Alternatively, run `muse settings.toml`, where `settings.toml` is an input
  file for a custom model
- run `muse --help` to get a description of the command-line arguments,
    including the name of any additional models provided with MUSE.

## Copyright

Copyright © 2023 Imperial College London
