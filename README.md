[![PyPI version shields.io](https://img.shields.io/pypi/v/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI status](https://img.shields.io/pypi/status/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![PyPI license](https://img.shields.io/pypi/l/MUSE-OS.svg)](https://pypi.python.org/pypi/MUSE-OS/)
[![QA, tests and publishing](https://github.com/SGIModel/MUSE_OS/actions/workflows/ci.yml/badge.svg)](https://github.com/SGIModel/MUSE_OS/actions/workflows/ci.yml)

# ModUlar energy system Simulation Environment: MUSE

## Installation

### Recommended way

The recommended way for **end users** to access and use the tool is via `pipx`.
This will create an isolated environment and install MUSE-OS within in one go,
also letting you to invoke `muse` anywhere in your system.

1. Install and configure [`pipx`](https://pypa.github.io/pipx/) following the
   instructions appropriate for your operative system. Make sure it works well before
   moving on.
2. Install MUSE-OS with `pipx install MUSE-OS`. It might take a
   while to complete, but afterwards updates should be pretty fast.
3. To run MUSE-OS just open a terminal and execute `muse`, with the appropriate input
   arguments, if relevant. See section below about usage.

Whenever there is a new version of MUSE-OS, just run `pipx upgrade MUSE-OS` and
it will be downloaded and installed with no fuss.

### Alternative way

If you want to have a bit more control - or you don't want to use `pipx`,
just create a virtual environment first and then install `MUSE-OS`.

Although not strictly necessary, **creating a virtual environment is highly recommended**:
it will isolate users and developers from changes occuring on their operating system,
and from conflicts between python packages. It ensures reproducibility from day to day.

There are several ways of creating a virtual environment - below we list two of them.
Regardless of the method used, **once it has been created and activated**, you can install
`MUSE-OS` within using:

```bash
python -m pip install MUSE-OS
```

And then use it by invoking `muse` with the relevant input arguments.

#### Creating a virtual environment using `conda`

Create a virtual environment including python with:

```bash
conda create -n muse_env python=3.9
```

Activate the environment with:

```bash
conda activate muse_env
```

Later, to recover the system-wide "normal" python, deactivate the environment with:

```bash
conda deactivate
```

#### Creating a virtual environment using `venv`

Create a virtual environment with:

```bash
python -m pip install venv
python -m venv muse_env
```

Activate the environment with:

```powershell
# In Powershell
muse_env\Scripts\Activate.ps1

# In Linux/MacOS
source muse_env/bin/activate
```

Later, to recover the system-wide "normal" python, deactivate the environment with:

```bash
deactivate
```

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

## Development

It is strongly recommened to use a virtual environment, as above. The simplest approach is to
first download the muse code with:

```bash
> git clone https://github.com/SGIModel/MUSE_OS.git
```

```bash
cd MUSE_OS
# Create virtual environment - for development, this is typically called ".venv"
# Activate virtual environment. Finally, install muse:
python -m pip install -e ."[dev,doc]"
```

Please note the quotation marks. The downloaded code can then be modified. The changes will be
automatically reflected in the environment.

To ensure the consistency of the code with other developers, install the pre-commit hooks with:

```bash
python -m pip install pre-commit
pre-commit install
```

This will ensure that a series of quality assurance tools are run with every commit you make.

In the developing phase, MUSE can also be used to run test cases to check that the model would reproduce expected results from a defined set of input data. Tests can be run with the command [pytest](https://docs.pytest.org/en/latest/), from the testing framework of the same name.

The documentation can be built with:

```bash
python -m sphinx -b html docs docs/build 
```

The main page for the documentation can then be found at
`docs\\build\\html\\index.html` (or `docs/build/html/index.html` on Mac and Linux).
The file can viewed from any web browser.

[vscode](https://code.visualstudio.com/) users will find that the repository is setup
with default settings file.  Users will still need to [choose the virtual
environment](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment),
or conda environment where to run the code. This will change the `.vscode/settings.json`
file and add a user-specific path to it. Users should try and avoid commiting changes to
`.vscode/settings.json` indiscriminately.

## Copyrigh

Copyright © 2023 Imperial College London
