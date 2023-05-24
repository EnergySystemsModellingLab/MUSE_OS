
Installation
============

Pre-requisite: A virtual environment
------------------------------------

Although not strictly necessary, creating a [conda](https://www.anaconda.com/what-is-anaconda/)
virtual environment is highly recommended: it will isolate users and developers from changes
occuring on their operating system, and from conflicts between python packages. It ensures
reproducibility from day to day.

Create a virtual env including python with:

```bash
> conda create -n muse python=3.8
```

Activate the environment with:

```bash
> conda activate muse
```

Later, to recover the system-wide "normal" python, deactivate the environment with:

```bash
> conda deactivate
```

Installing muse itself
----------------------

Once a virtual environment has been *activated*, as describe above, we can
install muse without fear of interfering with other python jobs. Run:

```bash
> python -m pip install https://github.com/SGIModel/MUSE_OS.git#egg=muse
```

Usage
-----

Once installed, users can:

- activate the virtual environment (needed only once per session) as explained
  above
- run `python -m muse --model default` to run the default example model
- run `python -m muse --model default --copy XXX` to copy the model to subfolder `XXX`.
- Alternatively, run `python -m muse settings.toml`, where `settings.toml` is an input
  file for a custom model
- run `python -m muse --help` to get a description of the command-line arguments,
    including the name of any additional models provided with MUSE.

Development
-----------

It is strongly recommened to use a conda virtual environment, as above. The simplest approach is to
first download the muse code with:

```bash
> git clone https://github.com/SGIModel/MUSE_OS.git muse
```

And then install the working directory into the conda environment:

```bash
> # after activating the virtual environment with:
> # conda activate muse
> python -m pip install -e ."muse[dev,doc]"
```

Please note the quotation marks. `muse` in the last line above is the path to source code that was
just downloaded with `git`. The downloaded code can then be modified. The changes will be
automatically reflected in the conda environment.

In the developing phase, MUSE can also be used to run test cases to check that the model would reproduce expected results from a defined set of input data.
Tests can be run with the command [pytest](https://docs.pytest.org/en/latest/), from
theb testing framework of the same name.

The documentation can be built with:

```bash
> python -m sphinx -b html docs docs/build 
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

Copyright
---------

Copyright © 2021 Imperial College London
