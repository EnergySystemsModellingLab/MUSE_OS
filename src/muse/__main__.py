"""Makes MUSE executable."""

from __future__ import annotations

import pathlib
from argparse import ArgumentParser

from muse import examples

parser = ArgumentParser(description="Run a MUSE simulation")
parser.add_argument(
    "settings",
    nargs="?",
    default="settings.toml",
    type=pathlib.Path,
    help="Path to the TOML file with the simulation settings.",
)
parser.add_argument(
    "--model",
    default=None,
    choices=examples.AVAILABLE_EXAMPLES,
    help="Runs a model distributed with MUSE. "
    "If provided, the 'settings' input is ignored.",
)
parser.add_argument(
    "--copy",
    default=None,
    type=pathlib.Path,
    help="Folder where to copy the model specified by the 'model' option. "
    "The folder must not exist: this command will refuse to overwrite existing "
    "data. Exits without running the model.",
)


def muse_main(settings, model, copy):
    """Runs a MUSE simulation.

    SETTINGS should be a .toml file.
    """
    from logging import getLogger
    from pathlib import Path

    from muse import add_file_logger, examples
    from muse.mca import MCA
    from muse.readers.toml import read_settings

    if (not model) and not Path(settings).exists():
        full_path = str(Path(settings).absolute())
        raise Exception(f"Invalid or missing input: file {full_path} does not exist.")

    if copy:
        examples.copy_model(name=model if model else "default", path=copy)
    elif model:
        examples.model(model).run()
    else:
        settings = read_settings(settings)
        getLogger("muse").setLevel(settings.log_level)
        add_file_logger()
        MCA.factory(settings).run()


def run():
    args = parser.parse_args()
    muse_main(args.settings, args.model, args.copy)


def patched_broadcast_compat_data(self, other):
    """Patch for xarray.core.variable._broadcast_compat_data.

    This has been introduced to disallow automatic broadcasting along the 'timeslice'
    dimension.

    If `self` and `other` differ in whether they have a 'timeslice' dimension (in which
    case automatic broadcasting would normally be performed), an error is raised.

    In this case, developers must explicitly handle broadcasting by calling either
    `broadcast_timeslice` or `distribute_timeslice` (see `muse.timeslices`). The
    appropriate choice of operation will depend on the context and the quantity in
    question.
    """
    from xarray.core.variable import Variable, _broadcast_compat_variables

    if (isinstance(other, Variable)) and ("timeslice" in self.dims) != (
        "timeslice" in getattr(other, "dims", [])
    ):
        raise ValueError(
            "Broadcasting along the 'timeslice' dimension is required, but automatic "
            "broadcasting is disabled. Please handle it explicitly using "
            "`broadcast_timeslice` or `distribute_timeslice` (see `muse.timeslices`)."
        )

    # The rest of the function is copied directly from
    # xarray.core.variable._broadcast_compat_data
    if all(hasattr(other, attr) for attr in ["dims", "data", "shape", "encoding"]):
        # `other` satisfies the necessary Variable API for broadcast_variables
        new_self, new_other = _broadcast_compat_variables(self, other)
        self_data = new_self.data
        other_data = new_other.data
        dims = new_self.dims
    else:
        # rely on numpy broadcasting rules
        self_data = self.data
        other_data = other
        dims = self.dims

    # Check output dimensions
    if "asset" in dims and any(
        dim in dims for dim in ["region", "technology", "installed"]
    ):
        raise ValueError(
            "DataArrays with an 'asset' dimension cannot be broadcasted along "
            "'region', 'technology', or 'installed' dimensions. "
            "This error is usually raised when attempting to combine asset-level data "
            "(e.g. a capacity dataset with an 'asset' dimension) with a fully explicit "
            "technology dataset (e.g. a technology dataset with 'region' and "
            "'technology' dimensions). "
            "Please use `broadcast_over_assets` on the latter object before performing "
            "this operation."
        )

    return self_data, other_data, dims


if "__main__" == __name__:
    from unittest.mock import patch

    with patch(
        "xarray.core.variable._broadcast_compat_data", patched_broadcast_compat_data
    ):
        run()
