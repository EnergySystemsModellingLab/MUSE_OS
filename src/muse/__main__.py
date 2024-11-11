"""Makes MUSE executable."""

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
    choices=examples.available_examples(),
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

    from muse import examples
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
        MCA.factory(settings).run()


def run():
    args = parser.parse_args()
    muse_main(args.settings, args.model, args.copy)


if "__main__" == __name__:
    run()
