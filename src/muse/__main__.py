"""Makes MUSE executable."""
import pathlib
from gooey import GooeyParser, Gooey
from muse import VERSION

parser = GooeyParser(description="Run a MUSE simulation")
parser.add_argument(
    "settings",
    nargs="?",
    default="settings.toml",
    type=pathlib.Path,
    help="Path to the TOML file with the simulation settings.",
    widget="FileChooser",
)
parser.add_argument(
    "--model",
    default=None,
    choices=["default", "multiple-agents", "medium", "minimum-service", "trade"],
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
    widget="DirChooser",
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
        raise Exception(f"Invalid or missing input: file {settings} does not exist.")

    if copy:
        examples.copy_model(name=model if model else "default", path=copy)
    elif model:
        examples.model(model).run()
    else:
        settings = read_settings(settings)
        getLogger("muse").setLevel(settings.log_level)
        MCA.factory(settings).run()


menu = [
    {
        "name": "Help",
        "items": [
            {
                "type": "Link",
                "menuTitle": "Join the mailing list",
                "url": "https://groups.google.com/g/muse-model",
            },
            {
                "type": "Link",
                "menuTitle": "Log a question in GitHub",
                "url": "https://github.com/SGIModel/MUSE_OS/issues/new/choose",
            },
            {
                "type": "AboutDialog",
                "menuTitle": "About MUSE",
                "name": "MUSE",
                "description": "ModUlar energy system Simulation Environment",
                "version": VERSION,
                "copyright": "2023",
                "website": "https://www.imperial.ac.uk/muse-energy/",
                "developer": "https://www.imperial.ac.uk/muse-energy/muser-group/",
                "license": "BSD-3",
            },
        ],
    }
]


@Gooey(
    program_name=f"MUSE - v{VERSION}",
    program_description="ModUlar energy system Simulation Environment",
    menu=menu,
)
def run():
    args = parser.parse_args()
    muse_main(args.settings, args.model, args.copy)


if "__main__" == __name__:
    run()
