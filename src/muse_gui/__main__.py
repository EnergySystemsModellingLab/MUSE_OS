import os
import pathlib

try:
    from gooey import Gooey, GooeyParser
except ImportError:
    msg = "Gooey not installed! Make sure you install it to run the MUSE GUI version."
    raise ImportError(msg)

os.environ["MUSE_COLOR_LOG"] = "False"
from muse import VERSION
from muse.__main__ import muse_main

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
parser.add_argument(
    "--working-directory",
    default=str(pathlib.Path.home()),
    type=pathlib.Path,
    help="Sets the working directory.",
    widget="DirChooser",
)


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
                "url": "https://github.com/EnergySystemsModellingLab/MUSE_OS/issues/new/choose",
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
    default_size=(600, 650),
    progress_regex=r"^Finish simulation year \d+ \((?P<current>\d+)/(?P<total>\d+)\)!$",
    progress_expr="current / total * 100",
    timing_options={
        "show_time_remaining": True,
        "hide_time_remaining_on_complete": True,
    },
)
def run():
    args = parser.parse_args()
    os.chdir(args.working_directory)
    muse_main(args.settings, args.model, args.copy)


if "__main__" == __name__:
    run()
