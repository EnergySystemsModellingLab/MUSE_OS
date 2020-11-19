"""Makes MUSE executable."""
if "__main__" == __name__:
    from sys import argv, executable

    import click

    INPUT_PATH = click.Path(exists=False, file_okay=True, resolve_path=True)
    MODEL_PATH = click.Path(file_okay=False, resolve_path=True)
<<<<<<< HEAD
    MODELS = click.Choice(
        ["default", "multiple-agents", "medium", "minimum-service", "trade"]
    )
=======
    MODELS = click.Choice(["default", "multiple-agents"])
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

    @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
    @click.argument("settings", default="settings.toml", type=INPUT_PATH)
    @click.option(
        "--model",
        type=MODELS,
        help="Runs a model distributed with MUSE. SETTINGS is ignored.",
    )
    @click.option(
        "--copy",
        type=MODEL_PATH,
        help=(
<<<<<<< HEAD
            "Folder where to copy the model specified by the '--model' option. "
            "The folder must not exist: this command will refuse to overwrite existing "
            "data. Exits without running the model."
=======
            "Folder where to copy a model. Defaults to the default model."
            ". Exists without running the model."
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        ),
    )
    def muse_main(settings, model, copy):
        """Runs a MUSE simulation.

        SETTINGS should be a .toml file.
        """
        from pathlib import Path
        from logging import getLogger
        from muse.readers.toml import read_settings
        from muse.mca import MCA
        from muse import examples

        if (not model) and not Path(settings).exists():
            print(f"Invalid or missing input: file {settings} does not exist.")
            return

        if copy:
            examples.copy_model(name=model if model else "default", path=copy)
        elif model:
            examples.model(model).run()
        else:
            settings = read_settings(settings)
            getLogger("muse").setLevel(settings.log_level)
            MCA.factory(settings).run()

    argv[0] = "%s -m muse" % executable
    muse_main()
