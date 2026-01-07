Developer Guide
===============

This is a work-in-progress guide for developers.

Making a new release
--------------------

The process for making a new release of MUSE is simple:

- Check the current version number. The best way to do this is to look at the latest
  tagged release on `GitHub <https://github.com/EnergySystemsModellingLab/MUSE_OS/releases>`_.
- Decide on the new version number, incrementing the second
  digit for major changes (e.g. ``v1.2.5`` -> ``v1.3.0``), or the third digit for minor changes
  (e.g. ``v1.2.5`` -> ``v1.2.6``). Note the the first digit must NOT be incremented as this
  is reserved for the `MUSE2 project <https://github.com/EnergySystemsModellingLab/MUSE2>`_.
- Update the version number and date in ``CITATION.cff``
- Write a release notes document in ``docs/release-notes/`` for the new version, following the
  template of previous release notes. Make sure to link this in ``docs/release-notes/index.rst``.
- On GitHub, go to "Releases" -> "Draft a new release". Create a new tag named after the
  new version number (e.g. "v1.3.0"), and give the release a matching title.
  Then click "Publish release".
- This will automatically trigger a new release on `PyPI <https://pypi.org/project/MUSE-OS/>`_,
  a new DOI on `Zenodo <https://zenodo.org/records/14832641>`_, and
  a new documentation build on `ReadTheDocs <https://muse-os.readthedocs.io/en/latest/>`_.
  Allow some time for these to complete, then check that everything looks correct.
