"""Define a sector, e.g. aggregation of agents.

There are three main kinds of sectors classes, encompassing three use cases:

- :class:`~muse.sectors.sector.Sector`: The main workhorse sector of the model. It
  contains only on kind of data, namely the agents responsible for holding assets and
  investing in new assets.
- :class:`~muse.sectors.preset_sector.PresetSector`: A sector that is meant to generate
  demand for the sectors above using a fixed formula or schedule.
- :class:`~muse.sectors.legacy_sector.LegacySector`: A wrapper around the original MUSE
  sectors.

All the sectors derive from :class:`AbstractSector`. The :class:`AbstractSector` defines
two `abstract`__ functions which should be declared by derived sectors. `Abstract`__
here means a common programming practice where some concept in the code (e.g. a sector)
is given an explicit interface, with the goal of making it easier for other programmers
to use and implement the concept.

__ https://docs.python.org/3/library/abc.html

__ https://www.python-course.eu/python3_abstract_classes.php

- :meth:`AbstractSector.factory`: Creates a sector from input data
- :meth:`AbstractSector.next`: A function which takes a market (demand, supply,
  prices) and returns a market.  What happens within could be anything, though it will
  likely consists of dispatch and investment.

New sectors can be registered with the MUSE input files using
:func:`muse.sectors.register.register_sector`.
"""

__all__ = [
    "AbstractSector",
    "Sector",
    "PresetSector",
    "LegacySector",
    "register_sector",
    "SECTORS_REGISTERED",
]
from muse.sectors.abstract import AbstractSector
from muse.sectors.legacy_sector import LegacySector
from muse.sectors.preset_sector import PresetSector
from muse.sectors.register import SECTORS_REGISTERED, register_sector
from muse.sectors.sector import Sector
