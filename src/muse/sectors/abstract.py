from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from xarray import Dataset


class AbstractSector(ABC):
    """Abstract base class for sectors.

    Sectors are part of type hierarchy with :py:class:`AbstractSector` at the apex: all
    sectors should derive from :py:class:`AbstractSector` directly or indirectly.

    MUSE only requires two things of a sector. Sector should be instanstiable via a
    :py:meth:`~AbstractSector.factory` function. And they should be callable via
    :py:meth:`~AbstractSector.next`.

    :py:class:`AbstractSector` declares an interface with these two functions. Sectors
    which derive from it will be warned if either method is not implemented.
    """

    @classmethod
    @abstractmethod
    def factory(cls, name: str, settings: Any) -> AbstractSector:
        """Creates class from settings named-tuple."""
        pass

    @abstractmethod
    def next(self, mca_market: Dataset) -> Dataset:
        """Advance sector by one time period."""
        pass

    def __repr__(self):
        return f"<{self.name.title()} sector - object at {hex(id(self))}>"
