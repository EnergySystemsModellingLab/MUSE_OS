from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

from muse.sectors.abstract import AbstractSector

SECTORS_REGISTERED: Mapping[str, Callable] = {}
"""Dictionary of sectors."""


def register_sector(
    sector_class: Optional[type[AbstractSector]] = None,
    name: Optional[Union[str, Sequence[str]]] = None,
) -> type[AbstractSector]:
    """Registers a sector so it is available MUSE-wide.

    Example:
        >>> from muse.sectors import AbstractSector, register_sector
        >>> @register_sector(name="MyResidence")
        ... class ResidentialSector(AbstractSector):
        ...     pass
    """
    from inspect import isclass, isfunction
    from logging import getLogger

    if sector_class is None:
        return lambda x: register_sector(x, name=name)  # type: ignore

    if isinstance(name, str):
        names: Sequence[str] = (name,)
    elif name is None:
        names = (sector_class.__name__,)
    else:
        names = name

    if isclass(sector_class) and not issubclass(sector_class, AbstractSector):
        raise RuntimeError("A sector must inherit from AbstractSector")

    for n in names:
        if n in SECTORS_REGISTERED:
            msg = f"A Sector class with the name {n} already exists"
            getLogger(__name__).warning(msg)
            return sector_class

        if isfunction(sector_class):
            SECTORS_REGISTERED[n] = sector_class  # type: ignore
        elif isclass(sector_class):
            SECTORS_REGISTERED[n] = sector_class.factory  # type: ignore

    if len(names) <= 1:
        aliases = ""
    elif len(names) == 2:
        aliases = f", with alias {names[-1]}"
    else:
        aliases = f", with aliases {' '.join(names[1:-1])} and {names[-1]}"

    getLogger(__name__).info(f"Sector {names[0]} registered{aliases}.")

    return sector_class
