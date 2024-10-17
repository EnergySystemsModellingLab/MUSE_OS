"""Methods and types around commodities."""

from collections.abc import Sequence
from enum import IntFlag, auto
from typing import Union

from numpy import ndarray
from xarray import DataArray, Dataset


class CommodityUsage(IntFlag):
    """Flags to specify the different kinds of commodities.

    For details on how ``enum``'s work, see `python's documentation`__. In practice,
    :py:class:`CommodityUsage` centralizes in one place the different kinds of
    commodities that are meaningful to the generalized sector, e.g. commodities that
    are consumed by the sector, and commodities that produced by the sectors, as well
    commodities that are, somehow, *environmental*.

    __ https://docs.python.org/3/library/enum.html#enum.IntFlag

    With the exception of ``CommodityUsage.OTHER``, flags can be combined in any
    fashion. ``CommodityUsage.PRODUCT | CommodityUsage.CONSUMABLE`` is a commodity that
    is both consumed and produced by a sector. ``CommodityUsage.ENVIRONMENTAL |
    CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE`` is an environmental energy
    commodity consumed by the sector.

    ``CommodityUsage.OTHER`` is an alias for *no* flag. It is meant for commodities
    that should be ignored by the sector.
    """

    OTHER = 0
    """Not relevant for current sector."""
    CONSUMABLE = auto()
    """Commodity which can be consumed by the sector."""
    PRODUCT = auto()
    """Commodity which can be produced by the sector."""
    ENVIRONMENTAL = auto()
    """Commodity which is a pollutant."""
    ENERGY = auto()
    """Commodity which is a fuel for this or another sector."""
    # BYPRODUCT = auto()

    @property
    def name(self) -> str:
        """Hack to get the name of the flag consistently across python versions."""
        return (
            self._name_
            if self._name_ is not None
            else "|".join(
                [
                    com._name_
                    for com in CommodityUsage
                    if com in self and com != CommodityUsage.OTHER
                ]
            )
        )

    @staticmethod
    def from_technologies(technologies: Dataset) -> DataArray:
        from numpy import array, bitwise_or

        def just_tech(x):
            dims = set(x.dims)
            if "commodity" in dims:
                dims.remove("commodity")
            return x.any(dims)

        if "fixed_outputs" not in technologies.data_vars:
            raise ValueError("Missing 'fixed_outputs' array in technologies")
        products = just_tech(technologies["fixed_outputs"] > 0)
        products = [
            CommodityUsage.PRODUCT if u else CommodityUsage.OTHER for u in products
        ]

        consumables = list(
            just_tech(technologies[x] > 0)
            for x in {"fixed_inputs", "flexible_inputs"}
            if x in technologies.data_vars
        )
        if len(consumables) == 0:
            raise ValueError("Missing input array in technologies")
        elif len(consumables) == 1:
            consumables = consumables[0]
        else:
            consumables = bitwise_or(*consumables)
        consumables = [
            CommodityUsage.CONSUMABLE if u else CommodityUsage.OTHER
            for u in consumables
        ]

        if "comm_type" in technologies:
            envs = [
                CommodityUsage.ENVIRONMENTAL if u else CommodityUsage.OTHER
                for u in (technologies.comm_type == "environmental")
            ]
            nrgs = [
                CommodityUsage.ENERGY if u else CommodityUsage.OTHER
                for u in (technologies.comm_type == "energy")
            ]
        else:
            envs = [CommodityUsage.OTHER for u in consumables]
            nrgs = [CommodityUsage.OTHER for u in consumables]

        return DataArray(
            array(
                [
                    a | b | c | d
                    for a, b, c, d in zip(products, consumables, envs, nrgs)
                ],
                dtype=CommodityUsage,
            ),
            coords={"commodity": technologies.commodity},
            dims="commodity",
        )


def check_usage(
    data: Sequence[CommodityUsage],
    flag: Union[str, CommodityUsage, None],
    match: str = "all",
) -> ndarray:
    """Match usage flags with input data array.

    Arguments:
        data: sequence for which to match flags elementwise.
        flag: flag or combination of flags to match. The input can be a string, such as
            "product | environmental", or a CommodityUsage instance.
            Defaults to "other".
        match: one of:
            - "all": should all flag match. Default.
            - "any", should match at least one flags.
            - "exact", should match each flag and nothing else.

    Examples:
        >>> from muse.commodities import CommodityUsage, check_usage
        >>> data = [
        ...     CommodityUsage.OTHER,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENVIRONMENTAL,
        ... ]

        Matching "all":

        >>> check_usage(data, CommodityUsage.PRODUCT).tolist()
        [False, True, True, False]

        >>> check_usage(data, CommodityUsage.ENVIRONMENTAL).tolist()
        [False, False, True, True]

        >>> check_usage(
        ...     data, CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT
        ... ).tolist()
        [False, False, True, False]

        Matching "any":

        >>> check_usage(data, CommodityUsage.PRODUCT, match="any").tolist()
        [False, True, True, False]

        >>> check_usage(data, CommodityUsage.ENVIRONMENTAL, match="any").tolist()
        [False, False, True, True]

        >>> check_usage(data, "environmental | product", match="any").tolist()
        [False, True, True, True]

        Matching "exact":

        >>> check_usage(data, "PRODUCT", match="exact").tolist()
        [False, True, False, False]

        >>> check_usage(data, CommodityUsage.ENVIRONMENTAL, match="exact").tolist()
        [False, False, False, True]

        >>> check_usage(data, "ENVIRONMENTAL | PRODUCT", match="exact").tolist()
        [False, False, True, False]

        Finally, checking no flags has been set can be done with:

        >>> check_usage(data, CommodityUsage.OTHER, match="exact").tolist()
        [True, False, False, False]
        >>> check_usage(data, None, match="exact").tolist()
        [True, False, False, False]
    """
    from functools import reduce

    from numpy import bitwise_and, equal

    if isinstance(flag, str) and len(flag) > 0:
        usage = {
            k.lower(): getattr(CommodityUsage, k)
            for k in dir(CommodityUsage)
            if isinstance(getattr(CommodityUsage, k), CommodityUsage)
        }

        flag = reduce(
            lambda x, y: x | y, [usage[a.lower().strip()] for a in flag.split("|")]
        )
    elif isinstance(flag, str) or flag is None:
        flag = CommodityUsage.OTHER

    if match.lower() == "all":
        return bitwise_and(data, flag) == flag
    elif match.lower() == "any":
        return bitwise_and(data, flag).astype(bool)
    elif match.lower() == "exact":
        return equal(data, flag).astype(bool)
    else:
        raise ValueError(f"Unknown match {match}")


def is_pollutant(data: Sequence[CommodityUsage]) -> ndarray:
    """Environmental product.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_pollutant
        >>> data = [
        ...     CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENVIRONMENTAL,
        ...     CommodityUsage.PRODUCT | CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT,
        ... ]
        >>> is_pollutant(data).tolist()
        [False, False, False, False, True]
    """
    return check_usage(
        data, CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT, match="all"
    )


def is_consumable(data: Sequence[CommodityUsage]) -> ndarray:
    """Any consumable.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_consumable
        >>> data = [
        ...     CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENVIRONMENTAL,
        ...     CommodityUsage.PRODUCT | CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT,
        ... ]
        >>> is_consumable(data).tolist()
        [True, False, False, True, False]
    """
    return check_usage(data, CommodityUsage.CONSUMABLE)


def is_fuel(data: Sequence[CommodityUsage]) -> ndarray:
    """Any consumable energy.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_fuel
        >>> data = [
        ...     CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENERGY,
        ...     CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE
        ...         | CommodityUsage.ENVIRONMENTAL,
        ...     CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE
        ...         | CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENERGY | CommodityUsage.PRODUCT,
        ... ]
        >>> is_fuel(data).tolist()
        [False, False, False, True, True, True, False]
    """
    return check_usage(data, CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE, "all")


def is_material(data: Sequence[CommodityUsage]) -> ndarray:
    """Any non-energy non-environmental consumable.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_material
        >>> data = [
        ...     CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENERGY,
        ...     CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.CONSUMABLE | CommodityUsage.ENVIRONMENTAL,
        ...     CommodityUsage.ENERGY | CommodityUsage.CONSUMABLE
        ...         | CommodityUsage.PRODUCT,
        ...     CommodityUsage.CONSUMABLE | CommodityUsage.PRODUCT,
        ... ]
        >>> is_material(data).tolist()
        [True, False, False, False, False, False, True]
    """
    from numpy import logical_and

    return logical_and(
        ~check_usage(
            data, CommodityUsage.ENERGY | CommodityUsage.ENVIRONMENTAL, match="any"
        ),
        check_usage(data, CommodityUsage.CONSUMABLE),
    )


def is_enduse(data: Sequence[CommodityUsage]) -> ndarray:
    """Non-environmental product.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_enduse
        >>> data = [
        ...     CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.ENVIRONMENTAL,
        ...     CommodityUsage.PRODUCT | CommodityUsage.CONSUMABLE,
        ...     CommodityUsage.ENVIRONMENTAL | CommodityUsage.PRODUCT,
        ... ]
        >>> is_enduse(data).tolist()
        [False, True, False, True, False]
    """
    from numpy import logical_and

    return logical_and(
        ~check_usage(data, CommodityUsage.ENVIRONMENTAL),
        check_usage(data, CommodityUsage.PRODUCT),
    )


def is_other(data: Sequence[CommodityUsage]) -> ndarray:
    """No flags are set.

    Examples:
        >>> from muse.commodities import CommodityUsage, is_other
        >>> data = [
        ...     CommodityUsage.OTHER,
        ...     CommodityUsage.PRODUCT,
        ...     CommodityUsage.PRODUCT | CommodityUsage.OTHER,
        ... ]
        >>> is_other(data).tolist()
        [True, False, False]
    """
    return check_usage(data, CommodityUsage.OTHER, match="exact")
