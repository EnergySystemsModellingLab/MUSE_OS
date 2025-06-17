from __future__ import annotations

from logging import getLogger
from typing import Callable

SETTINGS_HOOKS_SIGNATURE = Callable[[dict], None]
"""settings checks signature."""

SETTINGS_HOOKS: list[tuple[int, str, SETTINGS_HOOKS_SIGNATURE]] = []
"""Dictionary of settings checks."""


def register_settings_hook(
    function: SETTINGS_HOOKS_SIGNATURE = None, *, priority: int = 100
) -> Callable:
    """Register a function to be called during settings validation.

    The function will be called with the settings dictionary as its only argument.
    The function can modify the settings dictionary in place.

    Args:
        function: The function to register
        priority: The priority of the function. Lower numbers are called first.

    Returns:
        The decorated function
    """
    if function is None:
        return lambda f: register_settings_hook(f, priority=priority)

    def decorated(func: SETTINGS_HOOKS_SIGNATURE) -> SETTINGS_HOOKS_SIGNATURE:
        """Register the function and return it unchanged."""
        getLogger(__name__).debug(
            f"Registering settings hook {func.__name__} with priority {priority}"
        )
        SETTINGS_HOOKS.append((priority, func.__name__, decorated))
        return func

    return decorated(function)
