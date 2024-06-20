from collections.abc import Mapping
from logging import getLogger
from typing import Callable

from muse.registration import registrator

SETTINGS_CHECKS_SIGNATURE = Callable[[dict], None]
"""settings checks signature."""

SETTINGS_CHECKS: Mapping[str, SETTINGS_CHECKS_SIGNATURE] = {}
"""Dictionary of settings checks."""


@registrator(registry=SETTINGS_CHECKS, loglevel="info")
def register_settings_check(function: SETTINGS_CHECKS_SIGNATURE):
    """Decorator to register a function as a settings check.

    Registers a function as a settings check so that it can be applied easily
    when validating the MUSE input settings.

    There is no restriction on the function name, although is should be
    in lower_snake_case, as it is a python function.
    """
    from functools import wraps

    @wraps(function)
    def decorated(settings) -> None:
        result = function(settings)

        msg = f" {function.__name__} PASSED"
        getLogger(__name__).info(msg)

        return result

    return decorated
