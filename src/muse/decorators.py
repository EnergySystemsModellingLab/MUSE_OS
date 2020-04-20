from typing import Callable
from logging import getLogger
from collections import namedtuple

from muse.registration import registrator, DECORATORS_REGISTRY


SETTINGS_CHECKS = {}
"""Dictionary of settings checks."""

SETTINGS_CHECKS_SIGNATURE = Callable[[dict], None]
"""settings checks signature."""


@registrator(registry=SETTINGS_CHECKS, loglevel="info")
def register_settings_check(function: SETTINGS_CHECKS_SIGNATURE = None):
    """Decorator to register a function as a settings check.

    Registers a function as a settings check so that it can be applied easily
    when validating the MUSE input settings.

    There is no restriction on the function name, although is should be
    in lower_snake_case, as it is a python function.
    """
    from functools import wraps

    @wraps(function)
    def decorated(*args, **kwargs) -> None:
        result = function(*args, **kwargs)

        msg = " {} PASSED".format(function.__name__)
        getLogger(__name__).info(msg)

        return result

    return decorated


decorators = namedtuple("Decorators", list(DECORATORS_REGISTRY.keys()))(
    **DECORATORS_REGISTRY
)
