"""Registrators that allow pluggable data to logic transforms."""
__all__ = ["registrator"]

from typing import Callable, Mapping, Optional, Sequence, Text, Union

from xarray import DataArray

# Registry of available decorators
DECORATORS_REGISTRY = {}


def name_variations(*args):
    """Standard name variations when registering functions with MUSE."""

    def camelCase(name):
        comps = name.split("_")
        return comps[0] + "".join(x.title() for x in comps[1:])

    def CamelCase(name):  # noqa
        return "".join(x.title() for x in name.split("_"))

    def kebab_case(name):
        return name.replace("_", "-")

    def nospacecase(name):
        return name.replace("_", "")

    # keep ordered function name because first one is the most likely variation.
    names = [a for a in args if a is not None]
    names += (
        [camelCase(n) for n in names]
        + [CamelCase(n) for n in names]
        + [kebab_case(n) for n in names]
        + [nospacecase(n) for n in names]
    )
    ordered = []
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def registrator(
    decorator: Callable[[Callable], Callable] = None,
    registry: Mapping = None,
    logname: Optional[Text] = None,
    loglevel: Optional[Text] = "Debug",
) -> Callable:
    """A decorator to create a decorator that registers functions with MUSE.

    This is a decorator that takes another decorator as an argument. Hence it
    returns a decorator. It simplifies and standardizes creating decorators to
    register functions with muse.

    The registrator expects as non-optional keyword argument a registry where
    the resulting decorator will register functions.

    Furthermore, the final function (the one passed to the decorator passed to
    this function) will emit a standardized log-call.

    Example:
        At it's simplest, creating a registrator and registrating happens by
        first declaring a registry.

        >>> REGISTRY = {}

        In general, it will be a variable owned directly by a module, hence the
        all-caps. Creating the registrator then follows:

        >>> from muse.registration import registrator
        >>> @registrator(registry=REGISTRY, logname='my stuff',
        ...              loglevel='Info')
        ... def register_mystuff(function):
        ...     return function


        This registrator does nothing more than register the function. A more
        interesting example is given below. Then a function can be registered:

        >>> @register_mystuff(name='yoyo')
        ... def my_registered_function(a, b):
        ...     return a + b

        The argument 'yoyo' is optional. It adds aliases for the function in the
        registry. In any case, functions are registered with default aliases
        corresponding to standard name variations, e.g. CamelCase, camelCase,
        and kebab-case, as illustrated below:

        >>> REGISTRY['my_registered_function'] is my_registered_function
        True
        >>> REGISTRY['my-registered-function'] is my_registered_function
        True
        >>> REGISTRY['yoyo'] is my_registered_function
        True

        A more interesting case would involve the registrator automatically
        adding functionality to the input function. For instance, the inputs
        could be manipulated and the result of the function could be
        automatically transformed to a string:

        >>> from muse.registration import registrator
        >>> @registrator(registry=REGISTRY)
        ... def register_mystuff(function):
        ...     from functools import wraps
        ...
        ...     @wraps(function)
        ...     def decorated(a, b) -> str:
        ...         result = function(2 * a, 3 * b)
        ...         return str(result)
        ...
        ...     return decorated

        >>> @register_mystuff
        ... def other(a, b):
        ...     return a + b

        >>> isinstance(REGISTRY['other'](-3, 2), str)
        True
        >>> REGISTRY['other'](-3, 2) == "0"
        True
    """
    from functools import wraps

    # allows specifyng the registered name as a keyword argument
    if decorator is None:
        return lambda x: registrator(
            x, loglevel=loglevel, logname=logname, registry=registry
        )

    if registry is None:
        raise Exception("registry keyword must be given and cannot be None")

    if logname is None:
        logname = decorator.__name__.replace("register_", "")

    @wraps(decorator)
    def register(
        function=None,
        name: Optional[Union[Text, Sequence[Text]]] = None,
        vary_name: bool = True,
        overwrite: bool = False,
    ):
        from logging import getLogger
        from inspect import signature
        from itertools import chain

        # allows specifyng the registered name as a keyword argument
        if function is None:
            return lambda x: register(
                x, name=name, vary_name=vary_name, overwrite=overwrite
            )

        if name is None:
            names = [function.__name__]
        elif isinstance(name, Text):
            names = [name, function.__name__]
        else:
            names = list(name) + [function.__name__]

        # all registered filters will use the same logger, at least for the
        # default logging done in the decorated function
        logger = getLogger(function.__module__)
        msg = "Computing {}: {}".format(logname, names[0])

        if "name" in signature(decorator).parameters:
            inner_decorated = decorator(function, names[0])
        else:
            inner_decorated = decorator(function)

        @wraps(function)
        def decorated(*args, **kwargs) -> DataArray:
            if loglevel is not None and hasattr(logger, loglevel):
                getattr(logger, loglevel)(msg)
            result = inner_decorated(*args, **kwargs)
            return result

        # There's just one name for the decorator
        if not vary_name:
            if function.__name__ in registry and not overwrite:
                msg = f"A {logname} with the name {function.__name__} already exists"
                getLogger(__name__).warning(msg)
                return
            registry[function.__name__] = decorated

        else:
            for n in chain(name_variations(function.__name__, *names)):
                if n in registry and not overwrite:
                    msg = f"A {logname} with the name {n} already exists"
                    getLogger(__name__).warning(msg)
                    return
                registry[n] = decorated

        return decorated

    DECORATORS_REGISTRY[decorator.__name__] = register

    return register
