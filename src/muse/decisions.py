"""Decision methods combining several objectives into ones.

Decisions methods create a single scalar from multiple objectives. To be available from
the input, functions implementing decision methods should follow a specific signature:

.. code-block:: Python

    @register_decision
    def weighted_sum(objectives: Dataset, parameters: Any, **kwargs) -> DataArray:
        pass


Arguments:
    objectives: An dataset where each array is a separate objective
    parameters: parameters, such as weights, whether to minimize or maximize, the names
        of objectives to consider, etc.
    kwargs: Extra input parameters. These parameters are expected to be set from the
        input file.

        .. warning::

            The standard :ref:`agent csv file<inputs-agents>` does not allow to set
            these parameters.

Returns:
    A data array with ranked replacement technologies.
"""

__all__ = [
    "register_decision",
    "mean",
    "weighted_sum",
    "lexical_comparison",
    "retro_lexical_comparison",
    "epsilon_constraints",
    "retro_epsilon_constraints",
    "single_objective",
    "factory",
]
from collections.abc import Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from xarray import DataArray, Dataset

from muse.registration import registrator

PARAMS_TYPE = Sequence[tuple[str, bool, float]]
"""Standard decision parameter type.

Until MUSE input is more flexible, we need to be able to translate from this
form to whatever the decision function allows. The standard form is a sequence
of tuples ('objective name', maximize if True else minimize, some float).
"""

DECISION_SIGNATURE = Callable[[Dataset, PARAMS_TYPE], DataArray]
"""Signature of functions implementing decisions."""

DECISIONS: MutableMapping[str, DECISION_SIGNATURE] = {}
"""Dictionary of decision functions.

Decision functions aggregate separate objectives into a single number per
asset and replacement technology. They are also known as multi-objectives.
"""


@registrator(registry=DECISIONS, loglevel="info")
def register_decision(function: DECISION_SIGNATURE, name: str):
    """Decorator to register a function as a decision.

    Registers a function as a decision so that it can be applied easily when aggregating
    different objectives together.
    """
    from functools import wraps

    # make sure the return array is named according to decision
    @wraps(function)
    def decorated(*args, **kwargs) -> DataArray:
        result = function(*args, **kwargs)
        if isinstance(result, DataArray):
            result.name = name
        return result

    return decorated


def coeff_sign(minimise: bool, coeff: Any):
    """Adds sign to coefficient depending on minimizing or maximizing.

    This function standardizes across the decision methods.
    """
    return coeff if minimise else -coeff


def factory(settings: Union[str, Mapping] = "mean") -> Callable:
    """Creates a decision method based on the input settings."""
    if isinstance(settings, str):
        function = DECISIONS[settings]
        params: dict = {}
    else:
        function = DECISIONS[settings["name"]]
        params = {k: v for k, v in settings.items() if k != "name"}

    def decision(objectives: Dataset, **kwargs) -> DataArray:
        return function(objectives, **params, **kwargs)  # type: ignore

    return decision


@register_decision
def mean(objectives: Dataset, *args, **kwargs) -> DataArray:
    """Mean over objectives."""
    from xarray import concat

    allobjectives = concat(objectives.data_vars.values(), dim="concat_var")
    return allobjectives.mean(set(allobjectives.dims) - {"asset", "replacement"})


@register_decision
def weighted_sum(objectives: Dataset, parameters: Mapping[str, float]) -> DataArray:
    r"""Weighted sum over normalized objectives.

    The objectives are each normalized to [-1, 1] over the `replacement`
    dimension by dividing by the maximum absolute value. Furthermore, the dimensions
    other than `asset` and `replacement` are reduced by taking the mean.

    More specifically, the objective function is:

    .. math::

        \sum_m c_m \frac{A_m - \min(A_m)}{\max(A_m) - \min(A_m)}

    where sum runs over the different objectives, c_m is a scalar coefficient,
    A_m is a matrix with dimensions (existing tech, replacement tech). `max(A)`
    and `min(A)` return the largest and smallest component of the input matrix.
    If c_m is positive, then that particular objective is minimized, whereas if
    it is negative, that particular objective is maximized.
    """
    from numpy import fabs

    # normalize input if given in DECISION_PARAMETERS format
    if not isinstance(parameters, Mapping):
        parameters = {u[0]: coeff_sign(u[1], u[2]) for u in parameters}

    # normalize objectives
    if len(objectives.replacement):
        norm = objectives.map(fabs).max("replacement")
        norm = norm.where(norm > 1e-12, 1)
        normalized = objectives / norm
    else:
        normalized = objectives

    # reduce dimensionality to only 'asset' and 'replacement'
    normalized = normalized.mean(set(normalized.dims) - {"asset", "replacement"})

    # sum all objectives together
    names = list(normalized.data_vars)
    result = parameters.get(names[0], 1) * normalized[names[0]]
    for name in names[1:]:
        result = result + parameters.get(name, 1) * normalized[name]
    return result


@register_decision(name="lexo")
def lexical_comparison(
    objectives: Dataset, parameters: Union[PARAMS_TYPE, Sequence[tuple[str, float]]]
) -> DataArray:
    """Lexical comparison over the objectives.

    Lexical comparison operates by binning the objectives into bins of width
    w_i = min_j(p_i o_i^j). Once binned, dimensions other than `asset` and
    `technology` are reduced by taking the max, e.g. the largest constraint.
    Finally, the objectives are ranked lexographically, in the order given by the
    parameters.

    The result is an array of tuples which can subsequently be compared
    lexicographically.
    """
    from muse.utilities import lexical_comparison

    assert len(parameters) > 0
    if len(parameters[0]) == 3:
        parameters = [(u[0], coeff_sign(u[1], u[2])) for u in parameters]
    assert set(objectives.data_vars).issuperset([u[0] for u in parameters])
    order = [u[0] for u in parameters]
    binsize = (Dataset(dict(parameters)) * objectives).min("replacement")
    return lexical_comparison(objectives, binsize, order=order, bin_last=False).rank(
        "replacement"
    )


@register_decision(name="retro_lexo")
def retro_lexical_comparison(
    objectives: Dataset, parameters: Union[PARAMS_TYPE, Sequence[tuple[str, float]]]
) -> DataArray:
    """Lexical comparison over the objectives.

    Lexical comparison operates by binning the objectives into bins of width
    w_i = p_i o_i, where i are the current assets. Once binned, dimensions other
    than `asset` and `replacement` are reduced by taking the max, e.g. the
    largest constraint.  Finally, the objectives are ranked lexographically, in
    the order given by the parameters.

    The result is an array of tuples which can subsequently be compared
    lexicographically.
    """
    from muse.utilities import lexical_comparison

    assert len(parameters) > 0
    if len(parameters[0]) == 3:
        parameters = [(u[0], coeff_sign(u[1], u[2])) for u in parameters]
    assert objectives.asset.isin(objectives.replacement).all()
    assert set(objectives.data_vars).issuperset([u[0] for u in parameters])

    order = [u[0] for u in parameters]
    binsize = Dataset(dict(parameters)) * objectives.sel(replacement=objectives.asset)
    return lexical_comparison(objectives, binsize, order=order, bin_last=False).rank(
        "replacement"
    )


def _epsilon_constraints(
    objectives: Dataset, optimize: str, mask: Optional[Any] = None, **epsilons
) -> DataArray:
    """Minimizes one objective subject to constraints on other objectives."""
    constraints = True
    for name, epsilon in epsilons.items():
        reduced_dims = set(objectives[name].dims) - {"asset", "replacement"}
        constraints = constraints & (objectives[name] <= epsilon).all(reduced_dims)

    if mask is None:
        mask = objectives[optimize].max() + 1
    return objectives[optimize].where(constraints, mask)


@register_decision(name=("epsilon", "epsilon_con"))
def epsilon_constraints(
    objectives: Dataset,
    parameters: Union[PARAMS_TYPE, Sequence[tuple[str, bool, float]]],
    mask: Optional[Any] = None,
) -> DataArray:
    r"""Minimizes first objective subject to constraints on other objectives.

    The parameters are a sequence of tuples `(name, minimize, epsilon)`, where
    `name` is the name of the objective, `minimize` is `True` if minimizing and
    false if maximizing that objective, and `epsilon` is the constraint. The
    first objective is the one that will be minimized according to:

    Given objectives :math:`O^{(i)}_t`, with :math:`i \in [|1, N|]` and :math:`t` the
    replacement technologies, this function computes the ranking with respect to
    :math:`t`:

    .. math::

        \mathrm{ranking}_{O^{(i)}_t < \epsilon_i} O^{(0)}_t


    The first tuple can be restricted to `(name, minimize)`, since `epsilon` is ignored.

    The result is the matrix :math:`O^{(0)}` modified such minimizing over the
    replacement dimension value would take into account the constraints and the
    optimization direction (minimize or maximize). In other words, calling
    `result.rank('replacement')` will yield the expected result.
    """
    assert set(objectives.data_vars).issuperset([param[0] for param in parameters])
    do_minimize = Dataset({k: coeff_sign(v, 1) for k, v, _ in parameters[1:]})
    do_minimize[parameters[0][0]] = 1 if parameters[0][1] else -1
    dict_params = {k: v for k, _, v in parameters[1:] if k in objectives.data_vars}
    constraints = do_minimize * Dataset(dict_params)
    return _epsilon_constraints(
        objectives * do_minimize, parameters[0][0], mask=mask, **constraints.data_vars
    )


@register_decision(name="retro_epsilon")
def retro_epsilon_constraints(
    objectives: Dataset, parameters: PARAMS_TYPE
) -> DataArray:
    """Epsilon constraints where the current tech is included.

    Modifies the parameters to the function such that the existing technologies are
    always competitive.
    """
    asset_objectives = objectives.sel(replacement=objectives.asset)

    def transform(name, minimize, epsilon=None):
        if epsilon is None:
            return name, minimize
        am = getattr(asset_objectives, name)
        new_eps = am.where(am > epsilon if minimize else am < epsilon, epsilon)
        return name, minimize, new_eps

    parameters = [
        transform(*param) for param in parameters if param[0] in objectives.data_vars
    ]
    return epsilon_constraints(objectives, parameters)


@register_decision(name=("single", "singleObj"))
def single_objective(
    objectives: Dataset,
    parameters: Union[str, tuple[str, bool], tuple[str, bool, float], PARAMS_TYPE],
) -> DataArray:
    """Single objective decision method.

    It only decides on minimization vs maximization and multiplies by a given factor.
    The input parameters can take the following forms:

    - Standard sequence `[(objective, direction, factor)]`, in which case it must have
      only one element.
    - A single string: defaults to standard sequence `[(string, 1, 1)]`
    - A tuple (string, bool): defaults to standard sequence
      `[(string, direction, 1)]`
    - A tuple (string, bool, factor): defaults to standard sequence
      `[(string, direction, factor)]`
    """
    if isinstance(parameters, str):
        params = parameters, 1, 1
    elif len(parameters) == 1 and isinstance(parameters[0], str):
        params = parameters[0], 1, 1
    elif len(parameters) == 1:
        params = parameters[0]
    elif len(parameters) == 2:
        params = parameters[0], parameters[1], 1
    elif len(parameters) == 3:
        params = parameters
    else:
        raise ValueError("Incorrect format for the agent input 'parameters'")
    return objectives[params[0]] * coeff_sign(params[1], params[2])
