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
    "epsilon_constraints",
    "factory",
    "lexical_comparison",
    "mean",
    "register_decision",
    "retro_epsilon_constraints",
    "retro_lexical_comparison",
    "single_objective",
    "weighted_sum",
]

from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
)

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from muse.registration import registrator
from muse.timeslices import broadcast_timeslice, drop_timeslice
from muse.utilities import tupled_dimension

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


def factory(settings: str | Mapping = "mean") -> Callable:
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
    return allobjectives.mean(
        set(allobjectives.dims) - {"asset", "replacement", "timeslice"}
    )


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
    objectives: Dataset, parameters: PARAMS_TYPE | Sequence[tuple[str, float]]
) -> DataArray:
    """Lexicographic comparison using the best available replacements as reference.

    Lexical comparison operates by binning the objectives into bins of width

        w_i = min_j(p_i o_i^j),

    where o_i^j are the objective values of the candidate replacements and the
    minimum is taken over the replacement dimension. Once binned, dimensions
    other than ``asset`` and ``replacement`` are reduced by taking the maximum
    (e.g. the largest constraint). Finally, the objectives are ranked
    lexicographically in the order given by ``parameters``.

    The result is an array of tuples which can subsequently be compared
    lexicographically.
    """
    assert len(parameters) > 0

    # Convert (name, min/max, coefficient) specifications to
    # (name, signed coefficient), where the sign encodes whether the
    # objective should be minimised or maximised.
    if len(parameters[0]) == 3:
        parameters = [
            (name, coeff_sign(minmax, coeff)) for name, minmax, coeff in parameters
        ]

    # Lexicographic priority of the objectives.
    order = tuple(name for name, _ in parameters)

    # All requested objectives must be present.
    assert set(objectives.data_vars).issuperset(order)

    # Define the bin widths as
    #
    #     w_i = min_j(p_i o_i^j),
    #
    # i.e. the weighted objective values of the candidate replacements,
    # taking the minimum over the replacement dimension.
    binsize = objectives.copy(deep=True)

    # Temporarily flatten the timeslice MultiIndex to avoid xarray's
    # deprecated Dataset assignment path when modifying variables.
    if "timeslice" in binsize.indexes:
        index_names = binsize.indexes["timeslice"].names
        binsize = binsize.reset_index("timeslice")

    # Apply the objective weights (including minimisation/maximisation
    # direction encoded in the sign).
    for name, weight in parameters:
        binsize[name] *= weight

    # Restore the original timeslice MultiIndex structure.
    if "timeslice" in objectives.indexes:
        binsize = binsize.set_index(timeslice=index_names)

    # Use the smallest weighted objective across replacements as the
    # bin width for each objective.
    binsize = binsize.min("replacement")

    # Construct lexicographically comparable tuples and rank the
    # replacement options accordingly.
    return _lexical_comparison(
        objectives,
        binsize,
        order=order,
        keep_last_continuous=True,
    ).rank("replacement")


@register_decision(name="retro_lexo")
def retro_lexical_comparison(
    objectives: Dataset,
    parameters: PARAMS_TYPE | Sequence[tuple[str, float]],
) -> DataArray:
    """Lexicographic comparison using current assets as reference.

    Lexical comparison operates by binning the objectives into bins of width

        w_i = p_i o_i,

    where o_i are the objective values of the current assets. Once binned,
    dimensions other than ``asset`` and ``replacement`` are reduced by taking
    the maximum (e.g. the largest constraint). Finally, the objectives are
    ranked lexicographically in the order given by ``parameters``.

    The result is an array of tuples which can subsequently be compared
    lexicographically.
    """
    assert len(parameters) > 0

    # Convert (name, min/max, coefficient) specifications to
    # (name, signed coefficient), where the sign encodes whether the
    # objective should be minimised or maximised.
    if len(parameters[0]) == 3:
        parameters = [
            (name, coeff_sign(minmax, coeff)) for name, minmax, coeff in parameters
        ]

    # Retrofitting compares candidate replacements against the current
    # asset, so every asset must also appear amongst the replacements.
    assert objectives.asset.isin(objectives.replacement).all()

    # Lexicographic priority of the objectives.
    order = tuple(name for name, _ in parameters)

    # All requested objectives must be present.
    assert set(objectives.data_vars).issuperset(order)

    # Define the bin widths as
    #
    #     w_i = p_i o_i,
    #
    # where o_i are the objective values of the current assets. The
    # objective values are selected by matching each asset to itself
    # along the replacement dimension.
    binwidths = Dataset(dict(parameters)) * objectives.sel(replacement=objectives.asset)

    # Construct lexicographically comparable tuples and rank the
    # replacement options accordingly.
    return _lexical_comparison(
        objectives,
        binwidths,
        order=order,
        keep_last_continuous=True,
    ).rank("replacement")


def _lexical_comparison(
    objectives: xr.Dataset,
    binwidths: xr.Dataset,
    order: Sequence[Hashable],
    *,
    keep_last_continuous: bool = False,
) -> xr.DataArray:
    """Lexical comparison over the objectives.

    Lexical comparison operates by binning the objectives into bins of width
    ``binwidths``. Once binned, dimensions other than ``asset`` and
    ``replacement`` are reduced by taking the maximum (e.g. the largest
    constraint). Finally, the objectives are ranked lexicographically in the
    order given by ``order``.

    Arguments:
        objectives: xr.Dataset containing the objectives to rank.
        binwidths: Bin widths used to discretise the objectives.
        order: Order in which objectives are compared lexicographically.
        keep_last_continuous: Whether the final objective should be left as a
            continuous value, rather than being discretised.

    Result:
        An array of tuples which can subsequently be compared
        lexicographically.
    """
    # Restrict to the objectives participating in the comparison and
    # temporarily flatten the timeslice MultiIndex to avoid Dataset
    # assignment issues in xarray.
    result = drop_timeslice(objectives[list(order)]).copy()

    # All objectives are discretised except, optionally, the final
    # tie-breaking objective.
    discretized = order[:-1] if keep_last_continuous else order

    # Convert objectives to integer-valued bins.
    for name in discretized:
        result[name] = np.floor(result[name] / binwidths[name]).astype(np.int64)

    # Preserve the final objective as a continuous quantity for
    # tie-breaking, while still normalising by its bin width.
    if keep_last_continuous:
        name = order[-1]
        result[name] = result[name] / binwidths[name]

    # Combine the ordered objectives into tuples that can be compared
    # lexicographically.
    return result.to_array(dim="objective").reduce(tupled_dimension, dim="objective")


def _epsilon_constraints(
    objectives: Dataset,
    optimize: str,
    mask: Any | None = None,
    **epsilons,
) -> DataArray:
    """Selects the best value of a target objective subject to epsilon constraints.

    Each constraint enforces that an objective must be below (or above, after
    sign handling upstream) a threshold, aggregated over all non-(asset,
    replacement) dimensions.
    """
    # Start with all options feasible
    constraints = True

    # Build feasibility mask from epsilon constraints
    for name, epsilon in epsilons.items():
        # Reduce over all non-decision dimensions (e.g. timeslice, region)
        reduced_dims = set(objectives[name].dims) - {"asset", "replacement"}

        # All slices must satisfy constraint
        constraints = constraints & (objectives[name] <= epsilon).all(reduced_dims)

    # Default mask = something worse than any feasible objective value
    if mask is None:
        mask = objectives[optimize].max() + 1

    # Return objective values, masking infeasible alternatives
    return objectives[optimize].where(constraints, mask)


@register_decision(name=("epsilon", "epsilon_con"))
def epsilon_constraints(
    objectives: Dataset,
    parameters: PARAMS_TYPE | Sequence[tuple[str, bool, float]],
    mask: Any | None = None,
) -> DataArray:
    """Epsilon-constraint optimisation.

    The first objective is optimised (min or max), while all subsequent
    objectives are treated as constraints of the form:

        objective_i <= epsilon_i

    after sign normalization.
    """
    assert set(objectives.data_vars).issuperset([p[0] for p in parameters])

    # Remove obj_data parameters if present
    optimize_name, optimize_minimize, _ = parameters[0]

    # Encode optimization direction
    do_minimize = Dataset({optimize_name: 1 if optimize_minimize else -1})

    # Remaining objectives also get sign encoding
    for name, minimize, _ in parameters[1:]:
        do_minimize[name] = coeff_sign(minimize, 1)

    # Extract epsilon constraints
    epsilons = {
        name: coeff_sign(minimize, 1) * eps
        for name, minimize, eps in parameters[1:]
        if name in objectives.data_vars
    }

    # Apply sign transformation + constraints
    if "timeslice" in objectives.indexes:
        do_minimize = broadcast_timeslice(do_minimize)
    return _epsilon_constraints(
        objectives * do_minimize,
        optimize_name,
        mask=mask,
        **epsilons,
    )


@register_decision(name="retro_epsilon")
def retro_epsilon_constraints(
    objectives: Dataset,
    parameters: PARAMS_TYPE,
) -> DataArray:
    """Epsilon-constraint optimisation with asset-relative thresholds.

    Epsilon thresholds are adjusted so that the current technology is always
    feasible, ensuring it remains in the choice set.
    """
    # Extract current asset baseline
    asset_objectives = objectives.sel(replacement=objectives.asset)

    def adapt_param(name, minimize, epsilon=None):
        """Adjust epsilon so that current asset is always feasible."""
        if epsilon is None:
            return name, minimize, None

        current = asset_objectives[name]

        # Work in the same transformed logic as epsilon_constraints
        sign = -1 if minimize else 1

        # Ensure current asset is not excluded by its own constraint
        adjusted = current.where(
            (sign * current) <= (sign * epsilon),
            epsilon,
        )

        return name, minimize, adjusted

    # Filter valid objectives and adapt epsilons
    parameters = [adapt_param(*p) for p in parameters if p[0] in objectives.data_vars]

    return epsilon_constraints(objectives, parameters)


@register_decision(name=("single", "singleObj"))
def single_objective(
    objectives: Dataset,
    parameters: str | tuple[str, bool] | tuple[str, bool, float] | PARAMS_TYPE,
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
