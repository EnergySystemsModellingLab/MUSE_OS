"""Utilities for adapting MUSE data structures to linear programming solvers.

This module provides utilities to convert MUSE's xarray-based data structures to and
from the format required by scipy's linear programming solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from muse.constraints import ConstraintKind
from muse.timeslices import broadcast_timeslice, drop_timeslice


def unified_dataset(lpcosts: xr.Dataset, *constraints) -> xr.Dataset:
    """Creates single xr.Dataset from costs and constraints."""
    from xarray import merge

    # Reformat constraints to lp format
    lp_constraints = [lp_constraint(constraint, lpcosts) for constraint in constraints]

    # Rename variables in lp constraints
    lp_constraints = [
        constraint.rename(
            b=f"b{i}", capacity=f"capacity{i}", production=f"production{i}"
        )
        for i, constraint in enumerate(lp_constraints)
    ]

    # Rename dimensions in lpcosts
    lpcosts = lpcosts.rename({k: f"d({k})" for k in lpcosts.dims})

    # Merge data
    data = merge([lpcosts, *lp_constraints])

    # An adjustment is required for lower bound constraints
    for i, constraint in enumerate(constraints):
        if constraint.kind == ConstraintKind.LOWER_BOUND:
            data[f"b{i}"] = -data[f"b{i}"]
            data[f"capacity{i}"] = -data[f"capacity{i}"]
            data[f"production{i}"] = -data[f"production{i}"]

    # Ensure consistent ordering of dimensions
    return data.transpose(*data.dims)


def selected_quantity(data: xr.Dataset, name: str) -> xr.Dataset:
    """Select and rename variables for a specific quantity."""
    # Select data for the specified quantity ("capacity", "production", or "b")
    result = data[[u for u in data.data_vars if str(u).startswith(name)]]

    # Rename variables ("costs" for the costs variable, 0/1/2 etc. for constraints)
    return result.rename(
        {
            k: ("costs" if k == name else int(str(k).replace(name, "")))
            for k in result.data_vars
        }
    )


def reshape_constraint_matrix(matrix: xr.DataArray) -> np.ndarray:
    """Convert constraints matrix to a 2D np array.

    The rows of the constraints matrix will represent the constraints, and the
    columns will represent the decision variables.
    """
    # Before building LP we need to sort dimensions for consistency
    if list(matrix.dims) != sorted(matrix.dims):
        matrix = matrix.transpose(*sorted(matrix.dims))

    # Size of the first dimension
    # This dimension represents the number of constraints
    size = np.prod([matrix[u].shape[0] for u in matrix.dims if str(u).startswith("c")])

    # Reshape into a 2D array: N constraints x N decision variables
    return matrix.values.reshape((size, -1))


def extract_constraint_matrices(
    capacities: xr.Dataset,
    productions: xr.Dataset,
    bs: xr.Dataset,
    constraints,
    *kinds: ConstraintKind,
):
    """Extracts A and b matrices for constraints of specified kinds.

    These will end up as A_ub and b_ub for inequality constraints, and A_eq and b_eq for
    equality constraints (see ScipyAdapter).
    """
    # Get indices of constraints of the specified kind
    indices = [i for i in range(len(bs)) if constraints[i].kind in kinds]

    # Convert constraints matrices to 2d np arrays
    capa_constraints = [reshape_constraint_matrix(capacities[i]) for i in indices]
    prod_constraints = [reshape_constraint_matrix(productions[i]) for i in indices]

    # Convert constraints vectors to 1d
    constraints_vectors = [bs[i].stack(constraint=sorted(bs[i].dims)) for i in indices]

    # Concatenate constraints
    if capa_constraints:
        A = np.concatenate(
            (
                np.concatenate(capa_constraints, axis=0),
                np.concatenate(prod_constraints, axis=0),
            ),
            axis=1,
        )
        b = np.concatenate(constraints_vectors, axis=0)
    else:
        # If there are no constraints of the given kind, return None
        A = None
        b = None
    return A, b


def back_to_muse_quantity(x: np.ndarray, template: xr.DataArray) -> xr.DataArray:
    """Convert a vector of decision variables to a DataArray.

    Args:
        x: 1D vector of decision variables, outputted from the scipy solver.
        template: Template for the decision variables. This may be for either
            capacity or production variables.
    """
    # First create a multidimensional dataarray based on the template
    result = xr.DataArray(
        x.reshape(template.shape), coords=template.coords, dims=template.dims
    )

    # Then rename the dimensions (e.g. "d(asset)" -> "asset")
    return result.rename({k: str(k)[2:-1] for k in result.dims})


@dataclass
class ScipyAdapter:
    """Adapts MUSE data structures to scipy's linear programming solver format.

    This adapter converts data (costs and constraints) from xarray DataArrays to
    the format required by scipy's linear programming solver, and back.
    """

    c: np.ndarray
    capacity_template: xr.DataArray
    production_template: xr.DataArray
    bounds: tuple[float | None, float | None] = (0, np.inf)
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None

    @classmethod
    def from_muse_data(
        cls,
        capacity_costs: xr.DataArray,
        constraints: list,
        commodities: list[str],
        timeslice_level: str | None = None,
    ) -> ScipyAdapter:
        """Creates a ScipyAdapter from MUSE data structures."""
        # Calculate costs for the linear problem
        lpcosts = lp_costs(capacity_costs, commodities, timeslice_level)

        # Create dataset from costs and constraints
        data = unified_dataset(lpcosts, *constraints)

        # Get capacity constraint matrix / costs
        capacities = selected_quantity(data, "capacity")

        # Get production constraint matrix / costs
        productions = selected_quantity(data, "production")

        # Get constraint vector
        bs = selected_quantity(data, "b")

        # Create costs vector by concatenating capacity and production costs
        c = np.concatenate(
            (
                capacities["costs"].values.flatten(),
                productions["costs"].values.flatten(),
            ),
            axis=0,
        )

        # Extract A and b for inequality constraints
        A_ub, b_ub = extract_constraint_matrices(
            capacities,
            productions,
            bs,
            constraints,
            ConstraintKind.UPPER_BOUND,
            ConstraintKind.LOWER_BOUND,
        )

        # Extract A and b for equality constraints
        A_eq, b_eq = extract_constraint_matrices(
            capacities, productions, bs, constraints, ConstraintKind.EQUALITY
        )

        return cls(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            capacity_template=capacities["costs"],
            production_template=productions["costs"],
        )

    def to_muse(self, x: np.ndarray) -> xr.Dataset:
        """Convert scipy solver output back to MUSE format."""
        n_capa = self.capacity_template.size
        capa = back_to_muse_quantity(x[:n_capa], self.capacity_template)
        prod = back_to_muse_quantity(x[n_capa:], self.production_template)
        return xr.Dataset({"capacity": capa, "production": prod})

    @property
    def kwargs(self):
        """Dictionary of kwargs for scipy.optimize.linprog."""
        return {
            "c": self.c,
            "A_eq": self.A_eq,
            "b_eq": self.b_eq,
            "A_ub": self.A_ub,
            "b_ub": self.b_ub,
            "bounds": self.bounds,
        }


def lp_costs(
    capacity_costs: xr.DataArray,
    commodities: list[str],
    timeslice_level: str | None = None,
) -> xr.Dataset:
    """Creates dataset of costs for solving with scipy's LP solver.

    Importantly, this also defines the decision variables in the linear program.

    The costs applied to the capacity decision variables are provided. This should
    have dimensions "asset" and "replacement". In other words, capacity addition
    is solved for each replacement technology for each existing asset.

    No cost is applied to the production decision variables. Thus, the production
    component of the resulting dataset is zero, with dimensions determining the
    production decision variables. This will have dimensions "asset", "replacement",
    "commodity", and "timeslice". In other words, production is solved for each
    replacement technology for each existing asset, for each commodity, and for each
    timeslice.

    Args:
        capacity_costs: DataArray with dimensions "asset" and "replacement" defining the
            costs of adding capacity to the system.
        commodities: List of commodities to create production decision variables for.
        timeslice_level: The timeslice level of the linear problem.
    """
    assert set(capacity_costs.dims) == {"asset", "replacement"}

    # Start with capacity costs as template (defines "asset" and "replacement" dims)
    production_costs = xr.zeros_like(capacity_costs)

    # Add a "timeslice" dimension, convert multiindex to single index
    production_costs = broadcast_timeslice(production_costs, level=timeslice_level)
    production_costs = drop_timeslice(production_costs)
    production_costs["timeslice"] = pd.Index(
        production_costs.get_index("timeslice"), tupleize_cols=False
    )

    # Add a "commodity" dimension
    production_costs = production_costs.expand_dims(commodity=commodities)
    assert set(production_costs.dims) == {
        "asset",
        "replacement",
        "commodity",
        "timeslice",
    }

    # Result is dataset of provided capacity costs and zero production costs
    return xr.Dataset(dict(capacity=capacity_costs, production=production_costs))


def lp_constraint(constraint, lpcosts: xr.Dataset) -> xr.Dataset:
    """Transforms the constraint to LP data.

    The goal is to create from ``lpcosts.capacity``, ``constraint.capacity``, and
    ``constraint.b`` a 2d-matrix ``constraint`` vs ``decision variables``.

    #. The dimensions of ``constraint.b`` are the constraint dimensions. They are
        renamed ``"c(xxx)"``.
    #. The dimensions of ``lpcosts`` are the decision-variable dimensions. They are
        renamed ``"d(xxx)"``.
    #. ``set(b.dims).intersection(lpcosts.xxx.dims)`` are diagonal
        in constraint dimensions and decision variables dimension, with ``xxx`` the
        capacity or the production
    #. ``set(constraint.xxx.dims) - set(lpcosts.xxx.dims) - set(b.dims)`` are reduced by
        summation, with ``xxx`` the capacity or the production
    #. ``set(lpcosts.xxx.dims) - set(constraint.xxx.dims) - set(b.dims)`` are added for
        expansion, with ``xxx`` the capacity or the production

    See :py:func:`muse.lp_adapter.lp_constraint_matrix` for a more detailed explanation
    of the transformations applied here.

    """
    constraint = constraint.copy(deep=False)

    # Deal with timeslice multiindex
    if "timeslice" in constraint.dims:
        constraint = drop_timeslice(constraint)
        constraint["timeslice"] = pd.Index(
            constraint.get_index("timeslice"), tupleize_cols=False
        )

    # Rename dimensions in b
    b = constraint.b.drop_vars(set(constraint.b.coords) - set(constraint.b.dims))
    b = b.rename({k: f"c({k})" for k in b.dims})

    # Create capacity constraint matrix
    capacity = lp_constraint_matrix(constraint.b, constraint.capacity, lpcosts.capacity)
    capacity = capacity.drop_vars(set(capacity.coords) - set(capacity.dims))

    # Create production constraint matrix
    production = lp_constraint_matrix(
        constraint.b, constraint.production, lpcosts.production
    )
    production = production.drop_vars(set(production.coords) - set(production.dims))

    # Combine data
    result = xr.Dataset(
        {"b": b, "capacity": capacity, "production": production}, attrs=constraint.attrs
    )
    return result


def lp_constraint_matrix(
    b: xr.DataArray, constraint: xr.DataArray, lpcosts: xr.DataArray
):
    """Transforms one constraint block into an lp matrix.

    The goal is to create from ``lpcosts``, ``constraint``, and ``b`` a 2d-matrix of
    constraints vs decision variables.

     #. The dimensions of ``b`` are the constraint dimensions. They are renamed
         ``"c(xxx)"``.
     #. The dimensions of ``lpcosts`` are the decision-variable dimensions. They are
         renamed ``"d(xxx)"``.
     #. ``set(b.dims).intersection(lpcosts.dims)`` are diagonal
         in constraint dimensions and decision variables dimension
     #. ``set(constraint.dims) - set(lpcosts.dims) - set(b.dims)`` are reduced by
         summation
     #. ``set(lpcosts.dims) - set(constraint.dims) - set(b.dims)`` are added for
         expansion
     #. ``set(b.dims) - set(constraint.dims) - set(lpcosts.dims)`` are added for
         expansion. Such dimensions only make sense if they consist of one point.

    The result is the constraint matrix, expanded, reduced and diagonalized for the
    conditions above.
    """
    from functools import reduce

    from numpy import eye

    # Sum over all dimensions that are not in the constraint or the decision variables
    result = constraint.sum(set(constraint.dims) - set(lpcosts.dims) - set(b.dims))

    # Rename dimensions for decision variables
    result = result.rename(
        {k: f"d({k})" for k in set(result.dims).intersection(lpcosts.dims)}
    )

    # Rename dimensions for constraints
    result = result.rename(
        {k: f"c({k})" for k in set(result.dims).intersection(b.dims)}
    )

    # Expand dimensions that are in the decision variables but not in the constraint
    expand = set(lpcosts.dims) - set(constraint.dims) - set(b.dims)
    result = result.expand_dims(
        {f"d({k})": lpcosts[k].rename({k: f"d({k})"}).set_index() for k in expand}
    )

    # Expand dimensions that are in the constraint but not in the decision variables
    expand = set(b.dims) - set(constraint.dims) - set(lpcosts.dims)
    result = result.expand_dims(
        {f"c({k})": b[k].rename({k: f"c({k})"}).set_index() for k in expand}
    )

    # Dimensions that are in both the decision variables and the constraint
    diag_dims = set(b.dims).intersection(lpcosts.dims)
    diag_dims = sorted(diag_dims)

    if diag_dims:

        def get_dimension(dim):
            if dim in b.dims:
                return b[dim].values
            if dim in lpcosts.dims:
                return lpcosts[dim].values
            return constraint[dim].values

        diagonal_submats = [
            xr.DataArray(
                eye(len(b[k])),
                coords={f"c({k})": get_dimension(k), f"d({k})": get_dimension(k)},
                dims=(f"c({k})", f"d({k})"),
            )
            for k in diag_dims
        ]
        reduced = reduce(xr.DataArray.__mul__, diagonal_submats)
        if "d(timeslice)" in reduced.dims:
            reduced = reduced.drop_vars("d(timeslice)")
        result = result * reduced

    return result
