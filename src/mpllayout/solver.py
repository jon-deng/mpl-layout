"""
Solvers for constrained geometric primitives
"""

from typing import Any
from numpy.typing import NDArray

import warnings

import jax
from jax import numpy as jnp
import numpy as np

from . import primitives as pr
from . import constraints as cr

from . import layout as lay

IntGraph = list[tuple[int, ...]]
StrGraph = list[tuple[str, ...]]

SolverInfo = dict[str, Any]

# TODO: Add different solver possibilities? (optimization, newton, etc.)
def solve(
    layout: lay.Layout,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-7,
    max_iter: int = 10,
) -> tuple[pr.PrimitiveNode, SolverInfo]:
    """
    Return geometric primitives that satisfy constraints

    This solves a set of, potentially non-linear, geoemtric constraints with an
    iterative Newton method.

    Parameters
    ----------
    layout: lay.Layout
        The layout of geometric primitives and constraints to solve
    abs_tol, rel_tol: float
        The absolute and relative tolerance for the iterative solution
    max_iter: int
        The maximum number of iterations for the iterative solution

    Returns
    -------
    pr.PrimitiveNode
        A primitive tree satisfying the constraints
    SolverInfo
        Information about the iterative solution

        Keys are:
            'abs_errs':
                A list of absolute errors for each solver iteration.
                This is the 2-norm of the constraint residual vector.
            'rel_errs':
                A list of relative errors for each solver iteration.
                This is the absolute error at each iteration, relative to the
                initial absolute error.
    """

    ## Set-up assembly function for the global residual as a function of a global
    ## parameter list

    # `prim_idx_bounds` stores the right/left indices for each primitive's
    # parameter vector in the global parameter vector array
    # For primitive with index `n`, for example,
    # `prim_idx_bounds[n], prim_idx_bounds[n+1]` are the indices between which
    # the parameter vectors are stored.
    root_prim = layout.root_prim
    prim_graph, prims = lay.build_prim_graph(root_prim)
    prim_idx_bounds = np.cumsum([0] + [prim.value.size for prim in prims])
    global_param_n = np.concatenate([prim.value for prim in prims])

    constraints, constraint_graph, constraint_params = layout.flat_constraints()

    @jax.jit
    def assem_global_res(global_param):
        new_prim_params = [
            global_param[idx_start:idx_end]
            for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
        ]
        residuals = assem_constraint_residual(
            root_prim, prim_graph, new_prim_params, constraints, constraint_graph, constraint_params
        )
        return jnp.concatenate(residuals)

    assem_global_jac = jax.jacfwd(assem_global_res)

    ## Iteratively minimize the global residual as function of the global parameter vector
    abs_errs = []
    rel_errs = []

    n = 0
    abs_err = np.inf
    rel_err = np.inf
    while (abs_err > abs_tol) and (rel_err > rel_tol) and (n < max_iter):

        global_res = assem_global_res(global_param_n)
        global_jac = assem_global_jac(global_param_n)

        dglobal_param, err, rank, s = np.linalg.lstsq(
            global_jac, -global_res, rcond=None
        )
        global_param_n = global_param_n + dglobal_param

        n += 1
        abs_err = np.linalg.norm(global_res)
        abs_errs.append(abs_err)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rel_err = abs_errs[-1] / abs_errs[0]
        rel_errs.append(rel_err)

    nonlinear_solve_info = {"abs_errs": abs_errs, "rel_errs": rel_errs}

    ## Build a new primitive tree from the global parameter vector
    prim_params_n = [
        np.array(global_param_n[idx_start:idx_end])
        for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
    ]
    root_prim_n = lay.build_tree(root_prim, prim_graph, prim_params_n)

    return root_prim_n, nonlinear_solve_info


def assem_constraint_residual(
    root_prim: pr.Primitive,
    prim_graph: dict[str, int],
    prim_values: list[NDArray],
    constraints: list[cr.Constraint],
    constraint_graph: list[cr.PrimKeys],
    constraint_params: list[cr.ResParams]
) -> list[NDArray]:
    """
    Return a list of constraint residual vectors

    Parameters
    ----------
    root_prim: pr.Primitive
        The root primitive tree
    prim_graph: dict[str, int]
        A mapping from each primitive key to a value index in `prim_values`
    prim_values: list[NDArray]
        A list of new primitive vectors
    constraints: list[cr.Constraint]
        A list of constraints
    constraint_graph: list[cr.PrimKeys]
        A list of keys indicating primitives in `root_prim` for each constraint
    constraint_params: list[cr.ResParams]
        A list of parameters for each constraint

    Returns
    -------
    residuals: list[NDArray]
        A list of constraint residual vectors

        Each residual vector corresponds to a constraint in `constraints`
    """
    root_prim = lay.build_tree(root_prim, prim_graph, prim_values)
    residuals = [
        constraint(tuple(root_prim[key] for key in prim_keys), param)
        for constraint, prim_keys, param in zip(constraints, constraint_graph, constraint_params)
    ]
    return residuals
