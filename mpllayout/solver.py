"""
Routines for solving collections of primitives and constraints

The strategy to solve a collection of constraints is to use three labelled
lists representing a system of non-linear equations:
    `primitives: PrimLabelledList`
        A list of `geo.Primitive` instances representing the unknowns of the
        non-linear equations

        Each `geo.Primitive.param' attribute represents the unknown(s) that must
        be solved for to satisfy the constraints.
    `constraints: ConstraintLabelledList`
        A list of `geo.Constraint` instances representing the non-linear
        equations

        Each `geo.Constraint.assem_res' represents the non-linear equation(s)
        that must be satisfied for the constraint.
    `constraint_graph: StrGraph`
        A graph linking each constraint to the primitives in `primitives`
        (through string labels) the constraint applies to

The class `Layout` handles construction of these three lists while functions
`solve` and `solve_linear` use these lists to solve the system of constraints.
"""

import typing as typ
from numpy.typing import NDArray

import warnings

import jax
from jax import numpy as jnp
import numpy as np

from . import geometry as geo
from .containers import LabelledList

from . import layout

PrimLabelledList = LabelledList[geo.Primitive]
ConstraintLabelledList = LabelledList[geo.Constraint]

IntGraph = typ.List[typ.Tuple[int, ...]]
StrGraph = typ.List[typ.Tuple[str, ...]]

SolverInfo = typ.Mapping[str, typ.Any]


def solve(
    prim_tree: layout.PrimitiveTree,
    constraints: typ.List[geo.Constraint],
    constraint_graph: IntGraph,
    abs_tol: float = 1e-10,
    rel_tol: float = 1e-7,
    max_iter: int = 10,
) -> typ.Tuple[layout.PrimitiveTree, SolverInfo]:
    """
    Return a set of primitives that satisfy the constraints

    This solves a set of, potentially non-linear constraints, with an
    iterative Newton method.

    Parameters
    ----------
    prim_tree: layout.PrimitiveTree
        The tree of primitives
    constraints: typ.List[geo.Constraint]
        The list of constraints
    constraint_graph: IntGraph
        A mapping from each constraint to the primitives it applies to

        For example, `constraint_graph[0] == (0, 5, 8)` means the first
        constraint applies to primitives `(prims[0], prims[5], prims[8])`.
    abs_tol, rel_tol: float
        The absolute and relative tolerance for the iterative solution
    max_iter: int
        The maximum number of iterations for the iterative solution

    Returns
    -------
    layout.PrimitiveTree
        The tree of primitives satisfying the constraints
    SolverInfo
        Information about the solve

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
    prims = prim_tree.prims()
    prim_idx_bounds = np.cumsum([0] + [prim.param.size for prim in prims])

    global_param_n = np.concatenate([prim.param for prim in prims])
    prim_graph = prim_tree.prim_graph()

    @jax.jit
    def assem_global_res(global_param):
        new_prim_params = [
            global_param[idx_start:idx_end]
            for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
        ]
        residuals = assem_constraint_residual(
            new_prim_params, prim_tree, prim_graph, constraints, constraint_graph
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

        dglobal_param, err, rank, s = np.linalg.lstsq(global_jac, -global_res, rcond=None)
        global_param_n = global_param_n + dglobal_param

        n += 1
        abs_err = np.linalg.norm(global_res)
        abs_errs.append(abs_err)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rel_err = abs_errs[-1] / abs_errs[0]
        rel_errs.append(rel_err)

    nonlinear_solve_info = {
        "abs_errs": abs_errs,
        "rel_errs": rel_errs
    }

    ## Build a new primitive tree from the global parameter vector
    prim_params_n = [
        np.array(global_param_n[idx_start:idx_end])
        for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
    ]
    prim_tree_n = layout.build_tree(
        prim_tree, prim_graph, prim_params_n, {}
    )

    return prim_tree_n, nonlinear_solve_info

def assem_constraint_residual(
    prim_params: typ.List[NDArray],
    prim_tree: layout.PrimitiveTree,
    prim_graph: typ.Mapping[geo.Primitive, int],
    constraints: typ.List[geo.Constraint],
    constraint_graph: IntGraph
) -> typ.List[NDArray]:
    """
    Return a list of constraint residual vectors

    Parameters
    ----------
    prim_params: typ.List[NDArray]
        A list of parameter vectors for each unique primitive in `prim_tree`
    prim_tree: layout.PrimitiveTree
        A primitive tree
    prim_graph: typ.Mapping[geo.Primitive, int]
        A mapping from each primitive in `prim_tree` to a parameter vector in `prim_params`
    constraints: typ.List[geo.Constraint]
        A list of constraints
    constraint_graph: IntGraph
        A list of integer tuples indicating primitive arguments for each constraint

    Returns
    -------
    constraint_residuals: typ.List[NDArray]
        A list of residual vectors corresponding to each constraint in `constraints`
    """
    new_tree = layout.build_tree(
        prim_tree, prim_graph, prim_params, {}
    )
    new_prims = new_tree.prims()
    constraint_residuals = [
        constraints[constraint_idx](tuple(new_prims[idx] for idx in prim_idxs))
        for constraint_idx, prim_idxs in enumerate(constraint_graph)
    ]
    return constraint_residuals
