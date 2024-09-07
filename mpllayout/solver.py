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
import warnings

import jax
from jax import numpy as jnp
import numpy as np

from . import geometry as geo
from .array import LabelledList

from . import layout

PrimIdx = geo.PrimitiveIndex
PrimIdxs = typ.Tuple[PrimIdx, ...]
PrimLabelledList = LabelledList[typ.Union[geo.Primitive, geo.PrimitiveArray]]
ConstraintLabelledList = LabelledList[geo.Constraint]

PrimIdxGraph = typ.List[PrimIdxs]
IntGraph = typ.List[typ.Tuple[int, ...]]
StrGraph = typ.List[typ.Tuple[str, ...]]

SolverInfo = typ.Mapping[str, typ.Any]

def solve(
        primitive_tree: layout.PrimitiveTree,
        constraints: ConstraintLabelledList,
        constraint_graph: IntGraph,
        abs_tol: float = 1e-10,
        rel_tol: float = 1e-7,
        max_iter: int = 10
    ) -> typ.Tuple[PrimLabelledList, SolverInfo]:
    """
    Return a set of primitives that satisfy the constraints

    This solves a set of, potentially non-linear constraints, with an
    iterative Newton method.

    Parameters
    ----------
    prims: PrimLabelledList
        The list of primitives
    constraints: ConstraintLabelledList
        The list of constraints
    constraint_graph: IntGraph
        A mapping from each constraint to the primitives it applies to

        For example, `constraint_graph[0] == (0, 5, 8)` means the first
        constraint applies to primitives `(prims[0], prims[5], prims[8])`.
    abs_tol, rel_tol: float
        The absolute and relative tolerance for the iterative solution
    max_iter: int = 10
        The maximum number of iterations for the iterative solution

    Returns
    -------
    PrimLabelledList
        The list of primitives satisfying the constraints
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

    nonlinear_solve_info = {}
    abs_errs = []
    rel_errs = []

    n = 0
    abs_err = np.inf
    rel_err = np.inf
    prims_n = primitive_tree
    while (abs_err > abs_tol) and (rel_err > rel_tol) and (n < max_iter):
        prims_n, linear_solve_info = solve_linear(prims_n, constraints, constraint_graph)

        n += 1
        abs_err = np.linalg.norm(linear_solve_info['res'])
        abs_errs.append(abs_err)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rel_err = abs_errs[-1]/abs_errs[0]
        rel_errs.append(rel_err)

    nonlinear_solve_info['abs_errs'] = abs_errs
    nonlinear_solve_info['rel_errs'] = rel_errs

    return prims_n, nonlinear_solve_info

def solve_linear(
        primitive_tree: layout.PrimitiveTree,
        constraints: ConstraintLabelledList,
        constraint_graph: IntGraph
    ) -> typ.Tuple[PrimLabelledList, SolverInfo]:
    """
    Return a set of primitives that satisfy the (linearized) constraints

    Parameters
    ----------
    prims: PrimLabelledList
        The list of primitives
    constraints: ConstraintLabelledList
        The list of constraints
    constraint_graph: IntGraph
        A mapping from each constraint to the primitives it applies to

        For example, `constraint_graph[0] == (0, 5, 8)` means the first
        constraint applies to primitives `(prims[0], prims[5], prims[8])`.

    Returns
    -------
    PrimLabelledList
        The list of primitives satisfying the (linearized) constraints
    SolverInfo
        Information about the solve

        Keys are:
            'err': the least squares solver error
            'rank': the rank of the linearized constraint problem
            's': a matrix of singular values for the problem
            'num_dof': the number of degrees of freedom in the problem
            'res': The global constraint residual vector
    """

    # `prim_idx_bounds` stores the right/left indices for each primitive's
    # parameter vector in the global parameter vector array
    # For primitive with index `n`, for example,
    # `prim_idx_bounds[n], prim_idx_bounds[n+1]` are the indices between which
    # the parameter vectors are stored.
    prims = primitive_tree.prims()
    prim_sizes = [prim.param.size for prim in prims]
    prim_idx_bounds = np.cumsum([0] + prim_sizes)

    global_param_n = np.concatenate([prim.param for prim in prims])

    def assem_global_res(global_param):
        new_prim_params = [
            global_param[idx_start:idx_end]
            for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
        ]
        new_tree = layout.build_tree(
            primitive_tree, primitive_tree.prim_graph(), new_prim_params, {}
        )
        new_prims = new_tree.prims()
        constraint_vals = []
        for constraint_idx, prim_idxs in enumerate(constraint_graph):
            constraint = constraints[constraint_idx]
            local_prims = tuple(new_prims[idx] for idx in prim_idxs)

            constraint_vals.append(constraint(local_prims))
        return jnp.concatenate(constraint_vals)

    global_res = assem_global_res(global_param_n)
    global_jac = jax.jacfwd(assem_global_res)(global_param_n)

    dglobal_param, err, rank, s = np.linalg.lstsq(global_jac, -global_res, rcond=None)
    global_param_n = global_param_n + dglobal_param
    solver_info = {
        'err': err, 'rank': rank, 's': s, 'num_dof': len(global_param_n),
        'res': global_res
    }

    ## Build a list of primitives from a global parameter vector
    new_prim_params = [
        np.array(global_param_n[idx_start:idx_end])
        for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
    ]
    new_tree = layout.build_tree(
        primitive_tree, primitive_tree.prim_graph(), new_prim_params, {}
    )

    return new_tree, solver_info
