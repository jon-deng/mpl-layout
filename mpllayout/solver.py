"""
Routines for solving and managing collections of primitives and constraints
"""

import typing as typ

import jax
from jax import numpy as jnp
import numpy as np

from .constraint import Constraint
from .primitive import Primitive

Prims = typ.Tuple[Primitive, ...]
Idxs = typ.Tuple[int]
ConstraintGraph = typ.List[Idxs]

def expand_prim(prim: Primitive, prim_idx=0):

    # Unpack the sub-primitives, constraints, and constraint graph
    exp_prims = list(prim.prims)
    exp_constrs = list(prim.constraints)
    exp_constr_graph = [
        (idx+prim_idx for idx in idxs) 
        for idxs in prim.constraint_graph
    ]

    # Recursively unpack any other sub-primitives
    if len(prim.prims) == 0:
        return exp_prims, exp_constrs, exp_constr_graph
    else:
        for sub_prim in prim.prims:
            _exp_prims, _exp_constrs, _exp_constr_graph = expand_prim(sub_prim)
            exp_prims += _exp_prims
            exp_constrs += _exp_constrs
            exp_constr_graph += _exp_constr_graph
        return exp_prims, exp_constrs, exp_constr_graph

def solve(
        prims: typ.List[Primitive], 
        constraints: typ.List[Constraint], 
        constraint_graph: ConstraintGraph
    ):
    # int_prims = []
    # int_constraints = []
    # int_constraint_graph = []
    
    # for _prim in prims:
    #     # Set `m` so that all internal primitives + constraints are appended to 
    #     # the end of the supplied `prims` list
    #     m = len(int_prims) + len(prims)
    #     _prims, _constrs, _constraint_graph = expand_prim(_prim, prim_idx=m)
    #     int_prims += _prims[:]
    #     int_constraints += _constrs
    #     int_constraint_graph += _constraint_graph

    # prims += int_prims
    # constraints += int_constraints
    # constraint_graph += int_constraint_graph

    prim_sizes = [prim.param.size for prim in prims]
    prim_global_idx_bounds = np.cumsum([0] + prim_sizes)

    global_param_n = np.concatenate([prim.param for prim in prims])

    def assem_global_res(global_param):
        constraint_vals = []
        for constraint_idx, prim_idxs in enumerate(constraint_graph):
            constraint = constraints[constraint_idx]
            params = tuple(
                global_param[
                    prim_global_idx_bounds[idx]:prim_global_idx_bounds[idx+1]
                ] 
                for idx in prim_idxs
            )
            local_prims = tuple(
                Prim(param) for Prim, param in zip(constraint.primitive_types, params)
            )

            constraint_vals.append(constraint(local_prims))
        return jnp.concatenate(constraint_vals)
        
    global_res = assem_global_res(global_param_n)
    global_jac = jax.jacfwd(assem_global_res)(global_param_n)

    dglobal_param, err, rank, s = np.linalg.lstsq(global_jac, -global_res, rcond=None)
    global_param_n = global_param_n + dglobal_param
    solver_info = {
        'err': err, 'rank': rank, 's': s
    }

    new_prim_params = [
        np.array(global_param_n[idx_start:idx_end])
        for idx_start, idx_end in zip(prim_global_idx_bounds[:-1], prim_global_idx_bounds[1:])
    ]

    return new_prim_params, solver_info
