
"""
Geometric constraints
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np

import jax
import jax.numpy as jnp

from .primitive import Primitive, Point

Prims = typ.Tuple[Primitive, ...]

Idxs = typ.Tuple[int]


class Constraint:
    """
    Constraint base class
    """
    
    primitive_types: typ.Tuple[typ.Type[Primitive], ...]

    def __call__(self, prims: typ.Tuple[Primitive, ...]):
        # Check the input primitives are valid
        assert len(prims) == len(self.primitive_types)
        for prim, prim_type in zip(prims, self.primitive_types):
            assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))
    
    def assem_res(self, prims: typ.Tuple[Primitive, ...]) -> NDArray:
        raise NotImplementedError()

class PointToPointAbsDistance(Constraint):

    primitive_types = (Point, Point)

    def __init__(
            self, distance: float, direction: typ.Optional[NDArray]=None
        ):
        if direction is None:
            direction = np.array([1, 0])
        else:
            direction = np.array(direction)

        self._direction = direction
        self._distance = distance

    @property
    def distance(self):
        return self._distance
    
    @property
    def direction(self):
        return self._direction

    def assem_res(self, prims):
        return jnp.dot(prims[1].param - prims[0].param, self.direction) - self.distance
    
class PointLocation(Constraint):

    primitive_types = (Point, )

    def __init__(
            self, 
            location: NDArray
        ):
        self._location = location

    def assem_res(self, prims):
        return prims[0].param - self._location

ConstraintGraph = typ.List[Idxs]

def depack_internal_constraints(prims: Prims, global_prim_idx=0):

    intern_prims = []
    intern_constraints = []
    intern_constraint_graph = []

    m = global_prim_idx
    for prim in prims:
        if prim.prims == ():
            sub_prims = prim.prims
            sub_constraints = prim.constraints
            sub_constraint_graph = [
                tuple(idx + m for idx in idxs)
                for idxs in prim.constraint_graph
            ]
        else:
            sub_prims, sub_constraints, sub_constraint_graph = depack_internal_constraints(prim.prims, m)
        
        m = m+len(sub_prims)
        intern_prims += sub_prims
        intern_constraints += sub_constraints
        intern_constraint_graph += sub_constraint_graph

    return intern_prims, intern_constraints, intern_constraint_graph

def expand_prim(prim: Primitive, prim_idx=0):
    exp_prims = [prim] + list(prim.prims)
    exp_constrs = list(prim.constraints)
    exp_constr_graph = [
        (idx+prim_idx for idx in idxs) 
        for idxs in prim.constraint_graph
    ]

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
    exp_prims = []
    exp_constraints = []
    exp_constraint_graph = []
    
    for _prim in prims:
        m = len(exp_prims) + len(prims)
        _exp_prims, _exp_constrs, _exp_constraint_graph = expand_prim(_prim, prim_idx=m)
        exp_prims += _exp_prims[1:]
        exp_constraints += _exp_constrs
        exp_constraint_graph += _exp_constraint_graph

    prims += exp_prims
    constraints += exp_constraints
    constraint_graph += exp_constraint_graph

    int_prims, int_constraints, int_constraint_graph = depack_internal_constraints(prims, global_prim_idx=len(prims))
    prims += int_prims
    constraints += int_constraints
    constraint_graph += int_constraint_graph

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
