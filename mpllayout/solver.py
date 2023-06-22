"""
Routines for solving and managing collections of primitives and constraints
"""

import typing as typ

import itertools

import jax
from jax import numpy as jnp
import numpy as np

from .constraint import Constraint
from .primitive import Primitive

Prims = typ.Tuple[Primitive, ...]
Constraints = typ.List[Constraint]
Idxs = typ.Tuple[int]
Graph = typ.List[Idxs]

class ConstrainedPrimitiveManager:
    """
    Manage a set of primitives and constraints

    Parameters
    ----------
    prims: typ.Optional[Prims]
        A list of primitives
    constraints: typ.Optional[Constraints]
        A list of constraints
    constraint_graph: typ.Optional[Graph]
        A constraint graph
    """

    def __init__(
            self, 
            prims: typ.Optional[Prims]=None, 
            constraints: typ.Optional[Constraints]=None, 
            constraint_graph: typ.Optional[Graph]=None
        ):

        if prims is None:
            prims = []
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._prims = LabelIndexedList(constraints)
        self._constraints = LabelIndexedList(constraints)
        self._constraint_graph = constraint_graph

        self._prim_type_count = {}
        self._label_to_primidx = {}

        self._constraint_type_count = {}
        self._label_to_constraintidx = {}

    @property
    def prims(self):
        return self._prims
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def constraint_graph(self):
        return self._constraint_graph

    def add_prim(
            self, 
            prim: Primitive, 
            prim_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a primitive to the collection
        """
        subprims, subprim_graph, subconstrs, subconstr_graph = \
            expand_prim(prim, prim_idx=len(self.prims)) 
        
        # Append the root primitive
        prim_label = self.prims.append(prim, label=prim_label)

        # Append all child primitives
            
        return prim_label

    def add_constraint(
            self, 
            constraint: Constraint, 
            prim_labels: typ.Tuple[typ.Union[str, int], ...], 
            constraint_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a constraint between primitives
        """
        constraint_label = self.constraints.append(constraint, label=constraint_label)
        prim_idxs = tuple(self.prims.key_to_idx(label) for label in prim_labels)
        self.constraint_graph.append(prim_idxs)
        return constraint_label

def expand_prim_labels(
        prim_label: str,
        prims: typ.List[Primitive], 
        prim_graph: typ.List[int]
    ):
    num_child = prim_graph[0]

    labels = []
    type_to_count = {}
    for prim in prims[:num_child]:
        PrimType = type(prim)
        if PrimType not in type_to_count:
            type_to_count[PrimType] = 0
        
        n = type_to_count[PrimType]
        type_to_count[PrimType] += 1

        labels.append(f'{prim_label}.{PrimType.__name__}{n:d}')

    # print(labels)

    # Recursively expand any child primitives
    if num_child == 0:
        return []
    else:
        for idx, (sub_prim, prefix) in enumerate(zip(prims[:num_child], labels)):
            labels = labels + expand_prim_labels(prefix, sub_prim.prims, prim_graph[idx:])
        return labels


T = typ.TypeVar('T')
class LabelIndexedList(typ.Generic[T]):
    """
    A list with automatically generated labels for indices
    """
    def __init__(
            self, 
            items: typ.Optional[typ.List[T]]=None, 
            keys: typ.Optional[typ.List[str]]=None
        ):
        if items is None:
            items = []
        self._items = items
        # Store the total number of items of each type
        self._type_to_count = {}
        self._label_to_idx = {}

    ## List/Dict interface
    def __len__(self):
        return len(self._items)

    def __getitem__(self, key: typ.Union[str, int]):
        key = self.key_to_idx(key)
        return self._items[key]
    
    def keys(self):
        return self._label_to_idx.keys()
    
    def values(self):
        return self._items
    
    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def append(self, item: T, label: typ.Optional[str]=None) -> str:
        ItemType = type(item)
        if ItemType in self._type_to_count:
            self._type_to_count[ItemType] += 1
        else:
            self._type_to_count[ItemType] = 1

        if label is None:
            n =  self._type_to_count[ItemType] - 1
            label = f'{ItemType.__name__}{n:d}'
        
        assert label not in self._label_to_idx
        self._items.append(item)
        self._label_to_idx[label] = len(self._items)-1
        return label
    
    def key_to_idx(self, key: typ.Union[str, int]):
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return self._label_to_idx[key]
        else:
            raise TypeError(f"`key` must be `str` or `int`, not `{type(key)}`")


def expand_prim(
        prim: Primitive, 
        prim_idx: typ.Optional[int]=0
    ):
    """
    Expand all child primitives and constraints of `prim`

    Parameters
    ----------
    prim: Primitive
        The primitive to be expanded
    prim_idx: int
        The index of the primitive in a global list of primitives

    Returns
    -------
    """

    # Expand child primitives, constraints, and constraint graph
    child_prims = list(prim.prims)
    child_constrs = list(prim.constraints)
    child_constr_graph = [
        (idx+prim_idx for idx in idxs) 
        for idxs in prim.constraint_graph
    ]
    prim_graph = [len(prim.prims)]

    # Recursively expand any child primitives
    if len(prim.prims) == 0:
        return child_prims, child_constrs, child_constr_graph, prim_graph
    else:
        for sub_prim in prim.prims:
            _exp_prims, _exp_constrs, _exp_constr_graph, _prim_graph = expand_prim(sub_prim)
            child_prims += _exp_prims
            child_constrs += _exp_constrs
            child_constr_graph += _exp_constr_graph
            prim_graph += _prim_graph
        return child_prims, child_constrs, child_constr_graph, prim_graph
    
def contract_prim(
        prim: Primitive, 
        prims: Prims,
        prim_graph: typ.List[int]
    ):
    """
    Collapse all child primitives into `prim`

    Parameters
    ----------
    prim: Primitive
        The primitive to be expanded
    prim_idx: int
        The index of the primitive in a global list of primitives

    Returns
    -------
    """
    num_child = prim_graph[0]
    _child_prims = prims[:num_child]
    # This denotes the index where the child primitives' child primitives would 
    # start
    _child_prim_idx = np.cumsum(prim_graph[:num_child])
    child_prims = tuple(
        contract_prim(_prim, prims, prim_graph[m:]) 
        for _prim, m in zip(_child_prims, _child_prim_idx)
    )

    return type(prim)(param=prim.param, prims=child_prims)
   
def solve(
        prims: typ.List[Primitive], 
        constraints: typ.List[Constraint], 
        constraint_graph: Graph,
        prim_graph: Graph
    ):
    """
    Return a set of primitives that satisfy the given constraints

    Parameters
    ----------
    prims: typ.List[Primitive]
        The list of primitives
    constraints: typ.List[Constraint]
        The list of constraints
    constraint_graph: Graph
        A mapping from each constraint to the primitives it applies to.
        For example, `constraint_graph[0] == (0, 5, 8)` means the first constraint
        applies to primitives `(prims[0], prims[5], prims[8])`.
    prim_graph: Graph
        Denotes that the next block of primitives parameterize the current primitive.
        For example, `subprim_graph[0] == 4` means `prim[0]` is parameterized 
        by primitives `prims[1:5]`. Primitives that have no sub-primitives obey 
        `subprim_graph[n] == 0`.
    """

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

    # new_prims = 

    # new_prims = [

    #     for Pirm, param in zip(, new_prim_params)
    # ]

    return new_prim_params, solver_info
