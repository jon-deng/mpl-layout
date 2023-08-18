"""Primitive
Routines for solving and managing collections of primitives and constraints
"""

import typing as typ

import jax
from jax import numpy as jnp
import numpy as np

from . import geometry as geo

PrimList = typ.List[geo.Primitive | geo.PrimitiveArray]
ConstraintList = typ.List[geo.Constraint]
Idxs = typ.Tuple[int]
Graph = typ.List[Idxs]
SolverInfo = typ.Mapping[str, typ.Any]

class Layout:
    """
    Represents a collection of primitives and associated constraints

    Parameters
    ----------
    prims: typ.Optional[PrimList]
        A list of primitives
    constraints: typ.Optional[ConstraintList]
        A list of constraints
    constraint_graph: typ.Optional[Graph]
        A constraint graph
    """

    def __init__(
            self, 
            prims: typ.Optional[PrimList]=None, 
            constraints: typ.Optional[ConstraintList]=None, 
            constraint_graph: typ.Optional[Graph]=None
        ):

        if prims is None:
            prims = []
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._prims = LabelIndexedList(prims)
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
            prim: geo.Primitive, 
            prim_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `geo.Primitive` to the `Layout`

        Parameters
        ----------
        constraint: geo.Constraint
            The constraint to apply
        prim_labels: typ.Tuple[typ.Union[str, int], ...]
            Labels for the primitives to apply the constraints on
        prim_idxs: typ.Optional[
                typ.Tuple[typ.Union[int, None], ...]
            ]
            Indices for each primitive if the primitive is a `PrimitiveList` type
        constraint_label: typ.Optional[str]
            An optional label for the constraint

        Returns
        -------
        prim_label: str
            The label for the added primitive
        """
    
        # Append the root primitive
        prim_label = self.prims.append(prim, label=prim_label)

        # Append all child primitives
        if len(prim.prims) > 0:
            subprims, subconstrs, subconstr_graph = \
                expand_prim(prim, prim_idx=len(self.prims)) 
            
            prim_labels = expand_prim_labels(prim, prim_label)
            for sub_label, sub_prim in zip(prim_labels, subprims):
                self.prims.append(sub_prim, label=sub_label)

            for constr, prim_idxs in zip(subconstrs, subconstr_graph):
                self.add_constraint(constr, prim_idxs)
            
        return prim_label

    def add_constraint(
            self, 
            constraint: geo.Constraint, 
            prim_labels: typ.Tuple[typ.Union[str, int], ...],
            prim_sub_idxs: typ.Optional[
                typ.Tuple[typ.Union[int, None], ...]
            ]=None,
            constraint_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `Constraint` between `Primitive`s

        Parameters
        ----------
        constraint: geo.Constraint
            The constraint to apply
        prim_labels: typ.Tuple[typ.Union[str, int], ...]
            Labels for the primitives the constraint applies to
        prim_sub_idxs: typ.Optional[
                typ.Tuple[typ.Union[int, None], ...]
            ]
            Indices for any `PrimitiveArray`s
        constraint_label: typ.Optional[str]
            An optional label for the constraint

        Returns
        -------
        constraint_label: str
            The label for the added constraint
        """
        # These are prims/prim indices the constraint applies to
        prims = tuple(self.prims[label] for label in prim_labels)
        prim_idxs = tuple(self.prims.key_to_idx(label) for label in prim_labels)

        # If any primitive is a `PrimitiveArray` type and is being indexed, then 
        # we have to modify the constraint and the primitives it applies to
        # to account for the sub-index
        if prim_sub_idxs is None:
            prim_sub_idxs = len(prim_labels)*(None,)

        if len(prim_sub_idxs) != len(prim_labels):
            raise ValueError("There must be as many sub-indices as primitives")
        
        # Depack all the sub-index specs for indexed `PrimitiveArray`s
        # Each element of the `prim_sub_idx_specs` is a tuple consisting of:
        # a function that takes a primitive tuple and returns the primitive being indexed
        # and the indices of child primitives that are input to that function to get the primitive
        prim_sub_idx_specs = tuple(
            (None, None) if sub_idx is None
            else prim.index_spec(sub_idx)
            for prim, sub_idx in zip(prims, prim_sub_idxs)
        )
        make_prims = tuple(spec[0] for spec in prim_sub_idx_specs)
        child_idxs = tuple(spec[1] for spec in prim_sub_idx_specs)

        # Modify the primitive indices to account for sub-indexes
        # This is done by the offsetting the root primitive index by any child 
        # index offsets from `child_idxs`
        new_prim_idxs = []
        for make_prim, child_idx, prim_idx in zip(make_prims, child_idxs, prim_idxs):
            if make_prim is None:
                new_prim_idxs = new_prim_idxs + [prim_idx]
            else:
                new_prim_idxs = new_prim_idxs + [prim_idx+ii for ii in child_idx]
        new_prim_idxs = tuple(new_prim_idxs)

        make_prim_arg_lengths = tuple(
            1 if sub_idx is None else len(sub_idx) 
            for sub_idx in child_idxs
        )

        # Modify the constraint function to account for sub-indexing
        def new_constraint(*args):
            arg_bounds = [0] + np.cumsum(make_prim_arg_lengths).tolist()
            prims = tuple(
                args[start] if make_prim is None
                else make_prim(*args[start:end]) 
                for make_prim, start, end in zip(make_prims, arg_bounds[:-1], arg_bounds[1:])
            )
            return constraint(prims)

        constraint_label = self.constraints.append(new_constraint, label=constraint_label)
        self.constraint_graph.append(new_prim_idxs)
        return constraint_label


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
            self._items = []
        else:
            self._items = items
        if keys is None:
            self._label_to_idx = {}
        else:
            self._label_to_idx = {key: idx for idx, key in enumerate(keys)}

        # Store the total number of items of each type
        self._type_to_count = Counter()

    def __repr__(self):
        return f"LabelIndexedList({list(self.values())}, {list(self.keys())})"
    
    def __str__(self):
        return str(self._items)

    ## List/Dict interface
    def __len__(self):
        return len(self._items)

    def __getitem__(self, key: typ.Union[str, int]) -> T:
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
        self._type_to_count.add(ItemType)

        if label is None:
            n =  self._type_to_count[ItemType] - 1
            label = f'{ItemType.__name__}{n:d}'
        
        assert label not in self._label_to_idx
        self._items.append(item)
        self._label_to_idx[label] = len(self._items)-1
        return label
    
    def key_to_idx(self, key: typ.Union[str, int, slice]):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self._label_to_idx[key]
        else:
            raise TypeError(f"`key` must be `str` or `int`, not `{type(key)}`")

def expand_prim_labels(
        prim: geo.Primitive,
        prim_label: str,
        # prims: typ.List[geo.Primitive],
        # prim_graph: typ.List[int]
    ):
    num_child = len(prim.prims)

    labels = []
    type_to_count = Counter()
    for subprim in prim.prims:
        PrimType = type(subprim)
        type_to_count.add(PrimType)
        n = type_to_count[PrimType] - 1

        labels.append(f'{prim_label}.{PrimType.__name__}{n:d}')

    # Recursively expand any child primitives
    if num_child == 0:
        return []
    else:
        for sub_prim, prefix in zip(prim.prims, labels):
            labels = labels + expand_prim_labels(sub_prim, prefix)
        return labels
    
def expand_prim(
        prim: geo.Primitive, 
        prim_idx: typ.Optional[int]=0
    ):
    """
    Expand all child primitives and constraints of `prim`

    Parameters
    ----------
    prim: geo.Primitive
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
        tuple(idx+prim_idx for idx in idxs) 
        for idxs in prim.constraint_graph
    ]

    # Recursively expand any child primitives
    if len(prim.prims) == 0:
        return child_prims, child_constrs, child_constr_graph
    else:
        for sub_prim in prim.prims:
            _exp_prims, _exp_constrs, _exp_constr_graph = expand_prim(sub_prim)
            child_prims += _exp_prims
            child_constrs += _exp_constrs
            child_constr_graph += _exp_constr_graph
        return child_prims, child_constrs, child_constr_graph
    
def contract_prim(
        prim: geo.Primitive, 
        prims: PrimList
    ):
    """
    Collapse all child primitives into `prim`

    Parameters
    ----------
    prim: geo.Primitive
        The primitive to be expanded
    prim_idx: int
        The index of the primitive in a global list of primitives

    Returns
    -------
    """
    num_child = len(prim.prims)

    child_prims = []
    m = num_child
    for subprim in prims[:num_child]:
        cprim, n = contract_prim(subprim, prims[m:])

        child_prims.append(cprim)
        m += n

    return type(prim)(param=prim.param, prims=tuple(child_prims)), m
   
def build_prims(prims, params):
    
    _new_prims = [
        type(prim)(param=param, prims=prim.prims) 
        for prim, param in zip(prims, params)
    ]

    # Contract all child primitives into parents
    m = 0
    new_prims = []
    while m < len(_new_prims):
        prim, dm = contract_prim(_new_prims[m], _new_prims[m+1:])
        cprims, *_ = expand_prim(prim)
        new_prims = new_prims + [prim] + cprims
        m += 1+dm

    return new_prims

def solve(
        prims: typ.List[geo.Primitive], 
        constraints: typ.List[geo.Constraint], 
        constraint_graph: Graph
    ) -> typ.Tuple[PrimList, SolverInfo]:
    """
    Return a set of primitives that satisfy the given constraints

    Parameters
    ----------
    prims: typ.List[geo.Primitive]
        The list of primitives
    constraints: typ.List[geo.Constraint]
        The list of constraints
    constraint_graph: Graph
        A mapping from each constraint to the primitives it applies to.
        For example, `constraint_graph[0] == (0, 5, 8)` means the first constraint
        applies to primitives `(prims[0], prims[5], prims[8])`.
    prim_graph: Graph
        Denotes that the next block of primitives parameterize the current primitive.
        For example, `subprim_graph[0] == 4` means `prim[0]` is parameterized 
        by primitives `prims[1:5]`. Primitives that have no sub-primitives have 
        `subprim_graph[n] == 0`.
    """

    # `prim_idx_bounds` stores the right/left indices for each primitive's 
    # parameter vector in the global parameter vector array
    # For primitive with index `n`, for example, 
    # `prim_idx_bounds[n], prim_idx_bounds[n+1]` are the indices between which
    # the parameter vectors are stored.
    prim_sizes = [prim.param.size for prim in prims]
    prim_idx_bounds = np.cumsum([0] + prim_sizes)

    global_param_n = np.concatenate([prim.param for prim in prims])

    def assem_global_res(global_param):
        new_prim_params = [
            global_param[idx_start:idx_end]
            for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
        ]
        new_prims = build_prims(prims, new_prim_params)
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
    new_prims = build_prims(prims, new_prim_params)
    new_prims = LabelIndexedList(new_prims, list(prims.keys()))

    return new_prims, solver_info

class Counter:

    def __init__(self):
        self._count = {}

    @property
    def count(self):
        return self._count
    
    def __in__(self, key):
        return key in self.count
    
    def __getitem__(self, key):
        return self.count.get(key, 0)
    
    def add(self, item):
        if item in self.count:
            self.count[item] += 1
        else:
            self.count[item] = 1
