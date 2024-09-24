"""
Classes for handling collections of primitives and constraints

The strategy to solve a collection of constraints is to use three labelled
lists representing a system of non-linear equations:
    `primitives: `
        A collection of `geo.Primitive` instances

        These represent the unknowns parameters of the geometric system and is
        represented as a tree.
    `constraints: tp.List[geo.Constraint]`
        A list of `geo.Constraint` instances representing the non-linear
        equations

        Each `geo.Constraint.assem_res' represents the non-linear equation(s)
        that must be satisfied for the constraint.
    `constraint_graph: StrGraph`
        A graph linking each constraint to the primitives in `primitives`
        (through string labels) the constraint applies to

The class `Layout` handles construction of these three things while functions
`solve` and `solve_linear` use the layout to solve the system of constraints.
"""

import typing as tp
from numpy.typing import NDArray

import numpy as np

from . import geometry as geo
from .containers import Node, iter_flat, flatten, unflatten

IntGraph = tp.List[tp.Tuple[int, ...]]
StrGraph = tp.List[tp.Tuple[str, ...]]


class Layout:
    """
    A constrained layout of primitives

    The class holds a collection of primitives, constraints on those primitives, and the
    graph between constraints and primitives they apply to.

    Parameters
    ----------
    root_prim:
        A `Node` container of `geo.Primitive`s
    constraints:
        A list of constraints
    constraint_graph:
        A list of keys indicating which primitives each constraint applies to
    """

    def __init__(
        self,
        root_prim: tp.Optional[Node] = None,
        constraints: tp.Optional[tp.List[geo.Constraint]] = None,
        constraint_graph: tp.Optional[StrGraph] = None,
    ):

        if root_prim is None:
            root_prim = Node(np.array([]), [], [])
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._root_prim = root_prim
        self._constraints = constraints
        self._constraint_graph = constraint_graph

        self._prim_type_count = {}
        self._label_to_primidx = {}

        self._constraint_type_count = {}
        self._label_to_constraintidx = {}

    @property
    def root_prim(self):
        return self._root_prim

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraint_graph(self) -> StrGraph:
        return self._constraint_graph

    def add_prim(self, prim: geo.Primitive, key: str):
        """
        Add a primitive to the layout

        The primitive will be added to `self.root_prim` under the given `key`.

        Parameters
        ----------
        prim: geo.Primitive
            The primitive to add
        key: str
            The key for the primitive
        """
        self.root_prim.add_child(key, prim)

    def add_constraint(self, constraint: geo.Constraint, prim_keys: tp.Tuple[str, ...]):
        """
        Add a constraint between primitives

        Parameters
        ----------
        constraint:
            The constraint to apply
        prim_labels:
            A tuple of strings referencing primitives (`self.root_prim`)
        """
        self.constraints.append(constraint)
        self.constraint_graph.append(prim_keys)


def build_prim_graph(
    root_prim: geo.Primitive,
) -> tp.Tuple[tp.Mapping[str, int], tp.List[geo.Primitive]]:
    """
    Return a map from flat keys to indices in a list of unique primitives
    """
    prims = list(set(prim for _, prim in iter_flat("", root_prim)))
    prim_to_idx = {prim: ii for ii, prim in enumerate(prims)}

    prim_graph = {key: prim_to_idx[prim] for key, prim in iter_flat("", root_prim)}

    return prim_graph, prims


def build_tree(
    root_prim: geo.Primitive, prim_graph: tp.Mapping[str, int], values: tp.List[NDArray]
) -> geo.Primitive:
    """
    Return a new tree where child node values have been updated

    Parameters
    ----------
    root_prim:
        The old tree
    prim_graph:
        A mapping from keys in `root_prim` to corresponding new values in `values`
    values:
        A list of new parameter values to build a new tree with

    Returns
    -------
    Node
        The new tree with values from `values`
    """
    old_prim_structs = flatten("", root_prim)

    # `key[1:]` remove the initial forward slash from the flat keys
    new_prim_values = [values[prim_graph[key]] for key, _ in iter_flat("", root_prim)]

    new_prim_structs = [
        (*old_struct[:2], new_value, *old_struct[3:])
        for old_struct, new_value in zip(old_prim_structs, new_prim_values)
    ]
    return unflatten(new_prim_structs)[0]
