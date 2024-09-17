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

import numpy as np

from . import geometry as geo
from .containers import Node

IntGraph = tp.List[tp.Tuple[int, ...]]
StrGraph = tp.List[tp.Tuple[str, ...]]

class Layout:
    """
    Class used to handle a collection of primitives and associated constraints

    The class contains functions to add primitives to the collection, add
    constraints on those primitives, and create the associated graph between
    constraints and primitives.

    Parameters
    ----------
    prims:
        A `Node` container of `geo.Primitive`s
    constraints:
        A list of constraints
    constraint_graph:
        A constraint graph
    """

    def __init__(
        self,
        prims: tp.Optional[Node] = None,
        constraints: tp.Optional[tp.List[geo.Constraint]] = None,
        constraint_graph: tp.Optional[StrGraph] = None,
    ):

        if prims is None:
            prims = Node(np.array([]), [], [])
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._prims = prims
        self._constraints = constraints
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
    def constraint_graph(self) -> StrGraph:
        return self._constraint_graph

    def add_prim(self, prim: geo.Primitive, key: tp.Optional[str] = None) -> str:
        """
        Add a `geo.Primitive` to the `Layout`

        The primitive will be added `self.prims` under the given `label`.

        Parameters
        ----------
        prim: geo.Primitive
            The primitive to add
        label: tp.Optional[str]
            An optional label for the primitive

            If not provided, an automatic name based on the primitive class will
            be used.

        Returns
        -------
        label: str
            The label for the added primitive
        """
        self.prims.add_child(key, prim)
        return key

    def add_constraint(
        self,
        constraint: geo.Constraint,
        prim_labels: tp.Tuple[str, ...]
    ) -> None:
        """
        Add a `geo.Constraint` between `geo.Primitive`s

        Parameters
        ----------
        constraint:
            The constraint to apply
        prim_labels:
            A tuple of strings referencing primitives (`self.prims`) to apply the constraint
        """
        self.constraints.append(constraint)
        self.constraint_graph.append(prim_labels)

def build_prim_graph(prim: geo.Primitive) -> tp.Mapping[geo.Primitive, int]:
    """
    Return a mapping from primitives to integer indices in `self.prims()`
    """
    _graph = {tree.value: None for tree in prim.values(flat=True)}
    return {prim: ii for ii, prim in enumerate(_graph)}

def build_constraint_graph_int(self) -> IntGraph:
    prim_graph = self.prims.prim_graph()
    return [
        tuple(prim_graph[self.prims[prim_label]] for prim_label in prim_labels)
        for prim_labels in self.constraint_graph
    ]

def prims(self) -> tp.List[geo.Primitive]:
    """
    Return a list of all unique primitives in the tree
    """
    return list(self.prim_graph().keys())

def build_tree(
    prim: geo.Primitive,
    prim_to_idx: tp.Mapping[geo.Primitive, int],
    params: tp.List[np.typing.NDArray],
    prim_to_newprim: tp.Mapping[geo.Primitive, geo.Primitive],
) -> geo.Primitive:
    """
    Return a new `PrimitiveTree` using new primitives for given parameter values

    Parameters
    ----------
    tree:
        The old `PrimitiveTree` instance
    prim_to_idx:
        A mapping from `geo.Primitive` in `tree` to corresponding parameters in `params`

        If `params` is a list with parameters in order from `tree.prims()`, then this
        corresponds to `tree.prim_graph()`.
    params:
        A list of parameter values to build a new `PrimitiveTree` with
    prim_to_newprim:
        A mapping from `geo.Primitive`s in `tree` to replacement `Primitives` in the new tree

        This should be an empty dictionary in the root `build_tree` call. As
        `build_tree` builds the new tree, it will populate this dictionary to preserve
        the mapping of primitives.

    Returns
    -------
    PrimitiveTree
        The new `PrimitiveTree` with parameters from `params`
    """
    oldprim = prim.value

    # If the tree has no children, we simply create the primitive with no children
    # If the tree has children, we need to recursively create all child trees
    if len(prim.children) == 0:
        children = {}
    else:
        children = {
            key: build_tree(childtree, prim_to_idx, params, prim_to_newprim)
            for key, childtree in prim.children.items()
        }

    if oldprim is None:
        newprim = None
    elif oldprim in prim_to_newprim:
        newprim = prim_to_newprim[oldprim]
    else:
        param = params[prim_to_idx[oldprim]]
        newprim = type(oldprim)(
            param, tuple(child_tree.value for child_tree in children.values())
        )
        prim_to_newprim[oldprim] = newprim

    return geo.Primitive(newprim, children)
