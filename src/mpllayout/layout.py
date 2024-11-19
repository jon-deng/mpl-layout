"""
Layout class and associated utilities

A `Layout` is the class used to represent an arrangement (layout) of figure
elements.
"""

from typing import Optional, Any
from numpy.typing import NDArray

import warnings

import numpy as np

from . import primitives as pr
from . import constraints as cr
from .containers import Node, ItemCounter, iter_flat, flatten, unflatten

IntGraph = list[tuple[int, ...]]
StrGraph = list[tuple[str, ...]]


class Layout:
    """
    A collection of geometric primitives and constraints

    A `Layout` stores a collection of geometric primitives and the constraints
    on those primitives to represent an arrangement of figure elements.

    Parameters
    ----------
    root_prim: Optional[pr.PrimitiveNode]
        A root primitive to store all geometric primitives
    root_constraint: Optional[cr.ConstraintNode]
        A root constraint to store all geometric constraints
    root_prim_keys: Optional[cr.PrimKeysNode]
        A root primitive arguments tree for `root_constraint`

        Each value in `root_prim_keys` is a tuple of primitive keys
        for the corresponding constraint.
        The tuple of primitive keys indicate primitives from `root_prim`.
    root_param: Optional[cr.ParamsNode]
        A root parameter tree for `root_constraint`

        Each value in `root_param` is a constraint residual parameter
        for the corresponding constraint in `root_constraint`.
    constraint_counter: Optional[ItemCounter]
        An item counter used to create automatic keys for constraints

        This does not have to be supplied.
        Supplying it will change any automatically generated constraint keys.

    Attributes
    ----------
    root_prim: pr.PrimitiveNode
        See `Parameters`
    root_constraint: cr.ConstraintNode
        See `Parameters`
    root_prim_keys: cr.PrimKeysNode
        See `Parameters`
    root_param: cr.ParamsNode
        See `Parameters`
    """

    def __init__(
        self,
        root_prim: Optional[pr.PrimitiveNode] = None,
        root_constraint: Optional[cr.ConstraintNode] = None,
        root_prim_keys: Optional[cr.PrimKeysNode] = None,
        root_param: Optional[cr.ParamsNode] = None,
        constraint_counter: Optional[ItemCounter] = None,
    ):

        if root_prim is None:
            root_prim = pr.PrimitiveNode(np.array([]), {})
        if root_constraint is None:
            root_constraint = cr.ConstraintNode(None, {})
        if root_prim_keys is None:
            root_prim_keys = cr.PrimKeysNode(None, {})
        if root_param is None:
            root_param = cr.ParamsNode(None, {})
        if constraint_counter is None:
            constraint_counter = ItemCounter()

        self._root_prim = root_prim
        self._root_constraint = root_constraint
        self._root_constraint_prim_keys = root_prim_keys
        self._root_constraint_param = root_param

        self._constraint_key_counter = constraint_counter

    @property
    def root_prim(self):
        return self._root_prim

    @property
    def root_constraint(self):
        return self._root_constraint

    @property
    def root_prim_keys(self):
        return self._root_constraint_prim_keys

    @property
    def root_param(self) -> Node[str, Node]:
        return self._root_constraint_param

    def flat_constraints(self):
        # The `[1:]` removes the 'root' constraint which is just a container
        constraints = [
            node for _, node in iter_flat('', self.root_constraint)
        ][1:]
        constraints_argkeys = [
            node.value for _, node in iter_flat('', self.root_prim_keys)
        ][1:]
        constraints_param = [
            node.value for _, node in iter_flat('', self.root_param)
        ][1:]

        constraints = [c.assem_res_atleast_1d for c in constraints]


        return constraints, constraints_argkeys, constraints_param

    def add_prim(self, prim: pr.Primitive, key: str):
        """
        Add a primitive to the layout

        The primitive will be added to `self.root_prim` under the given `key`.

        Parameters
        ----------
        prim: pr.Primitive
            The primitive to add
        key: str
            The key for the primitive
        """
        self.root_prim.add_child(key, prim)

    def add_constraint(
        self,
        constraint: cr.Constraint,
        prim_keys: tuple[str, ...],
        param: tuple[Any],
        key: str = ""
    ):
        """
        Add a constraint between primitives

        Parameters
        ----------
        constraint:
            The constraint to apply
        prim_labels:
            A tuple of strings referencing primitives (`self.root_prim`)
        """
        nodes = (
            self.root_constraint,
            self.root_prim_keys,
            self.root_param
        )
        if key == "":
            key = self._constraint_key_counter.add_item_to_nodes(constraint, *nodes)
        self.root_constraint.add_child(key, constraint)
        self.root_prim_keys.add_child(key, constraint.root_prim_keys(prim_keys))
        self.root_param.add_child(key, constraint.root_params(param))

def build_prim_graph(
    root_prim: pr.Primitive,
) -> tuple[dict[str, int], list[pr.Primitive]]:
    """
    Return a map from flat keys to indices in a list of unique primitives
    """
    prims = list(set(prim for _, prim in iter_flat("", root_prim)))
    prim_to_idx = {prim: ii for ii, prim in enumerate(prims)}

    prim_graph = {key: prim_to_idx[prim] for key, prim in iter_flat("", root_prim)}

    return prim_graph, prims


def build_tree(
    root_prim: pr.Primitive, prim_graph: dict[str, int], values: list[NDArray]
) -> pr.Primitive:
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


import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.axis import Axis, XAxis, YAxis


def update_layout_constraints(
        layout: Layout,
        axs: dict[str, Axes]
    ) -> cr.ParamsNode:
    """
    Update layout constraints which depend on matplotlib elements

    For example x and y axis dimensions depend on the size of tick labels
    from generated figures.
    """
    constraintkey_to_param = {}
    for key, constraint in iter_flat('', layout.root_constraint):
        # `key[1:]` removes the initial "/" from the key
        key = key[1:]
        if key != "":
            prim_keys = layout.root_prim_keys[key]
            constraint_param = layout.root_param[key]

            if isinstance(constraint, cr.XAxisHeight):
                axis_key, = prim_keys.value
                axes_key = axis_key.split("/", 1)[0]
                constraintkey_to_param[key] = (axs[axes_key].xaxis,)

            if isinstance(constraint, cr.YAxisWidth):
                axis_key, = prim_keys.value
                axes_key = axis_key.split("/", 1)[0]
                constraintkey_to_param[key] = (axs[axes_key].yaxis,)

    update_params(
        layout.root_constraint,
        layout.root_param,
        constraintkey_to_param
    )

    return layout

def update_params(
    root_constraint: cr.ConstraintNode,
    root_param: cr.ParamsNode,
    constraintkey_to_param: dict[str, cr.ResParams]
) -> cr.ParamsNode:

    for key, param in constraintkey_to_param.items():
        constraint = root_constraint[key]
        root_param[key] = constraint.root_params(param)

    return root_param
