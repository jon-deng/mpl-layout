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

        self._constraint_counter = constraint_counter

    @property
    def root_prim(self) -> pr.PrimitiveNode:
        return self._root_prim

    @property
    def root_constraint(self) -> cr.ConstraintNode:
        return self._root_constraint

    @property
    def root_prim_keys(self) -> cr.PrimKeysNode:
        return self._root_constraint_prim_keys

    @property
    def root_param(self) -> cr.ParamsNode:
        return self._root_constraint_param

    def flat_constraints(
        self
    ) -> tuple[list[cr.Constraint], list[cr.PrimKeys], list[cr.ResParams]]:
        """
        Return flat constraints, primitive argument keys and parameters

        Returns
        -------
        constraints: list[cr.Constraint]
            A flat list of constraints from the root constraint
        prim_keys: list[cr.PrimKeys]
            A list of primitive keys for each constraint
        params: list[cr.ResParams]
            A list of residual parameters for each constraint
        """
        # The `[1:]` removes the 'root' constraint which is just a container
        constraints = [
            node for _, node in iter_flat('', self.root_constraint)
        ][1:]
        prim_keys = [
            node.value for _, node in iter_flat('', self.root_prim_keys)
        ][1:]
        params = [
            node.value for _, node in iter_flat('', self.root_param)
        ][1:]

        constraints = [c.assem_res_atleast_1d for c in constraints]

        return constraints, prim_keys, params

    def add_prim(self, prim: pr.Primitive, key: str):
        """
        Add a primitive to the layout

        The primitive will be added to `self.root_prim` using the given `key`.

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
        prim_keys: cr.PrimKeys,
        param: cr.ResParams,
        key: str = ""
    ):
        """
        Add a constraint between primitives

        Parameters
        ----------
        constraint: cr.Constraint
            The constraint to add
        prim_keys: cr.PrimKeys
            A tuple of primitive keys the constraint applies to

            The primitive keys refer to primitives in `self.root_prim`.
        param: cr.ResParams
            Parameters for the constraint
        key: str
            An optional key to identify the constraint

            If not supplied, a key will be automatically generated using
            `_constraint_counter`.
        """
        nodes = (
            self.root_constraint, self.root_prim_keys, self.root_param
        )
        if key == "":
            key = self._constraint_counter.add_item_to_nodes(constraint, *nodes)
        self.root_constraint.add_child(key, constraint)
        self.root_prim_keys.add_child(key, constraint.root_prim_keys(prim_keys))
        self.root_param.add_child(key, constraint.root_params(param))

# TODO: Potentially move these to better places?
# TODO: Document/rename these based on what they do (have to figure that out again)
def build_prim_graph(
    root_prim: pr.Primitive,
) -> tuple[dict[str, int], list[pr.Primitive]]:
    """
    Return unique primitives from a root primitive and indicate their indices

    Note that primitives in a primitive node are not necessarily unique
    ; for example `Point`s are shared between lines in a polygon.

    When solving a set of geometric constraints, the geometric constraint
    residual should be linked to a function of unique primitives only.

    Returns
    -------
    prim_graph: dict[str, int]
        A mapping from each primitive key to its unique primitive index
    prims: list[pr.Primitive]
        A list of unique primitives
    """
    prims = list(set(prim for _, prim in iter_flat("", root_prim)))
    prim_to_idx = {prim: ii for ii, prim in enumerate(prims)}

    prim_graph = {key: prim_to_idx[prim] for key, prim in iter_flat("", root_prim)}

    return prim_graph, prims


def build_tree(
    root_prim: pr.Primitive, prim_graph: dict[str, int], values: list[NDArray]
) -> pr.Primitive:
    """
    Return a new primitive with values updated from unique values

    Parameters
    ----------
    root_prim: pr.Primitive
        The old primitive
    prim_graph: dict[str, int]
        A mapping from each primitive key to a unique primitive value in `values`
    values: list[NDArray]
        A list of primitive values for unique primitives in `root_prim`

    Returns
    -------
    pr.Primitive
        The new primitive with updated values
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


# TODO: Return an actual new `Layout` object here
# don't just implicity modify the layout?
def update_layout_constraints(
    layout: Layout,
    axs: dict[str, Axes]
) -> Layout:
    """
    Update layout constraints that depend on `matplotlib` elements

    Some constraints have parameters that depend on `matplotlib`.
    This function shoud identify these constraints in a `Layout` and replace
    their parameters with the correct `matplotlib` element.
    Currently this is only implemented for `XAxisHeight` and `YAxisWidth`.

    Parameters
    ----------
    layout: Layout
        The layout
    axs: dict[str, Axes]
        The `matplotlib` axes objects

        The key for every `matplotlib.Axes` should match a corresponding
        `pr.Axes` in the layout.

    Returns
    -------
    Layout
        The layout with updated constraint parameters
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

    update_root_param(
        layout.root_constraint,
        layout.root_param,
        constraintkey_to_param
    )

    return layout

def update_root_param(
    root_constraint: cr.ConstraintNode,
    root_param: cr.ParamsNode,
    constraintkey_to_param: dict[str, cr.ResParams]
) -> cr.ParamsNode:
    """
    Update the root constraint parameters node

    Parameters
    ----------
    root_constraint: cr.ConstraintNode
        The root constraint
    root_param: cr.ParamsNode
        The corresponding root parameters node
    constraintkey_to_param: dict[str, cr.ResParams]
        A mapping of constraint keys to replacement constraint parameters

        Each constraint key should indicate a node in `root_constraint` and
        `root_param`.
    """
    for key, param in constraintkey_to_param.items():
        constraint = root_constraint[key]
        root_param[key] = constraint.root_params(param)

    return root_param
