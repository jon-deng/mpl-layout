"""
Classes for handling collections of primitives and constraints

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

import numpy as np

from . import geometry as geo
from .array import LabelledList

Prim = geo.Primitive
PrimLabelledList = LabelledList[geo.Primitive]
ConstraintLabelledList = LabelledList[geo.Constraint]

IntGraph = typ.List[typ.Tuple[int, ...]]
StrGraph = typ.List[typ.Tuple[str, ...]]


class PrimitiveTree:
    """
    Tree structure mapping keys to `Primitive`s

    The tree structure reflects the `Primitive` structure where a primitive can have
    associated child primitives.

    Parameters
    ----------
    data:
        A `Primitive` associated with the node
    children:
        Any child `Primitive` of the primitive

        This follows the recursive layout of `Primitive`. For example, consider a `Line`
        instance `line` which has two points. The tree representation of `line` is
        ```
        PrimitiveTree(line, {'Point0': line[0], 'Point1': line[1]})
        ```
    """

    def __init__(
        self, data: typ.Union[None, Prim], children: typ.Mapping[str, "PrimitiveTree"]
    ):
        self._data = data
        self._children = children

    def prim_graph(self) -> typ.Mapping[Prim, int]:
        """
        Return a mapping from primitives to integer indices in `self.prims()`
        """
        _graph = {tree.data: None for tree in self.values(flat=True)}
        return {prim: ii for ii, prim in enumerate(_graph)}

    def prims(self) -> typ.List[Prim]:
        """
        Return a list of all unique primitives in the tree
        """
        return list(self.prim_graph().keys())

    @property
    def children(self):
        """
        Return any child primitives of the primitive in `self.data`
        """
        return self._children

    @property
    def data(self):
        """
        Return the associated primitive
        """
        return self._data

    ## Dict-like interface
    def keys(self, flat: bool = False) -> typ.List[str]:
        """
        Return child keys

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys

            Child keys are separated using '/'
        """
        if flat:
            flat_keys = []
            for key, child_tree in self.children.items():
                flat_keys += [f"{key}"]
                flat_keys += [
                    f"{key}/{child_key}" for child_key in child_tree.keys(flat)
                ]
            return flat_keys
        else:
            return list(self.children.keys())

    def values(self, flat: bool = False) -> typ.List["PrimitiveTree"]:
        """
        Return child primitives

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten child primitives
        """
        if flat:
            flat_values = []
            for key, child_tree in self.children.items():
                flat_values += [child_tree]
                flat_values += child_tree.values(flat)
            return flat_values
        else:
            return list(self.children.values())

    def items(self, flat: bool = False) -> typ.List[typ.Tuple[str, "PrimitiveTree"]]:
        """
        Return paired child keys and associated trees

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys and trees
        """

        return zip(self.keys(flat), self.values(flat))

    def __getitem__(self, key: str):
        """
        Return the primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split("/")
        parent_key = split_key[0]
        child_key = "/".join(split_key[1:])

        try:
            if len(split_key) == 1:
                return self.children[parent_key].data
            else:
                return self.children[parent_key][child_key]
        except KeyError as err:
            raise KeyError(f"{key}") from err

    def __setitem__(self, key: str, value: "PrimitiveTree"):
        """
        Add a primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split("/")
        parent_key = split_key[0]
        child_keys = split_key[1:]
        child_key = "/".join(child_keys)

        try:
            if len(child_keys) > 0:
                self.children[parent_key][child_key] = value
            elif len(child_keys) == 0:
                self.children[parent_key] = value
            else:
                assert False

        except KeyError as err:
            raise KeyError(f"{key}") from err

    def __len__(self):
        return len(self.data)


def convert_prim_to_tree(prim: Prim) -> PrimitiveTree:
    """
    Return a `PrimitiveTree` representation of a `Primitive`
    """
    # Recursively create any child trees:
    children = {
        child_key: convert_prim_to_tree(child_prim)
        for child_key, child_prim in prim.prims.items()
    }
    return PrimitiveTree(prim, children)


class Layout:
    """
    Class used to handle a collection of primitives and associated constraints

    The class contains functions to add primitives to the collection, add
    constraints on those primitives, and create the associated graph between
    constraints and primitives.

    Parameters
    ----------
    prim_tree:
        A `PrimitiveTree`
    constraints:
        A list of constraints
    constraint_graph:
        A constraint graph
    """

    def __init__(
        self,
        prim_tree: typ.Optional[PrimitiveTree] = None,
        constraints: typ.Optional[ConstraintLabelledList] = None,
        constraint_graph: typ.Optional[StrGraph] = None,
    ):

        if prim_tree is None:
            prim_tree = PrimitiveTree(None, {})
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._primitive_tree = prim_tree
        self._constraints = LabelledList(constraints)
        self._constraint_graph = constraint_graph

        self._prim_type_count = {}
        self._label_to_primidx = {}

        self._constraint_type_count = {}
        self._label_to_constraintidx = {}

    def prims(self):
        return self.prim_tree.prims()

    @property
    def prim_tree(self):
        return self._primitive_tree

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraint_graph(self) -> StrGraph:
        return self._constraint_graph

    @property
    def constraint_graph_int(self) -> IntGraph:
        prim_graph = self.prim_tree.prim_graph()
        return [
            tuple(prim_graph[self.prim_tree[prim_label]] for prim_label in prim_labels)
            for prim_labels in self.constraint_graph
        ]

    def add_prim(self, prim: geo.Primitive, label: typ.Optional[str] = None) -> str:
        """
        Add a `Primitive` to the `Layout`

        The primitive will be added `self.prim_tree` under the given `label`.

        Parameters
        ----------
        prim: geo.Primitive
            The primitive to add
        label: typ.Optional[str]
            An optional label for the primitive

            If not provided, an automatic name based on the primitive class will
            be used.

        Returns
        -------
        label: str
            The label for the added primitive
        """
        self.prim_tree[label] = convert_prim_to_tree(prim)
        return label

    def add_constraint(
        self,
        constraint: geo.Constraint,
        prim_labels: typ.Tuple[str, ...],
        constraint_label: typ.Optional[str] = None,
    ) -> str:
        """
        Add a `Constraint` between `Primitive`s

        Parameters
        ----------
        constraint:
            The constraint to apply
        prim_idxs:
            A tuple of strings referencing primitives (`self.prim_tree`) to apply the constraint
        constraint_label: typ.Optional[str]
            An optional label for the constraint.
            If not provided, an automatic name based on the constraint class
            will be used.

        Returns
        -------
        constraint_label: str
            The label for the added constraint
        """
        constraint_label = self.constraints.append(constraint, label=constraint_label)
        self.constraint_graph.append(prim_labels)
        return constraint_label


def build_tree(
    tree: PrimitiveTree,
    prim_to_idx: typ.Mapping[Prim, int],
    params: typ.List[np.typing.NDArray],
    prim_to_newprim: typ.Mapping[Prim, Prim],
) -> PrimitiveTree:
    """
    Return a new `PrimitiveTree` using new primitives for given parameter values

    Parameters
    ----------
    tree:
        The old `PrimitiveTree` instance
    prim_to_idx:
        A mapping from `Primitive` in `tree` to corresponding parameters in `params`

        If `params` is a list with parameters in order from `tree.prims()`, then this
        corresponds to `tree.prim_graph()`.
    params:
        A list of parameter values to build a new `PrimitiveTree` with
    prim_to_newprim:
        A mapping from `Primitive`s in `tree` to replacement `Primitives` in the new tree

        This should be an empty dictionary in the root `build_tree` call. As
        `build_tree` builds the new tree, it will populate this dictionary to preserve
        the mapping of primitives.

    Returns
    -------
    PrimitiveTree
        The new `PrimitiveTree` with parameters from `params`
    """
    oldprim = tree.data

    # If the tree has no children, we simply create the primitive with no children
    # If the tree has children, we need to recursively create all child trees
    if len(tree.children) == 0:
        children = {}
    else:
        children = {
            key: build_tree(childtree, prim_to_idx, params, prim_to_newprim)
            for key, childtree in tree.children.items()
        }

    if oldprim is None:
        newprim = None
    elif oldprim in prim_to_newprim:
        newprim = prim_to_newprim[oldprim]
    else:
        param = params[prim_to_idx[oldprim]]
        newprim = type(oldprim)(
            param, tuple(child_tree.data for child_tree in children.values())
        )
        prim_to_newprim[oldprim] = newprim

    return PrimitiveTree(newprim, children)
