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
import warnings
import functools

import jax
from jax import numpy as jnp
import numpy as np

from . import geometry as geo
from .array import LabelledList

Prim = geo.Primitive
PrimIdx = str
PrimIdxs = typ.Tuple[PrimIdx, ...]
PrimLabelledList = LabelledList[typ.Union[geo.Primitive, geo.PrimitiveArray]]
ConstraintLabelledList = LabelledList[geo.Constraint]

PrimIdxGraph = typ.List[PrimIdxs]
IntGraph = typ.List[typ.Tuple[int, ...]]
StrGraph = typ.List[typ.Tuple[str, ...]]

class PrimitiveTree:
    """
    Tree structure mapping keys to `Primitive`s
    """

    def __init__(
            self,
            value: typ.Union[None, Prim],
            tree: typ.Mapping[str, 'PrimitiveTree']
        ):
        self._value = value
        self._tree = tree

    @property
    def prim_graph(self):
        """
        Return a mapping from primitive instance to integer index in `prims`
        """
        graph = {tree.value: None for tree in self.values(flat=True)}

        for idx, prim in enumerate(graph.keys()):
            graph[prim] = idx

        return graph

    @property
    def prims(self):
        """
        Return a list of all unique primitives in the tree
        """
        return list(self.prim_graph.keys())

    @property
    def tree(self):
        return self._tree

    @property
    def value(self):
        return self._value

    ## Dict-like interface
    def keys(self, flat: bool=False) -> typ.List[str]:
        if flat:
            flat_keys = []
            for key, child_tree in self.tree.items():
                flat_keys += [f'{key}']
                flat_keys += [
                    f'{key}/{child_key}' for child_key in child_tree.keys(flat)
                ]
            return flat_keys
        else:
            return list(self.tree.keys())

    def values(self, flat: bool=False) -> typ.List['PrimitiveTree']:
        if flat:
            flat_values = []
            for key, child_tree in self.tree.items():
                flat_values += [child_tree]
                flat_values += child_tree.values(flat)
            return flat_values
        else:
            return list(self.tree.values())

    def items(self, flat: bool=False) -> typ.List[typ.Tuple[str, 'PrimitiveTree']]:

        return zip(self.items(flat), self.values(flat))

    def __getitem__(self, key: str):
        """
        Return the primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split('/')
        parent_key = split_key[0]
        child_key = '/'.join(split_key[1:])

        try:
            if len(split_key) == 1:
                return self.tree[parent_key].value
            else:
                return self.tree[parent_key][child_key]
        except KeyError as err:
            raise KeyError(f"{key}") from err

    def __setitem__(self, key: str, value: typ.Union['PrimitiveTree', Prim]):
        """
        Add a primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split('/')
        # parent_key = split_key[0]
        child_key = '/'.join(split_key[1:])

        try:
            # Add any child primitives of `value`
            if isinstance(value, Prim):
                self.tree[key] = PrimitiveTree(value, {})
                for child_key, child_prim in value.prims.items():
                    self.tree[key][child_key] = child_prim

        except KeyError as err:
            raise KeyError(f"{key}") from err

    def __len__(self):
        return len(self.value)


class Layout:
    """
    Class used to handle a collection of primitives and associated constraints

    The class contains functions to add primitives to the collection, add
    constraints on those primitives, and create the associated graph between
    constraints and primitives.

    Parameters
    ----------
    prims: typ.Optional[PrimLabelledList]
        A list of primitives
    constraints: typ.Optional[ConstraintLabelledList]
        A list of constraints
    constraint_graph: typ.Optional[StrGraph]
        A constraint graph
    """

    def __init__(
            self,
            prim_tree: typ.Optional[PrimitiveTree]=None,
            constraints: typ.Optional[ConstraintLabelledList]=None,
            constraint_graph: typ.Optional[StrGraph]=None
        ):

        if prim_tree is None:
            prim_tree = PrimitiveTree(None, {})
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._prim_tree = prim_tree
        self._constraints = LabelledList(constraints)
        self._constraint_graph = constraint_graph

        self._prim_type_count = {}
        self._label_to_primidx = {}

        self._constraint_type_count = {}
        self._label_to_constraintidx = {}

    @property
    def prim_tree(self):
        return self._prim_tree

    @property
    def prims(self):
        return self.prim_tree.prims

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraint_graph(self) -> StrGraph:
        return self._constraint_graph

    @property
    def constraint_graph_int(self) -> IntGraph:
        return [
            tuple(
                self.prim_tree.prim_graph[
                    self.prim_tree[prim_label]
                ]
                for prim_label in prim_labels
            )
            for prim_labels in self.constraint_graph
        ]

    def add_prim(
            self,
            prim: geo.Primitive,
            label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `geo.Primitive` to the `Layout`

        The primitive will be added with the label 'label'. In addition, all
        child primitives will be recursively added with label
        'label.child_prim_label'.

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
        self.prim_tree[label] = prim
        return label

    def add_constraint(
            self,
            constraint: geo.Constraint,
            prim_labels: PrimIdxs,
            constraint_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `Constraint` between `Primitive`s

        Parameters
        ----------
        constraint: geo.Constraint
            The constraint to apply
        prim_idxs: PrimIdxs
            Indices of the primitives the constraint applies to
        constraint_label: typ.Optional[str]
            An optional label for the constraint.
            If not provided, an automatic name based on the constraint class
            will be used.

        Returns
        -------
        constraint_label: str
            The label for the added constraint
        """
        constraint_label = self.constraints.append(
            constraint, label=constraint_label
        )
        self.constraint_graph.append(prim_labels)
        return constraint_label

def make_str_constraint_graph(
        constraint: geo.Constraint,
        prim_idxs: PrimIdxs,
        prims: PrimLabelledList
    ):
    """
    Return a new `Constraint` that doesn't apply on `PrimitiveArray` elements

    `Constraint`s that don't apply to `PrimitiveArray` elements are needed
    because these involve `PrimitiveIndex` objects, which do not directly
    point to a `Primitive`. The constraint graph entry implied by this function
    (i.e. `constraint` applies to the primitives pointed to by `prim_idxs`)
    can't be easily used due to the `PrimitiveIndex` objects.

    This function creates a modified `Constraint` which applies directly to
    `PrimitiveArray` instances instead of indexes into these instances. A new
    constraint graph entry is returned which can be simply turned into a list of
    equations.

    Parameters
    ----------
    constraint: geo.Constraint
        The constraint to apply
    prim_idxs: PrimIdxs
        Indices of the primitives the constraint applies to
    prims: PrimLabelledList
        The list of primitives the constraint applies to

    Returns
    -------
    geo.Constraint
        The new constraint which doesn't apply to any `PrimitiveArray` elements

        This new constraint shouldn't have any array indices, so should apply
        to purely string labelled primitives.
    typ.Tuple[str, ...]
        The string labels for primitives the new `geo.Constraint` applies to
    """
    # These are prims the constraint applies to
    prims = tuple(prims[prim_idx.label] for prim_idx in prim_idxs)

    # Get index specifications for each `PrimitiveIndex`
    # These specify how to get a primitive object from indexes into
    # `PrimitiveArray` objects
    def ident_prim(prims: geo.Primitive):
        return prims[0]

    # If the `PrimitiveIndex` doesn't index into a `PrimitiveArray`, then
    # the index spec is just an identity type mapping (`ident_prim`) on the
    # root primitive (indexed by `(None,)`)
    idx_specs = tuple(
        (ident_prim, (None,)) if prim_idx.array_idx is None
        else prim.index_spec(prim_idx.array_idx)
        for prim, prim_idx in zip(prims, prim_idxs)
    )
    # `make_prims` makes the primitives that input into `Constraint`s
    # using the primitives indexed by `arg_idxs`
    make_prims = tuple(spec[0] for spec in idx_specs)
    make_prims_arg_idxs = tuple(spec[1] for spec in idx_specs)

    # Modify the primitive indices to account for the `idx_specs` as an
    # intermediate step in applying the constraint
    new_prim_idxs = tuple(
        (
            f'{prim_idx.label}.{child_idx}' if child_idx is not None
            else prim_idx.label
        )
        for _, arg_idxs, prim_idx in zip(
            make_prims, make_prims_arg_idxs, prim_idxs
        )
        for child_idx in arg_idxs
    )

    # Modify the constraint function to account for sub-indexing
    class ModifiedConstraint(geo.Constraint):

        def __call__(self, new_prims):
            make_prim_arg_lengths = tuple(
                1 if sub_idx is None else len(sub_idx)
                for sub_idx in make_prims_arg_idxs
            )

            arg_bounds = [0] + np.cumsum(make_prim_arg_lengths).tolist()
            prims = tuple(
                make_prim(new_prims[start:end])
                for make_prim, start, end in zip(make_prims, arg_bounds[:-1], arg_bounds[1:])
            )
            return constraint(prims)

    if all(make_prim == ident_prim for make_prim in make_prims):
        ModifiedConstraint.__name__ = constraint.__class__.__name__
    else:
        ModifiedConstraint.__name__ = f'_{constraint.__class__.__name__}'

    new_constraint = ModifiedConstraint()
    return new_constraint, new_prim_idxs

# Basic primitive routines
def expand_prim(
        prim: geo.Primitive,
        label: str
    ) -> typ.Tuple[
        typ.List[geo.Primitive],
        typ.List[str]
    ]:
    """
    Expand all child primitives of `prim` into a flat list

    The flattening is done so that if a parent primitive has `n` child
    primitives, these are placed immediately after the parent.

    Parameters
    ----------
    prim: geo.Primitive
        The primitive to be expanded
    label: str
        The label for the primitive

    Returns
    -------
    child_prims: typ.List[geo.Primitive]
        The list of child primitives
    child_labels: typ.List[str]
        The list of child primitive labels
    """
    # Expand child primitives
    child_prims = list(prim.prims)
    child_labels = [
        f'{label}.{child_label}' for child_label in prim.prims.keys()
    ]

    # Recursively expand any child primitives
    if len(prim.prims) == 0:
        return child_prims, child_labels
    else:
        for child_prim, child_label in zip(child_prims, child_labels):
            (_re_child_prims, _re_child_labels) = expand_prim(child_prim, child_label)
            child_prims += _re_child_prims
            child_labels += _re_child_labels
        return child_prims, child_labels

def contract_prim(
        prim: geo.Primitive,
        prims: typ.List[geo.Primitive]
    ) -> typ.Tuple[geo.Primitive, int]:
    """
    Collapse a flat collection of child primitives into a parent

    This function builds a parent primitive from child primitives.
    This is needed because primitives are parameterized by child primitives and
    are immutable.
    This function should undo the result of `expand_prim`.

    Parameters
    ----------
    prim: geo.Primitive
        The parent primitive
    prims: typ.List[geo.Primitive]
        The child primitives to collapse into the parent

    Returns
    -------
    geo.Primitive
        The parent primitive with updated child primitives
    int
        The total number of child primitives used in the list
    """
    num_child = len(prim.prims)

    child_prims = []
    m = num_child
    for child_prim in prims[:num_child]:
        cprim, n = contract_prim(child_prim, prims[m:])

        child_prims.append(cprim)
        m += n

    return type(prim)(param=prim.param, prims=tuple(child_prims)), m

def build_prims(
        prims: typ.List[geo.Primitive],
        params: typ.List[np.typing.NDArray]
    ) -> typ.List[geo.Primitive]:
    """
    Create an updated list of `Primitive`s from new parameters

    This function rebuilds a list of primitives with new parameters in
    a corresponding list of parameters.

    Parameters
    ----------
    prims: typ.List[geo.Primitive]
        The old list of primitives
    params: typ.List[np.typing.NDArray]
        The new list of parameters for the primitives

    Returns
    -------
    typ.List[geo.Primitive]
        The new list of primitives
    """

    # First create primitives where the new parameters have been applied
    _new_prims = [
        type(prim)(param=param, prims=prim.prims)
        for prim, param in zip(prims, params)
    ]

    # Contract all child primitives into parents.
    # This is needed because primitive are parameterized both by parameter and
    # other primitives.
    m = 0
    new_prims = []
    while m < len(_new_prims):
        prim, dm = contract_prim(_new_prims[m], _new_prims[m+1:])
        cprims, _clabels = expand_prim(prim, '')
        new_prims = new_prims + [prim] + cprims
        m += 1+dm

    return new_prims
