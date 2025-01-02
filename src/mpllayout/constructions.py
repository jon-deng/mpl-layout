"""
Geometric constructions

Constructions are functions that accept primitives and return a vector.
For example, this could be the coordinates of a point, the angle between two
lines, or the length of a single line.
"""

from typing import Callable, Optional, Any, TypeVar
from collections.abc import Iterable
from numpy.typing import NDArray

import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat, flatten, unflatten
from .containers import map as node_map, accumulate as node_accumulate



Param = float | int | NDArray | bool
Params = tuple[Param, ...]
ParamTypes = tuple[type[Param]]

PrimKeys = tuple[str, ...]
Prims = tuple[pr.Primitive, ...]
PrimTypes = tuple[type[pr.Primitive], ...]

# NOTE: An `ArraySize` specifies the size of array returned by a construction
# but more generally it could be some `NDArray` that encodes the
# datatype, shape etc.
ArraySize = int

ConstructionSignature = tuple[tuple[PrimTypes, ParamTypes], ArraySize]

ChildParams = Callable[[Params], list[Params]]

ConstructionValue = tuple[
    list[PrimKeys],
    ChildParams,
    ConstructionSignature
]

TCons = TypeVar("TCons", bound="ConstructionNode")

class PrimKeysNode(Node[PrimKeys]):
    """
    A tree of primitive keys indicating primitives for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding primitive keys node.
    """

    pass


class ParamsNode(Node[Params]):
    """
    A tree of residual parameters (kwargs) for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding parameters node.
    """

    pass


class ConstructionNode(Node[ConstructionValue]):
    """
    Node representation of a geometric construction

    A construction is defined by two components:
    1. A function that maps from input primitives and parameters to an array
    2. A specification of any child constructions

    (1) The construction function is implemented through a class method
        ``
        ConstructionNode.assem(cls, prims: Prims, *params: Params) -> NDArray
        ``
    where `prims` are geometric primitives and `*params` are additional
    parameters. Note that `assem` should be implemented with `jax` for
    algorithmic differentiation.

    (2) Child constructions are specified using `ConstructionNode` instances as
    well as two objects that represent how `child_prims` and `child_params` are
    created for each child construction from the parent `prims` and `params`.

    Information for both these components are stored in the `value` and
    `children` structures of a `Node`. See 'Parameters' for more details.

    To create specific construction classes subclass one of
    `LeafConstruction` or `CompoundConstruction`.

    Parameters
    ----------
    value: ConstructionValue
        Tuple representing the construction children and signature

        A construction value is a tuple of three objects describing how child
        constructions are evaluated and the signature of the construction
            ``child_prim_keys, child_params, signature = value``
        These are further described below.

        child_prim_keys: list[PrimKeys]
            Specifies how `child_prims` are created from `prims`

            This is a list of `PrimKeys` string tuples where each list element
            corresponds to one child construction. Every `PrimKeys` string tuple
            has the format
                ``('arg{i1}/child_key1', ..., 'arg{iM}/child_keyM')``,
            where 'i1, ..., iM' are integers indicating argument numbers in
            `prims`. Each 'child_key1, ..., child_keyM' are child keys
            for that the primitive argument.

            For example, consider `N` child constructions
                ``children = {1: cnode1, ..., N: cnodeN}``
            and
                ``child_prim_keys[n] = ('arg0/Point0', 'arg1/Line0/Point1')``,
            for the `n`th child construction with signature
                ``cnoden.assem(cls, child_prims, *child_params)``
            Then the input primitive for `cnoden` will be
                ``child_prims = (prims[0]['Point0'], prims[1]['Line0/Point'])``.
        child_params: Callable[[Params], list[Params]]
            Specifies how `child_params` are created from `params`

            This function should return a list of `child_params` for each child
            construction from `params`.
        signature: ConstructionSignature
            A tuple describing types for `assem`

            This a tuple of the format
                ``(prim_types, param_types), value_size = signature``
            where
            `prim_types` is a tuple of primitive types for `prims`,
            `param_types` is a tuple of parameter types for `params,
            and `value_size` is the array size returned by `assem`.
    children: dict[str, TCons]
        Child constructions
    """

    def __init__(self, value: ConstructionValue, children: dict[str, TCons]):

        ## Check `value`

        # Check that `value` is a size 3 tuple
        if not isinstance(value, tuple):
            raise TypeError(f"`value` must be a tuple not {type(value)}")
        elif len(value) != 3:
            raise ValueError(f"`value` must have 3 items not {len(value)}")

        # Check each component of `value`
        child_prim_keys, child_params, signature = value
        (prim_types, param_types), value_size = signature

        # Check `child_prim_keys` has one `PrimKeys` tuple for each child
        if not isinstance(child_prim_keys, (tuple, list)):
            raise TypeError(f"`value[0]` must be a tuple or list of `PrimKeys`")
        elif len(child_prim_keys) != len(children):
            raise ValueError(
                f"`value[0]` must have length {len(children)}, matching the number of children"
            )

        # Check each `child_prim_keys` tuple indexes the right number of child
        # prims for the corresponding child construction
        child_signatures = {
            key: child.value[-1] for key, child in children.items()
        }
        child_prim_types = {
            key: prim_types
            for key, ((prim_types, _), _) in child_signatures.items()
        }
        valid_child_prim_keys = {
            key: len(prim_types) == len(key_tuple)
            for (key, prim_types), key_tuple
            in zip(child_prim_types.items(), child_prim_keys)
        }
        if not all(valid_child_prim_keys):
            invalid_cons_keys = {
                key for key, valid in valid_child_prim_keys.items() if not valid
            }
            raise ValueError(
                "`value` indexes wrong number of child primitives for"
                f" construction keys {invalid_cons_keys}"
            )

        # Check `child_params` is callable
        if not isinstance(child_params, Callable):
            raise TypeError(f"value[1] must be `Callable`")

        ## Check `children`

        # Check that all children are constructions
        if not all(
            isinstance(child, ConstructionNode)
            for _, child in children.items()
        ):
            raise TypeError("`children` must be a dictionary of constructions")

        super().__init__(value, children)

    ## Attributes related to `value`

    @property
    def child_prim_keys_template(self) -> tuple[PrimKeys, ...]:
        return self.value[0]

    def child_prim_keys(self, arg_keys: tuple[str, ...]) -> tuple[PrimKeys, ...]:
        """
        Return primitive key tuples for each child construction

        The 'arg{n}' prefix of each prim key is replaced with the corresponding
        key in `arg_keys`.
        """
        # Replace the 'arg{n}/...' component with the corresponding string
        # in `arg_keys`

        def argnum_from_key(arg_prefix: str):
            if arg_prefix[:3] == "arg":
                arg_num = int(arg_prefix[3:])
            else:
                raise ValueError(f"Argument key {arg_prefix} must contain 'arg' prefix")
            return arg_num

        def replace_prim_key_prefix(prim_key: str, arg_keys):
            arg_prefix, *child_key = prim_key.split("/", 1)
            new_prefix = arg_keys[argnum_from_key(arg_prefix)]
            return "/".join([new_prefix] + child_key)

        # For each child, find the parent primitive part of its argument tuple
        return tuple(
            tuple(
                replace_prim_key_prefix(prim_key, arg_keys)
                for prim_key in child_prim_keys
            )
            for child_prim_keys in self.child_prim_keys_template
        )

    def child_params(self, params: Params) -> tuple[Params, ...]:
        return self.value[1](params)

    @property
    def signature(self) -> ConstructionSignature:
        return self.value[2]

    ## Node representations for construction-related data

    def root_prim(self, prim_keys: PrimKeys, prims: Prims) -> pr.PrimitiveNode:
        """
        Return a primitive node representing construction primitive arguments

        Parameters
        ----------
        prim_keys: PrimKeys
            Labels for each primitive argument

            Note the 'template' to name these is 'arg0', 'arg1', ...
        prims: Prims
            Primitive arguments for the construction

        Returns
        -------
        PrimitiveNode
            A root primitive containing primitives for the construction
        """
        return pr.PrimitiveNode(
            np.array([]), {key: prim for key, prim in zip(prim_keys, prims)}
        )

    def root_prim_keys(self, prim_keys: PrimKeys) -> PrimKeysNode:
        """
        Return a tree of `prim_keys` for all constructions

        Every `prim_keys` string tuple in a node indicates primitives in
        `self.root_prim(prim_keys, prims)` that are passed to the corresponding
        construction node.

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the construction

        Returns
        -------
        root_prim_keys: PrimKeysNode
            A tree of primitive keys for the construction and all children
        """
        children = {
            key: child.root_prim_keys(child_prim_keys)
            for (key, child), child_prim_keys in zip(
                self.items(), self.child_prim_keys(prim_keys)
            )
        }
        return PrimKeysNode(prim_keys, children)

    def root_params(self, params: Params) -> ParamsNode:
        """
        Return a tree of `params` for all constructions

        Every `params` tuple in a node indicates parameters for the
        corresponding construction node.

        Parameters
        ----------
        params: Params
            Parameters for the construction

        Returns
        -------
        root_params: ParamsNode
            A tree of params
        """
        children_params = self.child_params(params)
        children = {
            key: child.root_params(child_params)
            for (key, child), child_params in zip(self.items(), children_params)
        }
        return ParamsNode(params, children)

    ## Construction function related methods

    def __call__(self, prims: Prims, *params: Params) -> NDArray:
        prim_keys = tuple(f"arg{n}" for n, _ in enumerate(prims))
        root_prim = self.root_prim(prim_keys, prims)
        root_prim_keys = self.root_prim_keys(prim_keys)
        root_params = self.root_params(params)
        return self.assem_from_tree(root_prim, root_prim_keys, root_params)

    def assem_from_tree(
        self,
        root_prim: pr.PrimitiveNode,
        root_prim_keys: PrimKeysNode,
        root_params: ParamsNode,
    ) -> NDArray:
        flat_constructions = [x for _, x in iter_flat("", self)]
        flat_prim_keys = [x.value for _, x in iter_flat("", root_prim_keys)]
        flat_params = [x.value for _, x in iter_flat("", root_params)]

        residuals = [
            construction.assem_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in argkeys), *params
            )
            for construction, argkeys, params in zip(
                flat_constructions, flat_prim_keys, flat_params
            )
        ]
        return jnp.concatenate(residuals)

    def assem_atleast_1d(self, prims: Prims, *params: Params) -> NDArray:
        return jnp.atleast_1d(self.assem(prims, *params))

    @classmethod
    def assem(cls, prims: Prims, *params: Params) -> NDArray:
        """
        Return the (local) construction output array

        The full construction output consists of recursively stacking all
        construction outputs in a tree.

        Parameters
        ----------
        prims: Prims
            A tuple of primitives the construction applies to
        *params: Params
            A set of parameters for the residual

            These are things like a direction vector, boolean flag, etc.

        Returns
        -------
        res: NDArray
            The result array of the construction
        """
        # NOTE: Some `Construction` instances have `**kwargs` that affect
        # construction outputs which has some overlap with `*params`.
        # Generally, `*params` involves direct changes to the (local)
        # construction function while `**kwargs` involve changes to the tree
        # structure of child construction.
        # Some changes can be implemented using either `*params` or `**kwargs`
        # so there is some overlap.
        raise NotImplementedError()

    # TODO: Add construction `prims`, `params` (output?) validation methods


class Construction(ConstructionNode):
    """
    Construction with parameterized primitive argument types and child constructions

    To specify a `Construction`, define `init_children` and `init_signature`.

    Parameters
    ----------
    **kwargs
        Parameter controlling the construction definition

        Subclasses should define what these keyword arguments are.
    """

    def __init__(self, **kwargs):
        (
            child_keys, child_constructions, child_prim_keys, child_params
        ) = self.init_children(**kwargs)
        signature = self.init_signature(**kwargs)

        value = (child_prim_keys, child_params, signature)
        children = {
            key: child for key, child in zip(child_keys, child_constructions)
        }
        super().__init__(value, children)

    @classmethod
    def init_children(cls, **kwargs) -> tuple[
        list[str],
        list[TCons],
        list[PrimKeys],
        ChildParams,
    ]:
        raise NotImplementedError()

    @classmethod
    def init_signature(cls, **kwargs) -> ConstructionSignature:
        raise NotImplementedError()


class CompoundConstruction(Construction):

    @classmethod
    def assem(cls, prims: Prims, *params: Params) -> NDArray:
        return np.array(())


class ArrayCompoundConstruction(CompoundConstruction):
    """
    Construction representing an array of child constructions
    """

    def __init__(self, shape: tuple[int, ...] = (0,)):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(shape=shape)


class StaticCompoundConstruction(CompoundConstruction):
    """
    Construction with static primitive argument types and child constructions

    To specify a `StaticConstraint`:
    - define `init_signature`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the construction will have no child
    constructions by default.

    If `split_children_params` is undefined, all child constructions will be passed
    empty parameters, and therefore use default values.
    """

    def __init__(self):
        super().__init__()


class LeafConstruction(Construction):
    """
    Construction without any child constructions

    To specify a `LeafConstruction`, define `assem` and `init_signature`
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def init_children(cls, **kwargs) -> tuple[
        list[str],
        list[TCons],
        list[PrimKeys],
        ChildParams,
    ]:
        def child_params(params: Params) -> Params:
            return ()
        return (), (), (), child_params


## Construction transform functions

# These functions transform constructions into new ones

# TODO: Add 'ConstantArray' construction and transform to represent the 'value'
# parameter in constraints. You could then refactor a constraint as
# 'construction - constarray'

def transform_ConstraintType(ConstructionType: type[TCons]):
    """
    Return a constraint from a construction type

    See `transform_constraint` for more details.

    Parameters
    ----------
    ConstructionType: type[TCons]
        The construction class to transform

    Returns
    -------
    DerivedConstraint: type[ConstructionNode]
        The transformed constraint class
    """

    class DerivedConstraint(ConstructionNode):

        def __new__(cls, **kwargs):
            construction = ConstructionType(**kwargs)
            return transform_constraint(construction)

        def __init__(self, **kwargs):
            pass

    DerivedConstraint.__name__ = ConstructionType.__name__

    return DerivedConstraint


def transform_constraint(construction: TCons):
    """
    Return a constraint from a construction

    The transformed constraint returns a value according to
    ```construction(prims, *params, value) - value```.
    The transformed constraint accepts an additional parameter `value` on top
    of the construction parameters.


    Parameters
    ----------
    construction: typeTCons
        The construction to transform

    Returns
    -------
    constraint: DerivedConstraint
        The transformed constraint class
    """

    # Tree of construction output sizes
    def value_size(node_value: ConstructionValue) -> int:
        *_, signature = node_value
        (_prim_types, _param_types), value_size = signature
        return value_size

    size_node = node_map(value_size, construction)
    cumsize_node = node_accumulate(lambda x, y: x + y, size_node, 0)

    vector = Vector(size_node)
    return transform_sum(construction, transform_scalar_mul(vector, -1))


def transform_flat_constraint(
    construction: TCons,
    child_value_sizes: tuple[int, ...]
):
    """
    Return a constraint structure from a construction

    This transforms all constructions in a node to have an additional `value`
    parameter.
    Each transformed constraint has a local output
    ```construction.assem(prims, *params, value) - value```
    where `value` is recursively chunked into sizes matching each construction
    size.

    Parameters
    ----------
    construction: TCons
        The construction to transform
    child_values_sizes: tuple[int, ...]
        The sizes of child construction outputs

        This is used to chunk the input `value` into `value` parameters for each
        child.

    Returns
    -------
    DerivedConstraint
        The derived constraint type with transformed `assem`
    derived_value
        The derived constraint 'value'

        This is a tuple of: primitive keys, a function to create child
        parameters, and a construction signature. See `ConstraintNode` for more
        details.
    CHILD_KEYS
        Keys for each child constraint
    """
    # Return a (nonrecursive) constraint from a construction
    # Every constraint modifies the construction by:
    # split_child_params = split_child_params + value_split
    #   Should return chunk of values corresponding to each construction value size
    #   e.g. ConA has children (ConB, ConC) with values sizes (2,) (4,)
    #   ConA must split an input value of size (6,) into two arrays of size (2,) and (4,)
    # prims = prims
    # params = params, value
    # assem = assem - value

    CHILD_KEYS = tuple(construction.keys())

    # The new construction has the same `CHILD_PRIM_KEYS`
    CHILD_PRIM_KEYS, CHILD_PARAMS, SIGNATURE = construction.value

    # Define derived `child_params` function
    def child_value(value):
        if isinstance(value, (float, int)):
            return len(CHILD_KEYS) * (value,)
        else:
            return tuple(chunk(value, child_value_sizes))

    def derived_child_params(derived_params):
        *params, value = derived_params
        return tuple(
            (*params, value)
            for params, value in zip(CHILD_PARAMS(params), child_value(value))
        )

    # Define derived `signature`
    (prim_types, param_types), value_size = SIGNATURE
    derived_signature = ((prim_types, param_types + (np.ndarray,)), value_size)

    # Define derived `assem behaviour`
    if isinstance(construction, CompoundConstruction):

        def derived_assem(prims, derived_params):
            *params, value = derived_params
            return np.array(())

    elif isinstance(construction, LeafConstruction):

        def derived_assem(prims, derived_params):
            *params, value = derived_params
            return type(construction).assem(prims, *params) - value

    else:
        assert False

    class DerivedConstraint(ConstructionNode):

        def __init__(self, **kwargs):
            raise NotImplementedError()

        @classmethod
        def assem(cls, prims, *derived_params):
            return derived_assem(prims, derived_params)

    DerivedConstraint.__name__ = type(construction).__name__

    derived_value = (CHILD_PRIM_KEYS, derived_child_params, derived_signature)
    return DerivedConstraint, derived_value, CHILD_KEYS


def transform_MapType(ConstructionType: type[TCons], PrimTypes: list[type[pr.Primitive]]):
    """
    Return a derived construction that maps over an array of primitives

    See `transform_map` for more details.

    Parameters
    ----------
    ConstructionType: type[TCons]
        The construction class to transform
    PrimType: list[type[pr.Primitive]]
        The list of primitives to map over

    Returns
    -------
    MapConstruction: type[ConstructionNode]
        The transformed map construction class
    """

    class MapConstruction(ConstructionNode):

        def __new__(cls, **kwargs):
            construction = ConstructionType(**kwargs)
            return transform_map(construction, PrimTypes)

        def __init__(self, **kwargs):
            pass

    MapConstruction.__name__ = f"Map{ConstructionType.__name__}"

    return MapConstruction


def transform_map(
    construction: TCons,
    PrimTypes: list[type[pr.Primitive]]
):
    """
    Return a derived construction that maps over an array of primitives

    This works in the typical way if `construction` accepts a single primitive.

    If `construction` accepts more than one primitive, all additional primitives
    are treated as `frozen`.
    These primitives are always taken from the last primitive in the
    map construction's primitive input array.

    For example, consider a `construction` with signature
    `Callable[[PrimA, PrimB1, ..., PrimBM], NDArray]`
    where M is the number of additional primitives.
    The map construction over prims `prims = [PrimA1, ... PrimAN]` returns a
    tree of child constructions
    `[construction(prims[0], prims[-M:]), ..., construction(prims[N-M-1], prims[N-M:])]`

    Parameters
    ----------
    construction: TCons
        The construction to map
    PrimType: list[type[pr.Primitive]]
        The list of primitives to map over

    Returns
    -------
    MapConstruction
        The transformed map construction
    """
    PRIM_KEYS, CHILD_PRIMS, SIGNATURE = construction.value
    N = len(PrimTypes)
    # `M` is the number of additional arguments past one for the construction
    (PRIM_TYPES, PARAM_TYPES), value_size = SIGNATURE
    M = len(PRIM_TYPES)-1

    num_constr = max(N - M, 0)

    child_keys = tuple(
        f"{type(construction).__name__}{n}" for n in range(num_constr)
    )
    child_constructions = num_constr*(construction,)

    constant_prim_keys = tuple(
        f"arg{ii}" for ii in range(N-M, N)
    )
    child_prim_keys = tuple(
        (f"arg{n}", *constant_prim_keys) for n in range(num_constr)
    )
    def child_params(map_params):
        num_params = len(PARAM_TYPES)
        return tuple(
            map_params[n * num_params : (n + 1) * num_params]
            for n in range(num_constr)
        )

    map_signature = (
        (num_constr*PRIM_TYPES[:1] + PRIM_TYPES[1:], num_constr*PARAM_TYPES), 0
    )

    class MapConstruction(ConstructionNode):

        def __init__(self):
            value = (child_prim_keys, child_params, map_signature)
            children = {
                key: child for key, child in zip(child_keys, child_constructions)
            }
            super().__init__(value, children)

        @classmethod
        def assem(cls, prims, *map_params):
            return np.array(())

    MapConstruction.__name__ = f"Map{type(construction).__name__}"

    return MapConstruction()


def transform_sum(cons_a: TCons, cons_b: TCons) -> ConstructionNode:
    """
    Return a construction representing the sum of two input constructions
    """
    # Check that the two constructions (and all children) have the same signature
    # Two constructions can be added if their output size nodes are the same

    def transform_SumConstruction(cons_a: TCons, cons_b: TCons) -> ConstructionNode:

        child_keys_a = cons_a.keys()
        child_keys_b = cons_b.keys()

        child_prim_keys_a, child_params_a, signature_a = cons_a.value
        child_prim_keys_b, child_params_b, signature_b = cons_b.value

        (prim_types_a, param_types_a), size_a = signature_a
        (prim_types_b, param_types_b), size_b = signature_b

        assert size_a == size_b

        sum_child_prim_keys = tuple(
            prim_keys_a + prim_keys_b
            for prim_keys_a, prim_keys_b
            in zip(child_prim_keys_a, child_prim_keys_b)
        )

        assert child_keys_a == child_keys_b
        sum_child_keys = child_keys_a

        sum_signature = (
            (prim_types_a + prim_types_b, param_types_a + param_types_b),
            size_a
        )

        def sum_child_params(sum_params: Params) -> tuple[Params, ...]:
            param_chunks = (len(param_types_a), len(param_types_b))
            params_a, params_b = tuple(chunk(sum_params, param_chunks))
            return (
                ca + cb for ca, cb
                in zip(child_params_a(params_a), child_params_b(params_b))
            )

        node_value = (sum_child_prim_keys, sum_child_params, sum_signature)

        class SumConstruction(ConstructionNode):

            @classmethod
            def assem(cls, sum_prims: Prims, *sum_params: Params) -> NDArray:
                prim_chunks = (len(prim_types_a), len(prim_types_b))
                prims_a, prims_b = tuple(chunk(sum_prims, prim_chunks))

                param_chunks = (len(param_types_a), len(param_types_b))
                params_a, params_b = tuple(chunk(sum_params, param_chunks))
                return cons_a.assem(prims_a, *params_a) + cons_b.assem(prims_b, *params_b)

        return SumConstruction, node_value, sum_child_keys

    flat_a = [a for a in iter_flat("", cons_a)]
    flat_b = [b for b in iter_flat("", cons_b)]

    flat_sum_constructions = [
        (key, *transform_SumConstruction(a, b))
        for (key, a), (_, b) in zip(flat_a, flat_b)
    ]

    return unflatten(flat_sum_constructions)[0]


def transform_scalar_mul(cons_a: TCons, scalar: Optional[float]=None) -> ConstructionNode:
    """
    Return a construction representing a construction multiplied by a scalar
    """
    # If `scalar=None` a new scalar parameter is added
    # If `scalar` is a float, then the scalar float is used to multiply the construction
    # and no additional parameter is added

    def transform_ScalarMultiple(
        cons_a: TCons, scalar: float | None
    ) -> ConstructionNode:
        child_keys = cons_a.keys()
        child_prim_keys, child_params, signature = cons_a.value

        if scalar is None:
            class ScalarMultipleConstruction(ConstructionNode):

                @classmethod
                def assem(cls, prims: Prims, *params: Params) -> NDArray:
                    *_params, scalar = params
                    return scalar * cons_a.assem(prims, *_params)

            def mul_child_params(params: Params) -> tuple[Params, ...]:
                *_params, scalar = params
                return tuple(
                    (*_child_params, scalar)
                    for _child_params in child_params(_params)
                )

            (prim_types, param_types), value_size = signature
            mul_signature = ((prim_types, param_types+(float,)), value_size)

        elif isinstance(scalar, (float, int)):
            class ScalarMultipleConstruction(ConstructionNode):

                @classmethod
                def assem(cls, prims: Prims, *params: Params) -> NDArray:
                    return scalar * cons_a.assem(prims, *params)

            def mul_child_params(params: Params) -> tuple[Params, ...]:
                return child_params(params)

            mul_signature = signature
        else:
            raise TypeError(
                "`scalar` must be `float | int` not `{type(scalar)}`"
            )


        node_value = (child_prim_keys, mul_child_params, mul_signature)
        return ScalarMultipleConstruction, node_value, child_keys

    flat_a = [a for a in iter_flat("", cons_a)]
    flat_sum_constructions = [
        (key, *transform_ScalarMultiple(a, scalar)) for key, a in flat_a
    ]

    return unflatten(flat_sum_constructions)[0]


T = TypeVar('T')

def chunk(array: list[T], chunk_sizes: list[int]) -> Iterable[list[T]]:

    slice_bounds = list(itertools.accumulate(chunk_sizes, initial=0))
    # cum_chunk_size = slice_bounds[-1]

    return (
        array[start:stop]
        for start, stop in zip(slice_bounds[:-1], slice_bounds[1:])
    )

## Construction signatures


def make_signature_class(arg_types: tuple[type[pr.Primitive], ...]):

    class PrimsSignature:
        @staticmethod
        def make_signature(
            value_size: ArraySize, param_types: ParamTypes = ()
        ):
            return ((arg_types, param_types), value_size)

    return PrimsSignature


_NullSignature = make_signature_class(())

_PointSignature = make_signature_class((pr.Point,))

_PointPointSignature = make_signature_class((pr.Point, pr.Point))

_LineSignature = make_signature_class((pr.Line,))

_LineLineSignature = make_signature_class((pr.Line, pr.Line))

_LinesSignature = make_signature_class((pr.Line, ...))

_PointLineSignature = make_signature_class((pr.Point, pr.Line))

_QuadrilateralSignature = make_signature_class((pr.Quadrilateral,))

_QuadrilateralQuadrilateralSignature = make_signature_class(
    (pr.Quadrilateral, pr.Quadrilateral)
)

_QuadrilateralsSignature = make_signature_class((pr.Quadrilateral, ...))

_AxesSignature = make_signature_class((pr.Axes,))

## Constant constructions

# TODO: Need to handle scalar value input (just pass on right vector size to each node?)

class Vector(Construction, _NullSignature):

    def __init__(self, size_node: Optional[Node[int]]=None):

        super().__init__(size_node=size_node)

    @classmethod
    def init_children(cls, size_node: Optional[Node[int]]=None):
        cumsize_node = node_accumulate(lambda x, y: x + y, size_node, 0)
        num_child = len(size_node)

        child_size_nodes = [node for _, node in size_node.items()]

        keys = tuple(size_node.keys())
        constructions = tuple(
            Vector(size_node=node) for node in child_size_nodes
        )
        prim_keys = num_child*((),)

        child_cumsizes = tuple(node.value for _, node in cumsize_node.items())
        child_sizes = tuple(node.value for _, node in size_node.items())

        def child_params(params: Params) -> tuple[Params, ...]:
            value, = params

            return tuple(
                (child_value[:size],)
                for child_value, size
                in zip(chunk(value, child_cumsizes), child_sizes)
            )

        return (keys, constructions, prim_keys, child_params)

    @classmethod
    def init_signature(cls, size_node: Optional[Node[int]]=None):
        return cls.make_signature(size_node.value, (np.ndarray,))

    @classmethod
    def assem(cls, prims, value):
        assert prims == ()
        return value


## Point constructions
# NOTE: These are actual construction classes that can be called so class docstrings
# document there `assem_res` function.

# Argument type: tuple[Point,]


class Coordinate(LeafConstruction, _PointSignature):
    """
    Return point coordinates

    Parameters
    ----------
    prims: tuple[pr.Point]
        The point
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Point]):
        (point,) = prims
        return point.value


# Argument type: tuple[Point, Point]


class DirectedDistance(LeafConstruction, _PointPointSignature):
    """
    Return the distance between two points along a direction

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    direction: NDArray
        The direction
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Point], direction: NDArray):
        point0, point1 = prims
        return jnp.dot(point1.value - point0.value, direction)


class XDistance(LeafConstruction, _PointPointSignature):
    """
    Return the x-distance between two points

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, ())

    @classmethod
    def assem(self, prims: tuple[pr.Point, pr.Point]):
        return DirectedDistance.assem(prims, np.array([1, 0]))


class YDistance(LeafConstruction, _PointPointSignature):
    """
    Return the y-distance between two points

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, ())

    @classmethod
    def assem(self, prims: tuple[pr.Point, pr.Point]):
        return DirectedDistance.assem(prims, np.array([0, 1]))


## Line constructions

# Argument type: tuple[Line,]


class LineVector(LeafConstruction, _LineSignature):
    """
    Return the vector of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        pointa, pointb = line.values()
        return pointb.value - pointa.value


class UnitLineVector(LeafConstruction, _LineSignature):
    """
    Return the unit vector of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        line_vec = LineVector.assem(prims)
        return line_vec / jnp.linalg.norm(line_vec)


class Length(LeafConstruction, _LineSignature):
    """
    Return the length of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return jnp.sum(LineVector.assem((line,)) ** 2) ** (1 / 2)


class DirectedLength(LeafConstruction, _LineSignature):
    """
    Return the length of a line along a direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    direction: NDArray
        The direction
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line], direction: NDArray):
        (line,) = prims
        return jnp.dot(LineVector.assem((line,)), direction)


class XLength(LeafConstruction, _LineSignature):
    """
    Return the length of a line along the x direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([1, 0]))


class YLength(LeafConstruction, _LineSignature):
    """
    Return the length of a line along the y direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([0, 1]))


class Midpoint(LeafConstruction, _LineSignature):
    """
    Return the midpoint coordinate of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """
    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return 1 / 2 * (
            Coordinate.assem((line["Point0"],))
            + Coordinate.assem((line["Point1"],))
        )


# Argument type: tuple[Line, Line]


class MidpointDirectedDistance(LeafConstruction, _LineLineSignature):
    """
    Return the distance between two line midpoints along a direction

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines

        The distance is measured from the first to the second line
    direction: NDArray
        The direction
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], direction: NDArray):
        line0, line1 = prims
        return jnp.dot(
            Midpoint.assem((line1,)) - Midpoint.assem((line0,)), direction
        )


class MidpointXDistance(LeafConstruction, _LineLineSignature):
    """
    Return the x-distance between two line midpoints

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines

        The distance is measured from the first to the second line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        return MidpointDirectedDistance.assem(prims, np.array([1, 0]))


class MidpointYDistance(LeafConstruction, _LineLineSignature):
    """
    Constrain the y-distance between two line midpoints

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines

        The distance is measured from the first to the second line
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        return MidpointDirectedDistance.assem(prims, np.array([0, 1]))


class Angle(LeafConstruction, _LineLineSignature):
    """
    Return the angle between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        line0, line1 = prims
        dir0 = UnitLineVector.assem((line0,))
        dir1 = UnitLineVector.assem((line1,))
        return jnp.arccos(jnp.dot(dir0, dir1))


# Argument type: tuple[Line, ...]


## Point and Line constructions

# Argument type: tuple[Point, Line]


class PointOnLineDistance(LeafConstruction, _PointLineSignature):
    """
    Return the distance of a point along a line

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Line]
        The point and line
    reverse: bool
        A boolean indicating whether to reverse the line direction

        The distance of the point on the line is measured either from the start
        or end point of the line based on `reverse`. If `reverse=False` then the
        start point is used.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (bool,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Line], reverse: bool):
        point, line = prims
        if reverse:
            origin = line["Point1"].value
            unit_vec = -UnitLineVector.assem((line,))
        else:
            origin = line["Point0"].value
            unit_vec = UnitLineVector.assem((line,))

        return jnp.dot(point.value - origin, unit_vec)


class PointToLineDistance(LeafConstruction, _PointLineSignature):
    """
    Return the orthogonal distance of a point to a line

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Line]
        The point and line
    reverse: NDArray
        Whether to reverse the line direction for measuring the orthogonal

        By convention the orthogonal direction rotates the unit line vector 90
        degrees counter-clockwise. If `reverse=True` then the orthogonal
        vector rotates clockwise.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (bool,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Line], reverse: bool):
        point, line = prims

        line_vec = UnitLineVector.assem((line,))

        if reverse:
            orth_vec = jnp.cross(line_vec, np.array([0, 0, 1]))[:2]
        else:
            orth_vec = jnp.cross(line_vec, np.array([0, 0, -1]))[:2]

        origin = line["Point0"].value

        return jnp.dot(point.value - origin, orth_vec)


## Quad constructions

# Argument type: tuple[Quadrilateral]


class AspectRatio(LeafConstruction, _QuadrilateralSignature):
    """
    Return the aspect ratio of a quadrilateral

    This is ratio of the bottom width over the side height

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The quadrilateral
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral]):
        (quad,) = prims
        width = Length.assem((quad["Line0"],))
        height = Length.assem((quad["Line1"],))
        return width / height


# Argument type: tuple[Quadrilateral, Quadrilateral]


class OuterMargin(CompoundConstruction, _QuadrilateralQuadrilateralSignature):
    """
    Return the outer margin between two quadrilaterals

    The is the distance between outside faces of the quadrilaterals.

    Class Parameters
    ----------------
    side: str
        The side of the quadrilateral to return a margin

        The returned margin depends on `side`.
        - If `side` is 'left', the margin is measured from the left face to the
        right face of the first and second quad, respectively.
        - If `side` is 'right', the margin is measured from the right face to the
        left face of the first and second quad, respectively.
        - If `side` is 'bottom', the margin is measured from the bottom face to the
        top face of the first and second quad, respectively.
        - If `side` is 'top', the margin is measured from the top face to the
        bottom face of the first and second quad, respectively.

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quadrilaterals
    """

    def __init__(self, side: str = "left"):
        super().__init__(side=side)

    @classmethod
    def init_children(cls, side: str = "left"):
        if side == "left":
            keys = ("LeftMargin",)
            constructions = (MidpointXDistance(),)
            prim_keys = (("arg1/Line1", "arg0/Line3"),)
        elif side == "right":
            keys = ("RightMargin",)
            constructions = (MidpointXDistance(),)
            prim_keys = (("arg0/Line1", "arg1/Line3"),)
        elif side == "bottom":
            keys = ("BottomMargin",)
            constructions = (MidpointYDistance(),)
            prim_keys = (("arg1/Line2", "arg0/Line0"),)
        elif side == "top":
            keys = ("TopMargin",)
            constructions = (MidpointYDistance(),)
            prim_keys = (("arg0/Line2", "arg1/Line0"),)
        else:
            raise ValueError()

        def child_params(params: Params) -> tuple[Params, ...]:
            return ((),)

        return (keys, constructions, prim_keys, child_params)

    @classmethod
    def init_signature(cls, side: str = "left"):
        return cls.make_signature(0, ())


class InnerMargin(CompoundConstruction, _QuadrilateralQuadrilateralSignature):
    """
    Return the inner margin between two quadrilaterals

    The is the distance between inside faces of the quadrilaterals.

    Class Parameters
    ----------------
    side: str
        The side of the quadrilateral to return an inner margin

        The returned margin depends on `side`.
        - If `side` is 'left', the margin is measured from the left face to the
        left face of the first and second quad, respectively.
        - If `side` is 'right', the margin is measured from the right face to the
        right face of the first and second quad, respectively.
        - If `side` is 'bottom', the margin is measured from the bottom face to the
        bottom face of the first and second quad, respectively.
        - If `side` is 'top', the margin is measured from the top face to the
        top face of the first and second quad, respectively.

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quadrilaterals
    """

    def __init__(self, side: str = "left"):
        super().__init__(side=side)

    @classmethod
    def init_children(cls, side: str = "left"):
        if side == "left":
            keys = ("LeftMargin",)
            constructions = (MidpointXDistance(),)
            prim_keys = (("arg1/Line3", "arg0/Line3"),)
        elif side == "right":
            keys = ("RightMargin",)
            constructions = (MidpointXDistance(),)
            prim_keys = (("arg0/Line1", "arg1/Line1"),)
        elif side == "bottom":
            keys = ("BottomMargin",)
            constructions = (MidpointYDistance(),)
            prim_keys = (("arg1/Line0", "arg0/Line0"),)
        elif side == "top":
            keys = ("TopMargin",)
            constructions = (MidpointYDistance(),)
            prim_keys = (("arg0/Line2", "arg1/Line2"),)
        else:
            raise ValueError()

        def child_params(params: Params) -> tuple[Params, ...]:
            return ((),)

        return (keys, constructions, prim_keys, child_params)

    @classmethod
    def init_signature(cls, side: str = "left"):
        return cls.make_signature(0, ())


# Argument type: tuple[Quadrilateral, ...]


## Axes constructions

# Argument type: tuple[Axes]
