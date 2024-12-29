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

Params = tuple[Any, ...]

Prims = tuple[pr.Primitive, ...]
PrimKeys = tuple[str, ...]
PrimTypes = tuple[type[pr.Primitive], ...]

TCons = TypeVar("TCons", bound="ConstructionNode")

# TODO: Refine construction signature type/representation
ConstructionSignature = dict[str, Any]

ChildParams = Callable[[Params], list[Params]]

ConstructionValue = tuple[
    list[PrimKeys],
    ChildParams,
    ConstructionSignature
]

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
    The base geometric construction class

    A geometric construction is a function returning a vector from geometric
    primitives.
    A construction has a tree structure.
    The return vector consists of recursively stacking constructions from all
    child constructions.

    The construction function is implemented through the method
        `assem(self, prims, *params)`
    where `prims` are the geometric primitives to construction and `*params` are
    additional parameters for the residual.
    This should be implemented using `jax` functions.

    Constructions must specify how to create `(prims, *params)` for each child
    construction.
    This is done through `child_prim_keys` and `child_params`.

    You must subclass `LeafConstruction` and/or `CompoundConstruction` to create
    specific construction implementations.

    Parameters
    ----------
    child_keys: list[str]
        Keys for any child constructions
    child_constructions: list[TCons]
        Child constructions
    child_prim_keys: list[PrimKeys]
        `prims` argument representation for each child construction

        This is stored as the "value" of the tree structure and encodes how to
        create primitives for each child construction from the parent construction's
        `prims` (in `assem(self, prims, *params)`).
        For each child construction, a tuple of primitive keys indicates a
        subset of parent primitives to form child construction primitive
        arguments.

        To illustrate this, consider a parent construction with residual

        ```
        Parent.assem(self, prims, *params)
        ```

        and the n'th child construction with primitive key tuple

        ```
        child_prim_keys[n] == ('arg0', 'arg3/Line2')
        ```.

        This indicates the n'th child construction should be evaluated with

        ```
        child_prims = (prims[0], prims[3]['Line2'])
        ```
    child_params: Callable[[Params], list[Params]]
        `*params` argument representation for each child construction

        This function should return `params` for each child construction given
        the parent `params`.
    signature: ConstructionSignature
        The construction signature

        This describes the construction input and output types.
    """

    # TODO: Implement `assem` type checking using `signature`
    def __init__(
        self,
        child_keys: list[str],
        child_constructions: list[TCons],
        child_prim_keys: list[PrimKeys],
        child_params: ChildParams,
        signature: ConstructionSignature,
    ):
        children = {key: child for key, child in zip(child_keys, child_constructions)}
        super().__init__((child_prim_keys, child_params, signature), children)

    def root_params(self, params: Params) -> ParamsNode:
        """
        Return a tree of residual kwargs for the construction and all children

        The tree structure should match the tree structure of the construction.

        Parameters
        ----------
        params: ResParams
            Residual keyword arguments for the construction

        Returns
        -------
        root_params: ParamsNode
            A tree of keyword arguments for the construction and all children
        """
        children_params = self.child_params(params)
        children = {
            key: child.root_params(child_params)
            for (key, child), child_params in zip(self.items(), children_params)
        }
        root_params = ParamsNode(params, children)
        return root_params

    def root_prim_keys(self, prim_keys: PrimKeys) -> PrimKeysNode:
        """
        Return a tree of primitive keys for the construction and all children

        The tree structure should match the tree structure of the construction.

        For a given construction, `c`, every key tuple in
        `c.root_prim_keys(prim_keys)` specifies a tuple of primitives for the
        corresponding construction by indexing from
        `c.root_prim(prim_keys, prims)`.

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

    def root_prim(self, prim_keys: PrimKeys, prims: Prims) -> pr.PrimitiveNode:
        """
        Return a root primitive containing primitives for the construction

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the construction
        prims: ResPrims
            Primitives for the construction

        Returns
        -------
        PrimitiveNode
            A root primitive containing primitives for the construction
        """
        return pr.PrimitiveNode(
            np.array([]), {key: prim for key, prim in zip(prim_keys, prims)}
        )

    @property
    def _child_prim_keys_template(self) -> tuple[PrimKeys, ...]:
        return self.value[0]

    def child_prim_keys(self, arg_keys: tuple[str, ...]) -> tuple[PrimKeys, ...]:
        """
        Return primitive key tuples for each child constraint

        The 'arg{n}' part of the prim key is replace with the corresponding key
        in `parent_prim_keys`.
        """
        # Replace the 'arg{n}/...' component with the corresponding string
        # in `prim_keys`

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
            for child_prim_keys in self._child_prim_keys_template
        )

    def child_params(self, params: Params) -> tuple[Params, ...]:
        return self.value[1](params)

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

        residuals = tuple(
            construction.assem_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in argkeys), *params
            )
            for construction, argkeys, params in zip(
                flat_constructions, flat_prim_keys, flat_params
            )
        )
        return jnp.concatenate(residuals)

    def assem_atleast_1d(self, prims: Prims, *params: Params) -> NDArray:
        return jnp.atleast_1d(self.assem(prims, *params))

    @classmethod
    def assem(cls, prims: Prims, *params: Params) -> NDArray:
        """
        Return the (local) construction output

        The full construction output consists of recursively stacking all
        construction outputs together in the tree.

        Parameters
        ----------
        prims: ResPrims
            A tuple of primitives the construction applies to
        *params: Params
            A set of parameters for the residual

            These are things like length, distance, angle, etc.

        Returns
        -------
        NDArray
        """
        # NOTE: Both *params and **kwargs affect construction outputs but are
        # used for different reasons.
        # Generally, `*params` involves direct changes to construction `assem`
        # methods while `**kwargs` involve changes to the tree structure of
        # child constraints.
        # Some changes to constructions can be implemented using either approach
        # so there is some overlap.
        raise NotImplementedError()


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
        super().__init__(
            *self.init_children(**kwargs), self.init_signature(**kwargs)
        )

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
    Constraint representing an array of child constraints
    """

    def __init__(self, shape: tuple[int, ...] = (0,)):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(shape=shape)


class StaticCompoundConstruction(CompoundConstruction):
    """
    Construction with static primitive argument types and child constructions

    To specify a `StaticConstraint`:
    - define `init_aux_data`,
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
    size_node = node_map(lambda value: value[-1]["RES_SIZE"], construction)
    cumsize_node = node_accumulate(lambda x, y: x + y, size_node, 0)

    flat_child_sizes = [
        tuple(child.value for child in node.values())
        for _, node in iter_flat("", cumsize_node)
    ]

    flat_construction_structs = [
        (key,) + transform_flat_constraint(cons, child_sizes)
        for (key, cons), child_sizes
        in zip(iter_flat("", construction), flat_child_sizes)
    ]

    return unflatten(flat_construction_structs)[0]


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

        This is a tuple of:
        - primitive keys,
        - a method to split parameters into child parameters
        - and auxiliary data describing the construction.

        See `ConstraintNode` for more details.
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
    CHILD_PRIM_KEYS, CHILD_PARAMS, AUX_DATA = construction.value

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

    # Define derived `aux_data`
    derived_aux_data = {
        "RES_ARG_TYPES": AUX_DATA["RES_ARG_TYPES"],
        "RES_PARAMS_TYPE": AUX_DATA["RES_PARAMS_TYPE"] + (np.ndarray,),
        "RES_SIZE": AUX_DATA["RES_SIZE"],
    }

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

    derived_value = (CHILD_PRIM_KEYS, derived_child_params, derived_aux_data)
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


def transform_map(construction: TCons, PrimTypes: list[type[pr.Primitive]]):
    """
    Return a derived construction that maps over an array of primitives

    See `transform_map` for more details.

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
    PRIM_KEYS, CHILD_PRIMS, AUX_DATA = construction.value
    N = len(PrimTypes)
    num_prims = len(AUX_DATA["RES_ARG_TYPES"])
    num_params = len(AUX_DATA["RES_PARAMS_TYPE"])
    num_constr = max(N - num_prims + 1, 0)

    child_keys = tuple(
        f"{type(construction).__name__}{n}" for n in range(num_constr)
    )
    child_constraints = num_constr*(construction,)
    child_prim_keys = tuple(
        tuple(f"arg{ii}" for ii in range(n, n + num_prims))
        for n in range(num_constr)
    )
    def child_params(map_params):
        return tuple(
            map_params[n * num_params : (n + 1) * num_params]
            for n in range(num_constr)
        )

    map_aux_data = {
        "RES_ARG_TYPES": N * AUX_DATA["RES_ARG_TYPES"],
        "RES_PARAMS_TYPE": N * AUX_DATA["RES_PARAMS_TYPE"],
        "RES_SIZE": 0,
    }

    class MapConstruction(ConstructionNode):

        def __init__(self):
            super().__init__(child_keys, child_constraints, child_prim_keys, child_params, map_aux_data)

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

    def transform_SumConstruction(a: TCons, b: TCons) -> ConstructionNode:

        child_keys_a = cons_a.keys()
        child_keys_b = cons_b.keys()

        child_prim_keys_a, child_params_a, signature_a = cons_a.value
        child_prim_keys_b, child_params_b, signature_b = cons_b.value

        size_a = signature_a["RES_PARAMS_TYPE"]
        size_b = signature_b["RES_PARAMS_TYPE"]

        assert size_a == size_b

        assert child_prim_keys_a == child_prim_keys_b
        sum_child_prim_keys = child_prim_keys_a

        assert child_keys_a == child_keys_b
        sum_child_keys = child_keys_a

        param_types_a = signature_a["RES_PARAMS_TYPE"]
        param_types_b = signature_b["RES_PARAMS_TYPE"]

        prim_types_a = signature_a["RES_ARG_TYPES"]
        prim_types_b = signature_b["RES_ARG_TYPES"]

        sum_signature = {
            'RES_SIZE': signature_a['RES_SIZE'],
            'RES_ARG_TYPES': param_types_a + param_types_b,
            'RES_PARAMS_TYPE': prim_types_a + prim_types_b
        }

        def sum_child_params(sum_params: Params) -> tuple[Params, ...]:
            param_chunks = (len(param_types_a), len(param_types_b))
            params_a, params_b = tuple(chunk(sum_params, param_chunks))
            return (
                (ca, cb) for ca, cb
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
                return a.assem(prims_a, *params_a) + b.assem(prims_b, *params_b)

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
            mul_signature = {
                'RES_SIZE': signature['RES_SIZE'],
                'RES_ARG_TYPES': signature['RES_ARG_TYPES'],
                'RES_PARAMS_TYPE': signature['RES_PARAMS_TYPE'] + (float,)
            }
        elif isinstance(scalar, (float, int)):
            class ScalarMultipleConstruction(ConstructionNode):

                @classmethod
                def assem(cls, prims: Prims, *params: Params) -> NDArray:
                    return scalar * cons_a.assem(prims, *params)

            def mul_child_params(params: Params) -> tuple[Params, ...]:
                return child_params(params)

            mul_signature = {
                'RES_SIZE': signature['RES_SIZE'],
                'RES_ARG_TYPES': signature['RES_ARG_TYPES'],
                'RES_PARAMS_TYPE': signature['RES_PARAMS_TYPE']
            }
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

# TODO: Add `relative` constraint to derive a relative constraint?

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
        def aux_data(
            value_size: int, params: tuple[any] = ()
        ):
            return {
                "RES_ARG_TYPES": arg_types,
                "RES_SIZE": value_size,
                "RES_PARAMS_TYPE": params,
            }

    return PrimsSignature


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


## Point constructions
# NOTE: These are actual constraint classes that can be called so class docstrings
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
        return cls.aux_data(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Point]):
        """
        Return the location error for a point
        """
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
        return cls.aux_data(1, (np.ndarray,))

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
        return cls.aux_data(1, ())

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
        return cls.aux_data(1, ())

    @classmethod
    def assem(self, prims: tuple[pr.Point, pr.Point]):
        return DirectedDistance.assem(prims, np.array([0, 1]))


## Line constructions

# Argument type: tuple[Line,]


class LineVector(LeafConstruction, _LineSignature):
    """
    Return the line vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        """
        Return the length error of a line
        """
        (line,) = prims
        pointa, pointb = line.values()
        return pointb.value - pointa.value


class UnitLineVector(LeafConstruction, _LineSignature):
    """
    Return the unit line vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(2)

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
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return jnp.sum(LineVector.assem((line,)) ** 2) ** (1 / 2)


class DirectedLength(LeafConstruction, _LineSignature):
    """
    Return the length of a line along a vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    direction: NDArray
        The direction
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line], direction: NDArray):
        (line,) = prims
        return jnp.dot(LineVector.assem((line,)), direction)


class XLength(LeafConstruction, _LineSignature):
    """
    Constrain the length of a line projected along the x direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([1, 0]))


class YLength(LeafConstruction, _LineSignature):
    """
    Constrain the length of a line projected along the y direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([0, 1]))


class Midpoint(LeafConstruction, _LineSignature):
    @classmethod
    def init_signature(cls):
        return cls.aux_data(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return 1 / 2 * (line["Point0"].value + line["Point1"].value)


# Argument type: tuple[Line, Line]


class MidpointDirectedDistance(LeafConstruction, _LineLineSignature):
    """
    Return the directed distance between two line midpoints

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
        return cls.aux_data(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], direction: NDArray):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims
        return jnp.dot(Midpoint.assem((line1,)) - Midpoint.assem((line0,)), direction)


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
        return cls.aux_data(1)

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
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
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
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        line0, line1 = prims
        dir0 = UnitLineVector.assem((line0,))
        dir1 = UnitLineVector.assem((line1,))
        return jnp.arccos(jnp.dot(dir0, dir1))


# Argument type: tuple[Line, ...]


## Point and Line constraints

# Argument type: tuple[Point, Line]


class PointOnLineDistance(LeafConstruction, _PointLineSignature):
    """
    Return the projected distance of a point along a line

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Line]
        The point and line
    reverse: bool
        A boolean indicating whether to reverse the line direction

        The distance of the point on the line is measured either from the start or end
        point of the line based on `reverse`. If `reverse=False` then the start point is
        used.
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1, (bool,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Line], reverse: bool):
        """
        Return the projected distance error of a point along a line
        """
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

        By convention the orthogonal direction points to the left of the line relative
        to the line direction. If `reverse=True` then the orthogonal direction points to
        the right of the line.
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1, (bool,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Line], reverse: bool):
        """
        Return the projected distance error of a point to a line
        """
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

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The quad
    """

    @classmethod
    def init_signature(cls):
        return cls.aux_data(1)

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

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
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
        return cls.aux_data(0, ())


class InnerMargin(CompoundConstruction, _QuadrilateralQuadrilateralSignature):
    """
    Return the inner margin between two quadrilaterals

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
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
        return cls.aux_data(0, ())


# Argument type: tuple[Quadrilateral, ...]


## Axes constructions

# Argument type: tuple[Axes]
