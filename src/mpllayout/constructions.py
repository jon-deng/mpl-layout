"""
Geometric constructions

Constructions are functions that accept primitives and return a vector.
For example, this could be the coordinates of a point, the angle between two
lines, or the length of a single line.
"""

from typing import Callable, Optional, Any, TypeVar, NamedTuple, Literal
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

class ConstructionSignature(NamedTuple):
    prim_types: PrimTypes
    param_types: ParamTypes
    value_size: ArraySize

ChildParams = Callable[[Params], list[Params]]

class ConstructionValue(NamedTuple):
    child_prim_keys: list[PrimKeys]
    child_params: ChildParams
    signature: ConstructionSignature


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
        A named tuple specifying the signature and child constructions

        A construction value is a named tuple of three objects describing
        the signature of the construction and how child constructions are
        evaluated
            ``child_prim_keys, child_params, signature = value``
        The tuple fields are further described below.

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
            A named tuple describing intput and output types for `assem`

            This a tuple with field `prim_types`, `param_types` and
            `value_size`
                ``prim_types, param_types, value_size = signature``,
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
        if not isinstance(value, ConstructionValue):
            raise TypeError(f"`value` must be a `ConstructionValue` not {type(value)}")
        elif len(value) != 3:
            raise ValueError(f"`value` must have 3 items not {len(value)}")

        # Check each component of `value`

        # Check `child_prim_keys` has one `PrimKeys` tuple for each child
        if not isinstance(value.child_prim_keys, (tuple, list)):
            raise TypeError(f"`value[0]` must be a tuple or list of `PrimKeys`")
        elif len(value.child_prim_keys) != len(children):
            raise ValueError(
                f"`value.child_prim_keys` must have length {len(children)}"
                " , matching the number of children"
            )

        # Check each `child_prim_keys` tuple indexes the right number of child
        # prims for the corresponding child construction
        child_prim_types = {
            key: child.value.signature.prim_types
            for key, child in children.items()
        }
        valid_child_prim_keys = {
            key: len(prim_types) == len(prim_keys)
            for (key, prim_types), prim_keys
            in zip(child_prim_types.items(), value.child_prim_keys)
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
        if not isinstance(value.child_params, Callable):
            raise TypeError(f"value.child_params must be `Callable`")

        # Check `signature` is a `ConstructionSignature`
        if not isinstance(value.signature, ConstructionSignature):
            raise TypeError(f"value.signature must be `ConstructionSignature`")

        ## Check `children`

        # Check that all children are constructions
        if not all(
            isinstance(child, ConstructionNode)
            for _, child in children.items()
        ):
            raise TypeError("`children` must be a dictionary of constructions")

        super().__init__(value, children)

    ## Attributes related to `value`

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
            for child_prim_keys in self.value.child_prim_keys
        )

    def child_params(self, params: Params) -> tuple[Params, ...]:
        return self.value.child_params(params)

    @property
    def signature(self) -> ConstructionSignature:
        return self.value.signature

    ## Input validation functions

    def validate_prims(self, prims: Prims):
        """
        Raise an exception if primitives do not match the signature
        """
        if len(prims) != len(self.signature.prim_types):
            raise TypeError(f"Incorrect number of primitives")

        invalid_types = [
            not isinstance(prim, prim_type)
            for prim, prim_type in zip(prims, self.signature.prim_types)
        ]
        if any(invalid_types):
            raise TypeError(f"Incorrect primitive types")

    def validate_params(self, params: Params):
        """
        Raise an exception if parameters do not match the signature
        """
        if len(params) != len(self.signature.param_types):
            raise TypeError(f"Incorrect number of parameters")

        # NOTE: Didn't check parameter types because types are much looser
        # In many cases, a float will work in place of an NDArray, etc.

    def validate_value(self, value: NDArray):
        """
        Raise an exception if the output value has the wrong size
        """
        value_size = self.signature.value_size

        if value.size != value_size:
            raise ValueError(f"`value` should have size {value_size}")

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
            np.array(()), {key: prim for key, prim in zip(prim_keys, prims)}
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
        self.validate_prims(prims)
        self.validate_params(params)
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

        value = ConstructionValue(child_prim_keys, child_params, signature)
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


## Construction signatures


def make_signature_class(prim_types: tuple[type[pr.Primitive], ...]):

    class PrimsSignature:
        @staticmethod
        def make_signature(
            value_size: ArraySize, param_types: ParamTypes = ()
        ):
            return ConstructionSignature(prim_types, param_types, value_size)

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
# Argument type: tuple[()]

class Vector(Construction, _NullSignature):
    """
    Return a vector

    The vector is returned in a tree format where each node in the tree contains
    a chunk of the vector.

    Parameters
    ----------
    size_node: Optional[Node[int]]
        The size of each vector chunk contained in a node

    Methods
    -------
    assem(prims: tuple[()], value: float | NDArray)
        Return the vector chunk contained in `value`
    """

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

            if isinstance(value, (float, int)):
                value = value* np.ones(cumsize_node.value)

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
    def assem(cls, prims: tuple[()], value: float | NDArray):
        # FIXME: Does this work for an empty root node???
        # Seems that it would return a non-empty vector which would be wrong?
        return value


class Scalar(LeafConstruction, _NullSignature):
    """
    Return a scalar

    The scalar is returned from a single node construction.

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[()], value: float)
        Return the scalar contained in `value`
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (float,))

    @classmethod
    def assem(cls, prims: tuple[()], value: float):
        return value

## Point constructions
# NOTE: These are actual construction classes that can be called so class docstrings
# document there `assem_res` function.

# Argument type: tuple[Point]


class Coordinate(LeafConstruction, _PointSignature):
    """
    Return a point's coordinates

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Point])
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
    Return the distance between two points projected along a direction

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Point, pr.Point], direction: NDArray)
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Point], direction: NDArray):
        point0, point1 = prims
        return jnp.dot(
            Coordinate.assem((point1,)) - Coordinate.assem((point0,)),
            direction
        )


class XDistance(DirectedDistance):
    """
    Return the x-distance between two points

    See `DirectedDistance` with fixed `direction=np.array((1, 0))`.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, ())

    @classmethod
    def assem(self, prims: tuple[pr.Point, pr.Point]):
        return super().assem(prims, np.array([1, 0]))


class YDistance(DirectedDistance):
    """
    Return the y-distance between two points

    See `DirectedDistance` with fixed `direction=np.array((0, 1))`.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, ())

    @classmethod
    def assem(self, prims: tuple[pr.Point, pr.Point]):
        return super().assem(prims, np.array([0, 1]))


## Line constructions

# Argument type: tuple[Line]


class LineVector(LeafConstruction, _LineSignature):
    """
    Return a line segment vector

    The vector points from the start point to the end point of the line.

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        pointa, pointb = line.values()
        return Coordinate.assem((pointb,)) - Coordinate.assem((pointa,))


class UnitLineVector(LeafConstruction, _LineSignature):
    """
    Return the line unit direction vector

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
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
    Return the line length

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
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
    None

    Methods
    -------
    assem(prims: tuple[pr.Line], direction: NDArray)
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (np.ndarray,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line], direction: NDArray):
        (line,) = prims
        return jnp.dot(LineVector.assem((line,)), direction)


class XLength(DirectedLength):
    """
    Return the length of a line along the x axis

    See `DirectedLength` with fixed `direction=np.array((1, 0))`.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return super().assem(prims, np.array((1, 0)))


class YLength(DirectedLength):
    """
    Return the length of a line along the y axis

    See `DirectedLength` with fixed `direction=np.array((0, 1))`.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return super().assem(prims, np.array((0, 1)))


class Midpoint(LeafConstruction, _LineSignature):
    """
    Return the midpoint coordinate of a line

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
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
    Return the projected distance between line midpoints along a direction

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line], direction: NDArray)
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


class MidpointXDistance(MidpointDirectedDistance):
    """
    Return the x-distance between two line midpoints

    See `MidpointDirectedDistance` with fixed `direction=np.array((1, 0))`
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        return super().assem(prims, np.array([1, 0]))


class MidpointYDistance(MidpointDirectedDistance):
    """
    Return the y-distance between two line midpoints

    See `MidpointDirectedDistance` with fixed `direction=np.array((0, 1))`
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        return super().assem(prims, np.array([0, 1]))


class Angle(LeafConstruction, _LineLineSignature):
    """
    Return the angle between two lines

    TODO: Document the convention for angle sign (CW or CCW? 0 to 360?)

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line])
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


## Point and Line constructions

# Argument type: tuple[Point, Line]


class PointOnLineDistance(LeafConstruction, _PointLineSignature):
    """
    Return the distance of a point projected along a line

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Point, pr.Line], reverse: bool)

        `reverse` indicates whether the distance is measured from the line start
        point towards the line end point or vice-versa. If `reverse = False`
        then distance is measure from the start towards the end.
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
    None

    Methods
    -------
    assem(prims: tuple[pr.Point, pr.Line], reverse: bool)

        `reverse` indicates how the orthogonal direction is measured. By
        convention the orthogonal direction rotates the unit line vector 90
        degrees counter-clockwise. If `reverse = True` the orthogonal direction
        is rotated clockwise.
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


class RelativePointOnLineDistance(LeafConstruction, _PointLineSignature):
    """
    Return the fractional distance of a point projected along a line

    A value of 1 indicates the point has covered the entire line length.

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Point, pr.Line], reverse: bool)

        `reverse` indicates whether the distance is measured from the line start
        point towards the line end point or vice-versa. If `reverse = False`
        then distance is measure from the start towards the end.
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (bool,))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Point, pr.Line],
        reverse: bool
    ):
        point, line = prims
        proj_dist = PointOnLineDistance.assem(prims, reverse)
        line_length = Length.assem((line,))
        return jnp.array([proj_dist/line_length])

## Quad constructions

# Argument type: tuple[Quadrilateral]

class RectangleDim(CompoundConstruction, _QuadrilateralSignature):
    """
    Return a rectangle dimension (width or height)

    Parameters
    ----------
    side: Literal['width', 'height']

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral])
    """
    def __init__(
        self,
        dim: Literal['width', 'height']='width'
    ):
        super().__init__(dim=dim)

    @classmethod
    def init_children(
        cls,
        dim: Literal['width', 'height']='width'
    ):
        if dim == "width":
            keys = ("Width",)
            constructions = (XLength(),)
            prim_keys = (("arg0/Line0",),)
        elif dim == "height":
            keys = ("Height",)
            constructions = (YLength(),)
            prim_keys = (("arg0/Line1",),)
        else:
            raise ValueError()

        def child_params(params: Params) -> tuple[Params, ...]:
            return ((),)

        return (keys, constructions, prim_keys, child_params)

    @classmethod
    def init_signature(
        cls,
        dim: Literal['width', 'height']='width'
    ):
        return cls.make_signature(0)

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral]):
        return super().assem(prims)


class Width(RectangleDim):
    """
    Return a rectangle width

    See `RectangleDim` with fixed `dim='width'` for more details.
    """

    def __init__(self):
        super().__init__(dim='width')


class Height(RectangleDim):
    """
    Return a rectangle height

    See `RectangleDim` with fixed `dim='height'` for more details.
    """

    def __init__(self):
        super().__init__(dim='height')


class AspectRatio(LeafConstruction, _QuadrilateralSignature):
    """
    Return the aspect ratio of a quadrilateral

    This is ratio of the bottom width over the side height.

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral])
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

def opposite_side(
    side: Literal[
        'bottom', 'top', 'left', 'right', 'Line0', 'Line2', 'Line1', 'Line3'
    ]
):
    """
    Return the opposite side string representation
    """
    if side == 'bottom':
        return 'top'
    elif side == 'top':
        return 'bottom'
    elif side == 'left':
        return 'right'
    elif side == 'right':
        return 'left'
    elif side == 'Line0':
        return 'Line2'
    elif side == 'Line2':
        return 'Line0'
    elif side == 'Line1':
        return 'Line3'
    elif side == 'Line3':
        return 'Line1'
    else:
        raise ValueError()

class Margin(CompoundConstruction, _QuadrilateralQuadrilateralSignature):
    """
    Return the margin between two quadrilaterals

    Parameters
    ----------
    outer: bool
        Whether to compute the inner or outer margin

        The outer margin is the margin from the outside faces of a quadrilateral
        to the opposite face of another quadrilateral. The inner margin is the
        margin from the inside face of a quadrilateral to the same face of the
        quadrilateral.
    side: str
        The side of the quadrilateral to return a margin

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral, pr.Quadrilateral])
    """

    def __init__(self, outer: bool=True, side: str = "left"):
        super().__init__(outer=outer, side=side)

    @classmethod
    def init_children(cls, outer: bool=True, side: str = "left"):
        if side == "left":
            keys = ("LeftMargin",)
            constructions = (MidpointXDistance(),)
            if outer:
                prim_keys = (("arg1/Line1", "arg0/Line3"),)
            else:
                prim_keys = (("arg1/Line3", "arg0/Line3"),)
        elif side == "right":
            keys = ("RightMargin",)
            constructions = (MidpointXDistance(),)
            if outer:
                prim_keys = (("arg0/Line1", "arg1/Line3"),)
            else:
                prim_keys = (("arg0/Line1", "arg1/Line1"),)
        elif side == "bottom":
            keys = ("BottomMargin",)
            constructions = (MidpointYDistance(),)
            if outer:
                prim_keys = (("arg1/Line2", "arg0/Line0"),)
            else:
                prim_keys = (("arg1/Line0", "arg0/Line0"),)
        elif side == "top":
            keys = ("TopMargin",)
            constructions = (MidpointYDistance(),)
            if outer:
                prim_keys = (("arg0/Line2", "arg1/Line0"),)
            else:
                prim_keys = (("arg0/Line2", "arg1/Line2"),)
        else:
            raise ValueError()

        def child_params(params: Params) -> tuple[Params, ...]:
            return ((),)

        return (keys, constructions, prim_keys, child_params)

    @classmethod
    def init_signature(cls, outer: bool=True, side: str = "left"):
        return cls.make_signature(0, ())

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral, pr.Quadrilateral]):
        return super().assem(prims)


class OuterMargin(Margin):
    """
    Return the outer margin between two quadrilaterals

    See `Margin` with `outer=True` for more details.
    """

    def __init__(self, side: str = "left"):
        super().__init__(outer=True, side=side)


class InnerMargin(Margin):
    """
    Return the outer margin between two quadrilaterals

    See `Margin` with `outer=False` for more details.
    """

    def __init__(self, side: str = "left"):
        super().__init__(outer=False, side=side)


## Construction transform functions

# Utilities for constructions transforms
T = TypeVar('T')

def chunk(
    array: list[T], chunk_sizes: list[int]
) -> Iterable[list[T]]:
    """
    Return an iterable of array chunks

    Parameters
    ----------
    array: list[T]
        The array
    chunk_sizes: list[int]
        A list of chunk sizes to chunk the array

        The array is split into `len(chunk_sizes)` chunks with each chunk having
        the given size.

    Returns
    -------
    Iterable[list[T]]
        An iterable over the array chunks
    """

    slice_bounds = list(itertools.accumulate(chunk_sizes, initial=0))
    # cum_chunk_size = slice_bounds[-1]

    return (
        array[start:stop]
        for start, stop in zip(slice_bounds[:-1], slice_bounds[1:])
    )

ConcatPrims = (
    Callable[[Prims, Prims], Prims]
    | Callable[[PrimKeys, PrimKeys], PrimKeys]
)
SplitPrims = (
    Callable[[Prims], tuple[Prims, Prims]]
    | Callable[[PrimKeys], tuple[PrimKeys, PrimKeys]]
)

ConcatParams = Callable[[Params, Params], Params]
SplitParams = Callable[[Params], tuple[Params, Params]]

def concatenate_construction_inputs(
    sig_a: ConstructionSignature, sig_b: ConstructionSignature, value_size: int
) -> tuple[
    ConstructionSignature,
    tuple[ConcatPrims, SplitPrims],
    tuple[ConcatParams, SplitParams]
]:
    """
    Concatenate construction inputs

    Concatenating two constructions represents a generic construction that
    accepts the concatenated `prims` and `*params` of the two input
    constructions.

    To fully define the generic construction, you must also define a new `assem`
    method over the combined `prims` and `*params`.
    """
    cat_sig = concatenate_signature(sig_a, sig_b, value_size)
    helper_prims = concatenate_prims(sig_a, sig_b)
    helper_params = concatenate_params(sig_a, sig_b)
    return cat_sig, helper_prims, helper_params


def concatenate_signature(
    sig_a: ConstructionSignature, sig_b: ConstructionSignature, value_size: int
) -> ConstructionSignature:
    """
    Concatenate two construction signatures to form a combined construction value

    See `concatenate_construction_inputs` for more details.
    """

    cat_signature = ConstructionSignature(
        sig_a.prim_types + sig_b.prim_types,
        sig_a.param_types + sig_b.param_types,
        value_size
    )

    return cat_signature


def concatenate_prims(
    signature_a: ConstructionSignature, signature_b: ConstructionSignature
) -> tuple[ConcatPrims, SplitPrims]:
    """
    See `concatenate_construction_inputs` for more details.
    """

    def concat(prims_a: Prims, prims_b: Prims) -> Prims:
        return prims_a + prims_b

    chunks = (len(signature_a.prim_types), len(signature_b.prim_types))
    def split(cat_prims) -> tuple[Prims, Prims]:
        return tuple(chunk(cat_prims, chunks))

    return concat, split


def concatenate_params(
    signature_a: ConstructionSignature, signature_b: ConstructionSignature
) -> tuple[ConcatParams, SplitParams]:
    """
    See `concatenate_construction_inputs` for more details.
    """

    chunks = (len(signature_a.param_types), len(signature_b.param_types))

    def concat(params_a: Params, params_b: Params) -> Params:
        return params_a + params_b

    def split(cat_params) -> tuple[Params, Params]:
        return tuple(chunk(cat_params, chunks))

    return concat, split


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
        __doc__ = f"""
        Return the error between {ConstructionType} and an input value

        See {ConstructionType} for more details.

        Parameters
        ----------
        See {ConstructionType} for more details.

        Methods
        -------
        assem(derived_prims: Prims, *derived_params)

            The difference between {ConstructionType} and a value is returned in
            the following format.
            ``
            def assem(derived_prims: Prims, *derived_params):
                prims = derived_prims
                params, value = derived_params

                return original_construction.assem(prims, *params) - value
            ``
            Note that the new constraint has an additional appended parameter,
            `value`, representing the desired construction value.
        """

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
        return node_value.signature.value_size

    size_node = node_map(value_size, construction)

    vector = Vector(size_node)
    return transform_sum(construction, transform_scalar_mul(vector, -1))


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


# NOTE: Refactor `transform_map` to accept `*PrimTypes`?
# This would be a tuple of `PrimTypes` lists for each primitive in the
# construction `prims` parameter.
# This would allow you to treat constructions with multiple `prims` as
# single unary functions rather than the special treatment used now.

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
    PRIM_KEYS, CHILD_PRIMS, SIG = construction.value
    N = len(PrimTypes)
    # `M` is the number of additional arguments past one for the construction
    M = len(SIG.prim_types)-1

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
        num_params = len(SIG.param_types)
        return tuple(
            map_params[n * num_params : (n + 1) * num_params]
            for n in range(num_constr)
        )

    map_signature = ConstructionSignature(
        num_constr*SIG.prim_types[:1] + SIG.prim_types[1:],
        num_constr*SIG.param_types,
        0
    )

    class MapConstruction(ConstructionNode):

        def __init__(self):
            value = ConstructionValue(child_prim_keys, child_params, map_signature)
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

        # Check the two constructions have the same number of children / child keys
        child_keys_a = cons_a.keys()
        child_keys_b = cons_b.keys()

        assert child_keys_a == child_keys_b
        sum_child_keys = child_keys_a


        # Check the two constructions have the same output size
        value_a, value_b = cons_a.value, cons_b.value
        signature_a, signature_b = value_a.signature, value_b.signature
        assert signature_a.value_size == signature_b.value_size

        # Create the combined construction signature
        sum_signature, cat_split_prims, cat_split_params = concatenate_construction_inputs(
            signature_a, signature_b, signature_a.value_size
        )
        concat_prims, split_prims = cat_split_prims
        concat_params, split_params = cat_split_params

        # Build the sum construction `ConstructionValue` tuple
        sum_child_prim_keys = tuple(
            concat_prims(prim_keys_a, prim_keys_b) for prim_keys_a, prim_keys_b
            in zip(value_a.child_prim_keys, value_b.child_prim_keys)
        )

        def sum_child_params(sum_params: Params) -> tuple[Params, ...]:
            params_a, params_b = split_params(sum_params)
            return (
                concat_params(ca, cb) for ca, cb
                in zip(value_a.child_params(params_a), value_b.child_params(params_b))
            )

        node_value = ConstructionValue(
            sum_child_prim_keys, sum_child_params, sum_signature
        )

        class SumConstruction(ConstructionNode):

            @classmethod
            def assem(cls, sum_prims: Prims, *sum_params: Params) -> NDArray:
                prims_a, prims_b = split_prims(sum_prims)
                params_a, params_b = split_params(sum_params)
                return cons_a.assem(prims_a, *params_a) + cons_b.assem(prims_b, *params_b)

        return SumConstruction, node_value, sum_child_keys

    flat_a = [a for a in iter_flat("", cons_a)]
    flat_b = [b for b in iter_flat("", cons_b)]

    flat_sum_constructions = [
        (key, *transform_SumConstruction(a, b))
        for (key, a), (_, b) in zip(flat_a, flat_b)
    ]

    return unflatten(flat_sum_constructions)[0]


# TODO: Implement a parameter freezing transform which fixes a parameter
# Then constant `Scalar` and `Vector` could be represented by freezing
# `value` parameter

def transform_scalar_mul(cons_a: TCons, scalar: float | Scalar) -> ConstructionNode:
    """
    Return a construction representing a construction multiplied by a scalar
    """
    # If `scalar` is `Scalar` a new scalar parameter is added
    # If `scalar` is `float`, then the scalar float is used to multiply the
    # construction and no additional parameter is added

    def transform_ScalarMultiple(
        cons_a: TCons, scalar: float | Scalar
    ) -> ConstructionNode:
        child_keys = cons_a.keys()
        value_a = cons_a.value
        signature_a = value_a.signature

        if isinstance(scalar, Scalar):
            signature_scalar = scalar.value.signature

            mul_signature, cat_split_prims, cat_split_params = concatenate_construction_inputs(
                signature_a, signature_scalar, signature_a.value_size
            )
            concat_prims, split_prims = cat_split_prims
            concat_params, split_params = cat_split_params

            class ScalarMultipleConstruction(ConstructionNode):

                @classmethod
                def assem(cls, prims: Prims, *params: Params) -> NDArray:
                    prims_a, prims_scalar = split_prims(prims)
                    params_a, params_scalar = split_params(params)
                    return (
                        scalar.assem(prims_scalar, *params_scalar)
                        * cons_a.assem(prims_a, *params_a)
                    )

            def mul_child_params(params: Params) -> tuple[Params, ...]:
                params_a, params_scalar = split_params(params)
                return tuple(
                    concat_params(child_cons_params, params_scalar)
                    for child_cons_params in value_a.child_params(params_a)
                )

        elif isinstance(scalar, (float, int)):
            mul_signature = signature_a

            class ScalarMultipleConstruction(ConstructionNode):

                @classmethod
                def assem(cls, prims: Prims, *params: Params) -> NDArray:
                    return scalar * cons_a.assem(prims, *params)

            def mul_child_params(params: Params) -> tuple[Params, ...]:
                return value_a.child_params(params)

        else:
            raise TypeError(
                f"`scalar` must be `float | int | Scalar` not `{type(scalar)}`"
            )

        node_value = ConstructionValue(
            cons_a.value.child_prim_keys, mul_child_params, mul_signature
        )
        return ScalarMultipleConstruction, node_value, child_keys

    flat_a = [a for a in iter_flat("", cons_a)]
    flat_sum_constructions = [
        (key, *transform_ScalarMultiple(a, scalar)) for key, a in flat_a
    ]

    return unflatten(flat_sum_constructions)[0]
