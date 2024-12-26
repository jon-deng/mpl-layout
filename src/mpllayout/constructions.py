"""
Geometric constructions

Constructions are functions that accept primitives and return a vector.
For example, this could be the coordinates of a point, the angle between two
lines, or the length of a single line.
"""

from typing import Callable, Optional, Any
from numpy.typing import NDArray

from collections import namedtuple
import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat
from .containers import map as node_map, accumulate as node_accumulate

Params = tuple[Any, ...]

Prims = tuple[pr.Primitive, ...]
PrimKeys = tuple[str, ...]
PrimTypes = tuple[type[pr.Primitive], ...]


def load_named_tuple(NamedTuple: namedtuple, args: dict[str, Any] | tuple[Any, ...]):
    if isinstance(args, dict):
        args = NamedTuple(**args)
    elif isinstance(args, tuple):
        args = NamedTuple(*args)
    elif isinstance(args, NamedTuple):
        pass
    else:
        raise TypeError()
    return args


class PrimKeysNode(Node[PrimKeys, "PrimKeysNode"]):
    """
    A tree of primitive keys indicating primitives for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding primitive keys node.
    """

    pass


class ParamsNode(Node[Params, "ParamsNode"]):
    """
    A tree of residual parameters (kwargs) for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding parameters node.
    """

    pass


class ConstructionNode(Node[tuple[PrimKeys, ...], "ConstructionNode"]):
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
    child_constructions: list[Construction]
        Child constructions
    child_prim_keys: tuple[PrimKeys, ...]
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
    aux_data: Mapping[str, Any]
        Any auxiliary data

        This is usually for type checking/validation of inputs
    """

    # TODO: Implement `assem` type checking using `aux_data`
    def __init__(
        self,
        child_keys: list[str],
        child_constructions: list["ConstructionNode"],
        child_prim_keys: list[PrimKeys],
        child_params: Callable[[Params], list[Params]],
        aux_data: Optional[dict[str, Any]] = None,
    ):
        children = {key: child for key, child in zip(child_keys, child_constructions)}
        super().__init__((child_prim_keys, child_params, aux_data), children)

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

    def __call__(self, prims: Prims, *params):
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
    ):
        flat_constructions = (x for _, x in iter_flat("", self))
        flat_prim_keys = (x.value for _, x in iter_flat("", root_prim_keys))
        flat_params = (x.value for _, x in iter_flat("", root_params))

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
        raise NotImplementedError()


ChildKeys = tuple[str, ...]
ChildConstraints = tuple[ConstructionNode, ...]


class Construction(ConstructionNode):
    """
    Construction with parameterized primitive argument types and child constructions

    To specify a `ParameterizedConstraint`:
    - define `init_aux_data`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the construction will have no child
    constructions by default.

    If `split_children_params` is undefined, all child constructions will be passed
    empty parameters, and therefore use default values.

    Parameters
    ----------
    **kwargs
        Parameter controlling the construction definition

        Subclasses should define what these keyword arguments are.
    """

    def __init__(self, **kwargs):
        (
            c_keys,
            c_construction_types,
            c_construction_type_kwargs,
            c_prim_keys,
            c_params,
        ) = self.init_children(**kwargs)
        c_constructions = tuple(
            ConstructionType(**kwargs)
            for ConstructionType, kwargs in zip(
                c_construction_types, c_construction_type_kwargs
            )
        )
        super().__init__(
            c_keys, c_constructions, c_prim_keys, c_params, self.init_aux_data(**kwargs)
        )

    @classmethod
    def init_children(cls, **kwargs) -> tuple[
        list[str],
        list[type["Construction"]],
        list[dict[str, any]],
        list[PrimKeys],
        Callable[[Params], list[Params]],
    ]:
        raise NotImplementedError()

    @classmethod
    def init_aux_data(cls, **kwargs) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def assem(cls, prims: Prims, *params):
        raise NotImplementedError()


class CompoundConstruction(Construction):

    @classmethod
    def assem(cls, prims: Prims, *params):
        return np.array([])


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

    To specify a `LeafConstruction`, define `assem` and `init_aux_data`
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def init_children(
        cls,
    ) -> tuple[
        list[str],
        list[type["Construction"]],
        list[dict[str, any]],
        list[PrimKeys],
        Callable[[Params], list[Params]],
    ]:
        return (), (), {}, (), lambda x: ()


## Construction functions

# These functions accept one or more `Construction` class instances


def generate_construction_type_node(
    ConstructionType: type[Construction], **kwargs
) -> Node:
    """
    Return a tree representing the construction type
    """
    cons_type_children = ConstructionType.init_children(**kwargs)
    aux_data = ConstructionType.init_aux_data(**kwargs)

    c_keys, c_cons_types, c_cons_type_kwargs, *_ = cons_type_children

    children = {
        key: generate_construction_type_node(ChildConstructionType, **type_kwargs)
        for key, ChildConstructionType, type_kwargs in zip(
            c_keys, c_cons_types, c_cons_type_kwargs
        )
    }
    return Node((cons_type_children, aux_data), children)


def generate_constraint(
    ConstructionType: type[Construction],
    construction_name: str,
    construction_output_size: Optional[Node[int, Node]] = None,
):
    if issubclass(ConstructionType, CompoundConstruction):

        def derived_assem(prims, derived_params):
            *params, value = derived_params
            return np.array([])

    elif issubclass(ConstructionType, LeafConstruction):

        def derived_assem(prims, derived_params):
            *params, value = derived_params
            return ConstructionType.assem(prims, *params) - value

    else:
        raise TypeError()

    class DerivedConstraint(ConstructionType):

        @classmethod
        def init_children(cls, **kwargs):
            (
                c_keys,
                c_construction_types,
                c_construction_type_kwargs,
                c_prim_keys,
                c_params,
            ) = ConstructionType.init_children(**kwargs)

            if construction_output_size is None:
                cons_node = generate_construction_type_node(ConstructionType, **kwargs)

                _construction_output_size = node_accumulate(
                    lambda x, y: x + y,
                    node_map(lambda value: value[1]["RES_SIZE"], cons_node),
                    0,
                )
            else:
                _construction_output_size = construction_output_size

            derived_child_construction_types = tuple(
                generate_constraint(
                    ChConstraintType, child_key, _construction_output_size[child_key]
                )
                for child_key, ChConstraintType in zip(c_keys, c_construction_types)
            )

            child_res_sizes = [
                node.value for node in _construction_output_size.values()
            ]
            cum_child_res_sizes = [
                size for size in itertools.accumulate([0] + child_res_sizes, initial=0)
            ]
            child_value_slices = [
                slice(start, stop)
                for start, stop in zip(
                    cum_child_res_sizes[:-1], cum_child_res_sizes[1:]
                )
            ]

            def child_value(value):
                if isinstance(value, (float, int)):
                    return len(c_keys) * (value,)
                else:
                    return tuple(value[idx] for idx in child_value_slices)

            def derived_child_params(derived_params):
                *params, value = derived_params
                return tuple(
                    (*params, value)
                    for params, value in zip(c_params(params), child_value(value))
                )

            return (
                c_keys,
                derived_child_construction_types,
                c_construction_type_kwargs,
                c_prim_keys,
                derived_child_params,
            )

        @classmethod
        def init_aux_data(cls, **kwargs):
            aux_data = ConstructionType.init_aux_data(**kwargs)
            derived_aux_data = {
                "RES_ARG_TYPES": aux_data["RES_ARG_TYPES"],
                "RES_PARAMS_TYPE": namedtuple(
                    "Parameters", aux_data["RES_PARAMS_TYPE"]._fields + ("value",)
                ),
                "RES_SIZE": aux_data["RES_SIZE"],
            }
            return derived_aux_data

        @classmethod
        def assem(cls, prims, *derived_params):
            return derived_assem(prims, derived_params)

    DerivedConstraint.__name__ = construction_name

    return DerivedConstraint


def map(ConstructionType: type[Construction], PrimTypes: list[type[pr.Primitive]]):
    N = len(PrimTypes)

    class MapConstruction(CompoundConstruction):

        @classmethod
        def init_children(cls, **kwargs):
            c_keys = tuple(f"MAP{n}" for n in range(N))
            c_construction_types = N * (ConstructionType,)
            c_construction_type_kwargs = N * (kwargs,)

            n_prims = len(ConstructionType.init_aux_data(**kwargs)["RES_ARG_TYPES"])
            n_params = len(
                ConstructionType.init_aux_data(**kwargs)["RES_PARAMS_TYPE"]._fields
            )
            n_cons = N - (n_prims - 1)
            assert n_cons >= 0

            c_prim_keys = tuple(
                tuple(f"arg{ii}" for ii in range(n, n + n_prims)) for n in range(n_cons)
            )

            def child_params(map_params):
                # breakpoint()
                return tuple(
                    map_params[n * n_params : (n + 1) * n_params] for n in range(n_cons)
                )

            return (
                c_keys,
                c_construction_types,
                c_construction_type_kwargs,
                c_prim_keys,
                child_params,
            )

        @classmethod
        def init_aux_data(cls, **kwargs):
            aux_data = ConstructionType.init_aux_data(**kwargs)
            derived_aux_data = {
                "RES_ARG_TYPES": N * aux_data["RES_ARG_TYPES"],
                "RES_PARAMS_TYPE": namedtuple(
                    "Parameters", N * aux_data["RES_PARAMS_TYPE"]._fields
                ),
                "RES_SIZE": aux_data["RES_SIZE"],
            }
            return derived_aux_data

        @classmethod
        def assem(cls, prims, *map_params):
            return np.array(())

    MapConstruction.__name__ = f"Map{ConstructionType.__name__}"

    return MapConstruction


# TODO: Add `relative` functions to derive a relative constraint over
# primitive iterables

## Construction signatures


def make_signature_class(arg_types: tuple[type[pr.Primitive], ...]):

    class PrimsSignature:
        @staticmethod
        def aux_data(
            value_size: int, params: tuple[any] = namedtuple("Parameters", ())
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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ("direction",)))

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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ()))

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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ()))

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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ("direction",)))

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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([0, 1]))


class Midpoint(LeafConstruction, _LineSignature):
    @classmethod
    def init_aux_data(cls):
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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ("direction",)))

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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ("reverse",)))

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
    def init_aux_data(cls):
        return cls.aux_data(1, namedtuple("Parameters", ("reverse",)))

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
    def init_aux_data(cls):
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
            c_keys = ("LeftMargin",)
            c_construction_types = (MidpointXDistance,)
            c_prim_keys = (("arg1/Line1", "arg0/Line3"),)
        elif side == "right":
            c_keys = ("RightMargin",)
            c_construction_types = (MidpointXDistance,)
            c_prim_keys = (("arg0/Line1", "arg1/Line3"),)
        elif side == "bottom":
            c_keys = ("BottomMargin",)
            c_construction_types = (MidpointYDistance,)
            c_prim_keys = (("arg1/Line2", "arg0/Line0"),)
        elif side == "top":
            c_keys = ("TopMargin",)
            c_construction_types = (MidpointYDistance,)
            c_prim_keys = (("arg0/Line2", "arg1/Line0"),)
        else:
            raise ValueError()

        c_construction_type_kwargs = ({},)

        def c_params(params):
            return [()]

        return (
            c_keys,
            c_construction_types,
            c_construction_type_kwargs,
            c_prim_keys,
            c_params,
        )

    @classmethod
    def init_aux_data(cls, side: str = "left"):
        return cls.aux_data(0, namedtuple("Parameters", ("side",)))


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
            c_keys = ("LeftMargin",)
            c_construction_types = (MidpointXDistance,)
            c_prim_keys = (("arg1/Line3", "arg0/Line3"),)
        elif side == "right":
            c_keys = ("RightMargin",)
            c_construction_types = (MidpointXDistance,)
            c_prim_keys = (("arg0/Line1", "arg1/Line1"),)
        elif side == "bottom":
            c_keys = ("BottomMargin",)
            c_construction_types = (MidpointYDistance,)
            c_prim_keys = (("arg1/Line0", "arg0/Line0"),)
        elif side == "top":
            c_keys = ("TopMargin",)
            c_construction_types = (MidpointYDistance,)
            c_prim_keys = (("arg0/Line2", "arg1/Line2"),)
        else:
            raise ValueError()

        c_construction_type_kwargs = ({},)

        def c_params(params):
            return [()]

        return (
            c_keys,
            c_construction_types,
            c_construction_type_kwargs,
            c_prim_keys,
            c_params,
        )

    @classmethod
    def init_aux_data(cls, side: str = "left"):
        return cls.aux_data(0, namedtuple("Parameters", ("side",)))


# Argument type: tuple[Quadrilateral, ...]


## Axes constructions

# Argument type: tuple[Axes]
