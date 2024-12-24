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

Params = tuple[Any, ...]

Prims = tuple[pr.Primitive, ...]
PrimKeys = tuple[str, ...]
PrimTypes = tuple[type[pr.Primitive], ...]


def load_named_tuple(
        NamedTuple: namedtuple,
        args: dict[str, Any] | tuple[Any, ...]
    ):
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
    The condition is implemented through a function
        `assem(self, prims, *args)`
    where
    - `prims` are the geometric primitives to construction
    - and `*args` are additional arguments for the residual.
    The construction is satisified when `assem(self, prims, *args) == 0`.
    To implement `assem`, `jax` functions should be used to return a
    residual vector from the parameter vectors of input primitives.

    Constraints have a tree-like structure.
    A construction can contain child constructions by passing subsets of its input
    primitives (`prims` in `assem`) on to child constructions.
    The residual of a construction is the result of concatenating all child
    construction residuals.

    To create a construction, subclass `Construction` then:
        1. Define the residual for the construction (`assem`)
        2. Specify the parameters for `Construction.__init__` (see `Parameters`)\
        3. Define `Construction.split_children_params` (see below)
    Note that some of the `Construction.__init__` parameters are for type checking
    inputs to `assem` while the others are for specifying child constructions.
    `StaticConstraint` and `ParameterizedConstraint` are two `Construction`
    subclasses that can be subclassed to create constructions.

    Parameters
    ----------
    child_prim_keys: tuple[PrimKeys, ...]
        Primitive key tuples for each child construction

        This is stored as the "value" of the tree structure and encodes how to
        create primitives for each child construction from the parent construction's
        `prims` (in `assem(self, prims, *args)`).
        For each child construction, a tuple of primitive keys indicates a
        subset of parent primitives to form child construction primitive
        arguments.

        To illustrate this, consider a parent construction with residual

        ```python
        Parent.assem(self, prims, *args)
        ```

        and the n'th child construction with primitive key tuple

        ```python
        child_prim_keys[n] == ('arg0', 'arg3/Line2')
        ```.

        This indicates the n'th child construction should be evaluated with

        ```
        child_prims = (prims[0], prims[3]['Line2'])
        ```
    child_keys: list[str]
        Keys for any child constructions
    child_constructions: list[Construction]
        Child constructions
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
        aux_data: Optional[dict[str, Any]] = None
    ):
        self.propogate_child_params = child_params
        children = {
            key: child for key, child in zip(child_keys, child_constructions)
        }
        super().__init__((child_prim_keys, aux_data), children)

    def propogate_child_params(self, params: Params) -> tuple[Params, ...]:
        """
        Return children construction parameters from parent construction parameters
        """
        raise NotImplementedError()

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
        children_params = self.propogate_child_params(params)
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
        # Replace the first 'arg{n}/...' key with the appropriate parent argument keys

        def parent_argnum_from_key(arg_key: str):
            arg_number_str = arg_key.split("/", 1)[0]
            if arg_number_str[:3] == "arg":
                arg_number = int(arg_number_str[3:])
            else:
                raise ValueError(f"Argument key, {arg_key}, must contain 'arg' prefix")
            return arg_number

        def replace_prim_key_prefix(arg_key: str, parent_prim_keys):
            split_key = arg_key.split("/", 1)
            prefix, postfix = split_key[0], split_key[1:]
            new_prefix = parent_prim_keys[parent_argnum_from_key(arg_key)]
            return "/".join([new_prefix] + postfix)

        # For each child, find the parent primitive part of its argument tuple
        children_prim_keys = tuple(
            tuple(
                replace_prim_key_prefix(prim_key, prim_keys)
                for prim_key in child_prim_keys
            )
            for child_prim_keys in self.child_prim_keys
        )

        children = {
            key: child.root_prim_keys(child_prim_keys)
            for (key, child), child_prim_keys
            in zip(self.items(), children_prim_keys)
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
    def child_prim_keys(self):
        return self.value[0]

    @property
    def RES_PARAMS_TYPE(self):
        return self.value[1]['RES_PARAMS_TYPE']

    def __call__(
            self,
            prims: Prims,
            *params
        ):
        prim_keys = tuple(f'arg{n}' for n, _ in enumerate(prims))
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
            for construction, argkeys, params in zip(flat_constructions, flat_prim_keys, flat_params)
        )
        return jnp.concatenate(residuals)

    def assem_atleast_1d(
            self, prims: Prims, *params: Params
        ) -> NDArray:
        return jnp.atleast_1d(self.assem(prims, *params))

    @classmethod
    def assem(
            cls, prims: Prims, *params
        ) -> NDArray:
        """
        Return a residual vector representing the construction satisfaction

        Parameters
        ----------
        prims: ResPrims
            A tuple of primitives the construction applies to
        *args:
            A set of parameters for the residual

            These are things like length, distance, angle, etc.

        Returns
        -------
        NDArray
            The residual representing whether the construction is satisfied. The
            construction is satisfied when the residual is 0.
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
        super().__init__(*self.init_children(**kwargs), self.init_aux_data(**kwargs))

    @classmethod
    def init_children(
        cls, **kwargs
    ) -> tuple[
        list[str],
        list[ConstructionNode],
        list[PrimKeys],
        Callable[[Params], list[Params]]
    ]:
        raise NotImplementedError()

    @classmethod
    def init_aux_data(
        cls, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def assem(cls, prims: Prims, *params):
        raise NotImplementedError()


class CompoundConstruction(Construction):

    @classmethod
    def assem(cls, prims: Prims, *params):
        return np.array([])


class ArrayConstruction(CompoundConstruction):
    """
    Constraint representing an array of child constraints
    """

    def __init__(self, shape: tuple[int, ...]=(0,)):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(shape=shape)


class StaticConstruction(CompoundConstruction):
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

    @classmethod
    def init_children(
        cls
    ) -> tuple[
        list[str],
        list[ConstructionNode],
        list[PrimKeys],
        Callable[[Params], list[Params]]
    ]:
        raise NotImplementedError()

    @classmethod
    def init_aux_data(
        cls
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def assem(cls, prims: Prims, *params):
        return np.array([])


class LeafConstruction(Construction):
    """
    Construction without any child constructions

    To specify a `LeafConstruction`, define `assem` and `init_aux_data`
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def init_children(
        cls
    ) -> tuple[
        list[str],
        list[ConstructionNode],
        list[PrimKeys],
        Callable[[Params], list[Params]]
    ]:
        return [], [], [], lambda x: ()


## Point constructions
# NOTE: These are actual constraint classes that can be called so class docstrings
# document there `assem_res` function.

# Argument type: tuple[Point,]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Point,),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class Coordinate(LeafConstruction):
    """
    Return point coordinates

    Parameters
    ----------
    prims: tuple[pr.Point]
        The point
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Point]):
        """
        Return the location error for a point
        """
        point, = prims
        return point.value

# Argument type: tuple[Point, Point]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Point, pr.Point),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class DirectedDistance(LeafConstruction):
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
        return _AUX_DATA(1, namedtuple('Parameters', ('direction',)))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Point, pr.Point],
        direction: NDArray
    ):
        point0, point1 = prims
        return jnp.dot(point1.value - point0.value, direction)


class XDistance(LeafConstruction):
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
        return _AUX_DATA(1)

    @classmethod
    def assem(
        self, prims: tuple[pr.Point, pr.Point]
    ):
        return DirectedDistance.assem(prims, np.array([1, 0]))


class YDistance(LeafConstruction):
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
        return _AUX_DATA(1)

    @classmethod
    def assem(
        self, prims: tuple[pr.Point, pr.Point]
    ):
        return DirectedDistance.assem(prims, np.array([0, 1]))

## Line constructions

# Argument type: tuple[Line,]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Line,),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class LineVector(LeafConstruction):
    """
    Return the line vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        """
        Return the length error of a line
        """
        line, = prims
        pointa, pointb = line.values()
        return pointb.value - pointa.value


class UnitLineVector(LeafConstruction):
    """
    Return the unit line vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        line_vec = LineVector.assem(prims)
        return line_vec/jnp.linalg.norm(line_vec)


class Length(LeafConstruction):
    """
    Return the length of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return jnp.sum(LineVector.assem((line,))**2)**(1/2)


class DirectedLength(LeafConstruction):
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
        return _AUX_DATA(1, namedtuple('Parameters', ('direction',)))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Line],
        direction: NDArray
    ):
        (line,) = prims
        return jnp.dot(LineVector.assem((line,)), direction)


class XLength(LeafConstruction):
    """
    Constrain the length of a line projected along the x direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([1, 0]))


class YLength(LeafConstruction):
    """
    Constrain the length of a line projected along the y direction

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return DirectedLength.assem(prims, np.array([0, 1]))


class Midpoint(LeafConstruction):
    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        line, = prims
        return 1/2*(line["Point0"].value + line["Point1"].value)

# Argument type: tuple[Line, Line]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Line, pr.Line),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class MidpointDirectedDistance(LeafConstruction):
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
        return _AUX_DATA(1, namedtuple("Parameters", ("direction",)))

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], direction: NDArray):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims
        return jnp.dot(Midpoint.assem((line1,)) - Midpoint.assem((line0,)), direction)


class MidpointXDistance(LeafConstruction):
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
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        return MidpointDirectedDistance.assem(prims, np.array([1, 0]))


class MidpointYDistance(LeafConstruction):
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
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        return MidpointDirectedDistance.assem(prims, np.array([0, 1]))


class Angle(LeafConstruction):
    """
    Return the angle between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        line0, line1 = prims
        dir0 = UnitLineVector.assem((line0,))
        dir1 = UnitLineVector.assem((line1,))
        return jnp.arccos(jnp.dot(dir0, dir1))

## Point and Line constraints

# Argument type: tuple[Point, Line]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Point, pr.Line),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }


class PointOnLineDistance(LeafConstruction):
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
        return _AUX_DATA(1, namedtuple('Parameters', ('reverse',)))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Point, pr.Line],
        reverse: bool
    ):
        """
        Return the projected distance error of a point along a line
        """
        point, line = prims
        if reverse:
            origin = line['Point1'].value
            unit_vec = -UnitLineVector.assem((line,))
        else:
            origin = line['Point0'].value
            unit_vec = UnitLineVector.assem((line,))

        return jnp.dot(point.value-origin, unit_vec)


class PointToLineDistance(LeafConstruction):
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
        return _AUX_DATA(1, namedtuple('Parameters', ('reverse',)))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Point, pr.Line],
        reverse: bool
    ):
        """
        Return the projected distance error of a point to a line
        """
        point, line = prims

        line_vec = UnitLineVector.assem((line,))

        if reverse:
            orth_vec = jnp.cross(line_vec, np.array([0, 0, 1]))[:2]
        else:
            orth_vec = jnp.cross(line_vec, np.array([0, 0, -1]))[:2]

        origin = line['Point0'].value

        return jnp.dot(point.value-origin, orth_vec)

## Quad constructions

# Argument type: tuple[Quadrilateral]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Quadrilateral,),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class AspectRatio(LeafConstruction):
    """
    Return the aspect ratio of a quadrilateral

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The quad
    """

    @classmethod
    def init_aux_data(cls):
        return _AUX_DATA(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral]):
        quad, = prims
        width = Length.assem((quad['Line0'],))
        height = Length.assem((quad['Line1'],))
        return  width/height

# Argument type: tuple[Quadrilateral, Quadrilateral]

def _AUX_DATA(
    value_size: int,
    params: tuple[any] = namedtuple('Parameters', ())
):
    return {
        'RES_ARG_TYPES': (pr.Quadrilateral, pr.Quadrilateral),
        'RES_PARAMS_TYPE': params,
        'RES_SIZE': value_size
    }

class OuterMargin(CompoundConstruction):
    """
    Return the outer margin between two quadrilaterals

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    """

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    @classmethod
    def init_children(cls, side: str="left"):
        child_keys = ("Margin",)
        if side == "left":
            child_constructions = (MidpointXDistance(),)
            child_prim_keys = (("arg1/Line1", "arg0/Line3"),)
        elif side == "right":
            child_constructions = (MidpointXDistance(),)
            child_prim_keys = (("arg0/Line1", "arg1/Line3"),)
        elif side == "bottom":
            child_constructions = (MidpointYDistance(),)
            child_prim_keys = (("arg1/Line2", "arg0/Line0"),)
        elif side == "top":
            child_constructions = (MidpointYDistance(),)
            child_prim_keys = (("arg0/Line2", "arg1/Line0"),)
        else:
            raise ValueError()

        def propogate_child_params(params):
            return [()]

        return child_keys, child_constructions, child_prim_keys, propogate_child_params

    @classmethod
    def init_aux_data(cls, side: str="left"):
        return _AUX_DATA(0, namedtuple('Parameters', ('side',)))


class InnerMargin(CompoundConstruction):
    """
    Return the inner margin between two quadrilaterals

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    """

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    @classmethod
    def init_children(cls, side: str="left"):
        child_keys = ["Margin"]
        if side == "left":
            child_constructions = [MidpointXDistance()]
            child_prim_keys = [("arg1/Line3", "arg0/Line3")]
        elif side == "right":
            child_constructions = [MidpointXDistance()]
            child_prim_keys = [("arg0/Line1", "arg1/Line1")]
        elif side == "bottom":
            child_constructions = [MidpointYDistance()]
            child_prim_keys = [("arg1/Line0", "arg0/Line0")]
        elif side == "top":
            child_constructions = [MidpointYDistance()]
            child_prim_keys = [("arg0/Line2", "arg1/Line2")]
        else:
            raise ValueError()

        def propogate_child_params(params):
            return [()]

        return child_keys, child_constructions, child_prim_keys, propogate_child_params

    @classmethod
    def init_aux_data(cls, side: str="left"):
        return _AUX_DATA(0, namedtuple('Parameters', ('side',)))

