"""
Geometric primitive definitions
"""

from typing import Optional, TypeVar
from numpy.typing import NDArray

import numpy as np
import jax

# from .containers import Node, _make_flatten_unflatten, iter_flat, unflatten, FlatNodeStructure
import mpllayout.containers as cn


## Generic primitive class/interface
# You can create specific primitive definitions by inheriting from these and
# defining appropriate class attributes

ChildPrimitive = TypeVar("ChildPrimitive", bound="Primitive")


class Primitive(cn.Node[NDArray[np.float64], ChildPrimitive]):
    """
    The base geometric primitive class

    A `Primitive` is represented by a parameter vector and child primitives.
    For example in 2D:
    - a point has a size 2 parameter vector representing (x, y) coordinates and
    no child primitives,
    - a straight line segment has an empty parameter vector with two point
    child primitives representing the start point and end point.

    This class shouldn't be used to create geometric primitives directly.
    Subclasses that represent specific geometric primitives should
    be defined instead (for example, see `Point` or `Line` below).
    Subclasses `StaticPrimitive` and `DynamicPrimitive` are intermediate
    sub-classes that can be used to define these subclasses.

    Parameters
    ----------
    value: NDArray with shape (n,)
        A parameter vector for the primitive
    child_keys: list[str]
        Child primitive keys
    child_prims: list[ChildPrimitive]
        Child primitives
    """

    def __init__(
        self,
        value: NDArray,
        child_keys: list[str],
        child_prims: list[ChildPrimitive],
    ):
        children = {key: prim for key, prim in zip(child_keys, child_prims)}
        super().__init__(value, children)


class PrimitiveNode(cn.Node[NDArray[np.float64], Primitive]):
    """
    A container to store an arbitrary number of child primitives

    You can use the `cn.Node` methods to add child primitives to this container.
    """
    # TODO: Define `Primitive` methods for this?
    # TODO: Make classes differ between immutable/mutable Nodes?
    # NOTE: `PrimitiveNode` has a mutable number of child primitives while
    # other geometric primitives are immutable (points, lines, etc.)
    pass


# TODO: Add type checking for `StaticPrimitive` and `ParameterizedPrimitive`
# Both the classes have requirements on the types of `value` and `prims` but
# these aren't validated
class StaticPrimitive(Primitive):
    """
    A "static" geometric primitive

    Static primitives have fixed (see `Parameters`):
    - parameter vector shape
    - and child primitive types

    To define a static primitive, create a subclass and define
    `default_value`, `default_prims` and `init_children`.

    Parameters
    ----------
    value: NDArray, optional
        A parameter vector

        The parameter vector should match a known shape.

        If not supplied, `default_value` will be used to generate a default.
    prims: list[Primitive], optional
        Parameterizing primitives

        The parameterizing primitives must match the known primitive types.

        Note that in many cases parameterizing primitives are child primitives
        ; however, this depends on `init_children`.
    """

    def default_value(self) -> NDArray:
        """
        Return a default parameter vector
        """
        raise NotImplementedError()

    def default_prims(self) -> list[Primitive]:
        """
        Return default parameterizing primitives
        """
        raise NotImplementedError()

    def init_children(
        self, prims: list[Primitive]
    ) -> tuple[list[str], list[ChildPrimitive]]:
        """
        Return child primitives from parameterizing primitives

        Parameters
        ----------
        prims: list[Primitive]
            Parameterizing primitives

        Returns
        -------
        List[str]
            Child primitive keys
        List[ChildPrimitive]
            Child primitives
        """
        raise NotImplementedError()

    def __init__(
        self,
        value: Optional[NDArray] = None,
        prims: Optional[list[Primitive]] = None,
    ):
        if value is None:
            value = self.default_value()
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        elif isinstance(value, (np.ndarray, jax.numpy.ndarray)):
            value = value
        else:
            raise TypeError()

        if prims is None:
            prims = self.default_prims()

        super().__init__(value, *self.init_children(prims))


class ParameterizedPrimitive(Primitive):
    """
    A "parameterized" geometric primitive

    Parameterized primitives have variable (see `Parameters`):
    - parameter vector shape
    - and child primitive types
    based on additional keyword arguments.

    To define a parameterized primitive, create a subclass and define
    `default_value`, `default_prims` and `init_children`.

    Parameters
    ----------
    value: NDArray, optional
        A parameter vector

        The parameter vector should match a known shape.

        If not supplied, `default_value` will be used to generate a default.
    prims: list[Primitive], optional
        Parameterizing primitives

        The parameterizing primitives should match known primitive types.

        Note that in many cases parameterizing primitives are child primitives
        ; however, this depends on `init_children`.
    **kwargs
        Arbitrary additional keyword arguments

        Subclasses should define what these arguments are and how they affect
        the primitive.
        For example, a `size` keyword argument could specify the length of
        `prims` (see `Polygon`).
    """

    def default_value(self, **kwargs) -> NDArray:
        """
        Return a default parameter vector
        """
        raise NotImplementedError()

    def default_prims(self, **kwargs) -> list[Primitive]:
        """
        Return default parameterizing primitives
        """
        raise NotImplementedError()

    def init_children(
        self, prims: list[Primitive], **kwargs
    ) -> tuple[list[str], list[ChildPrimitive]]:
        """
        Return child primitives from parameterizing primitives

        Parameters
        ----------
        prims: list[Primitive]
            Parameterizing primitives

        Returns
        -------
        List[str]
            Child primitive keys
        List[ChildPrimitive]
            Child primitives
        """
        raise NotImplementedError()

    def __init__(
        self,
        value: Optional[NDArray] = None,
        prims: Optional[list[Primitive]] = None,
        **kwargs
    ):
        if value is None:
            value = self.default_value(**kwargs)
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        else:
            raise TypeError()

        if prims is None:
            prims = self.default_prims(**kwargs)

        super().__init__(value, *self.init_children(prims, **kwargs))


## Primitive definitions

class Point(StaticPrimitive):
    """
    A point

    Child primitives are:
    - no child primitives

    Parameters
    ----------
    value: NDArray (2,), optional
        The point coordinates
    prims: tuple[]
        An empty set of primitives
    """

    def default_value(self):
        return np.array([0, 0])

    def default_prims(self):
        return ()

    def init_children(self, prims):
        return (), ()


class Line(StaticPrimitive):
    """
    A straight line segment between two points

    Child primitives are:
    - `line['Point0']` : start point
    - `line['Point1']` : end point

    Parameters
    ----------
    value: NDArray (), optional
        An empty array
    prims: Tuple[Point, Point]
        The start and end point
    """

    def default_value(self):
        return np.array([0, 0])

    def default_prims(self):
        return (Point([0, 0]), Point([0, 1]))

    def init_children(self, prims: tuple[Point, Point]):
        return ("Point0", "Point1"), prims


class Polygon(ParameterizedPrimitive):
    """
    A polygon through a given set of points

    Child primitives are:
    - `Polygon[f'Line{n}']` : the n'th line in the polygon

    The end point of a line joins the start point of the next line to form a
    closed loop.

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: List[Point]
        A list of vertices the polygon passes through

        The final point in `prims` will automatically be connected to the first
        point in `prims`.
        The length of `prims` should match `size`.
    size: int
        The number of points in the polygon
    """

    def default_value(self, size: int=3):
        return np.array([])

    def default_prims(self, size: int=3):
        # Generate points around circle
        ii = np.arange(size)
        xs = np.cos(2*np.pi/size * ii)
        ys = np.sin(2*np.pi/size * ii)
        return [Point((x, y)) for x, y in zip(xs, ys)]

    def init_children(
        self, prims: list[Point], size: int=3
    ):
        points = prims
        child_prims = [
            Line(np.array([]), (pointa, pointb))
            for pointa, pointb in zip(points[:], points[1:] + points[:1])
        ]
        child_keys = [f"Line{n}" for n, _ in enumerate(child_prims)]
        return child_keys, child_prims


class Quadrilateral(Polygon):
    """
    A quadrilateral (4 sided polygon)

    Child primitives are:
    - `quad['Line0']` : the first line in the quadrilateral
    - ...
    - `quad['Line3']` : the last line in the quadrilateral

    For modelling rectangles in matplotlib (axes, bbox, etc.) the lines are
    treated as the bottom, right, top, and left of a box in a clockwise fasion.
    Specifically, the lines correspond to:
    - 'Line0' : bottom
    - 'Line1' : right
    - 'Line2' : top
    - 'Line3' : left

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: List[Point]
        A list of 4 vertices the quadrilateral passes through
    """

    def default_value(self, size: int=4):
        return np.array([])

    def default_prims(self, size: int=4):
        # Generate a unit square
        xs = [0, 1, 1, 0]
        ys = [0, 0, 1, 1]
        return [Point((x, y)) for x, y in zip(xs, ys)]

    def __init__(
        self,
        value: Optional[NDArray] = None,
        children: Optional[list[Point]] = None,
    ):
        super().__init__(value, children, size=4)


AxesChildPrims = tuple[Quadrilateral, Quadrilateral, Point, Quadrilateral, Point]
class Axes(ParameterizedPrimitive):
    """
    A collection of quadrilaterals and points representing an axes

    Child primitives are:
    - `quad['Frame']` : A `Quadrilateral` representing the plotting area
    - `quad['XAxis']` : A `Quadrilateral` representing the x-axis
    - `quad['XAxisLabel']` : A `Point` representing the x-axis label anchor
    - `quad['YAxis']` : A `Quadrilateral` representing the y-axis
    - `quad['YAxisLabel']` : A `Point` representing the y-axis label anchor

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: tuple[Quadrilateral, Quadrilateral, Point, Quadrilateral, Point]
        A tuple of quadrilateral and points

        The the number of quadrilaterals and points to supply depends on
        where an x/y axis is included.
    xaxis, yaxis: bool
        Whether to include an x/y axis and corresponding label

        If false for a given axis, the corresponding child primitives will not
        be present.
    """

    def default_value(self, xaxis: bool=False, yaxis: bool=False):
        return np.array([])

    def default_prims(self, xaxis: bool=False, yaxis: bool=False):
        if xaxis:
            xaxis_prims = (Quadrilateral(), Point())
        else:
            xaxis_prims = ()

        if yaxis:
            yaxis_prims = (Quadrilateral(), Point())
        else:
            yaxis_prims = ()
        return (Quadrilateral(),) + xaxis_prims + yaxis_prims

    def init_children(
        self,
        prims: AxesChildPrims,
        xaxis: bool=False,
        yaxis: bool=False
    ) -> tuple[list[str], list[ChildPrimitive]]:

        if xaxis:
            xaxis_keys = ("XAxis", "XAxisLabel")
        else:
            xaxis_keys = ()

        if yaxis:
            yaxis_keys = ("YAxis", "YAxisLabel")
        else:
            yaxis_keys = ()
        return ("Frame",) + xaxis_keys + yaxis_keys, prims

    def __init__(
            self,
            value: Optional[NDArray]=None,
            prims: Optional[AxesChildPrims]=None,
            xaxis: bool=False,
            yaxis: bool=False
        ):
        super().__init__(value, prims, xaxis=xaxis, yaxis=yaxis)


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [
    Primitive,
    PrimitiveNode,
    Point,
    Line,
    Polygon,
    Quadrilateral,
    Axes,
]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = cn._make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )


## Primitive value vector methods
# These are used to get the primitive parameter vector from a primitive
# and to update primitives with new parameters

def filter_unique_values_from_prim(
    root_prim: Primitive,
) -> tuple[dict[str, int], list[Primitive]]:
    """
    Return unique primitives from a root primitive and indicate their indices

    Note that primitives in a primitive node are not necessarily unique
    ; for example `Point`s are shared between lines in a polygon.

    When solving a set of geometric constraints, the geometric constraint
    residual should be linked to a function of unique primitives only.

    Returns
    -------
    prim_to_idx: dict[str, int]
        A mapping from each primitive key to its unique primitive index
    prims: list[Primitive]
        A list of unique primitives
    """
    value_id_to_idx = {}
    values = []
    prim_to_idx = {}

    for key, prim in cn.iter_flat("", root_prim):
        value_id = id(prim.value)

        if value_id not in value_id_to_idx:
            values.append(prim.value)
            value_idx = len(values) - 1
            value_id_to_idx[value_id] = value_idx
        else:
            value_idx = value_id_to_idx[value_id]

        prim_to_idx[key] = value_idx

    return prim_to_idx, values

def build_prim_from_unique_values(
    flat_prim: list[cn.FlatNodeStructure], prim_to_idx: dict[str, int], values: list[NDArray]
) -> Primitive:
    """
    Return a new primitive with values updated from unique values

    Parameters
    ----------
    flat_prim: list[FlatNodeStructure]
        The flat primitive tree (see `flatten`)
    prim_to_idx: dict[str, int]
        A mapping from each primitive key to a unique primitive value in `values`
    values: list[NDArray]
        A list of primitive values for unique primitives in `root_prim`

    Returns
    -------
    Primitive
        The new primitive with updated values
    """
    prim_keys = (flat_struct[0] for flat_struct in flat_prim)
    new_prim_values = (values[prim_to_idx[key]] for key in prim_keys)

    new_prim_structs = [
        (prim_key, PrimType, new_value, child_keys)
        for (prim_key, PrimType, _old_value, child_keys), new_value
        in zip(flat_prim, new_prim_values)
    ]
    return cn.unflatten(new_prim_structs)[0]
