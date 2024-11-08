"""
Geometry primitives
"""

import typing as tp
from numpy.typing import NDArray

import numpy as np
import jax

from .containers import Node, _make_flatten_unflatten


ArrayShape = tp.Tuple[int, ...]

## Generic primitive class/interface
# You can create specific primitive definitions by inheriting from these and
# defining appropriate class attributes

ChildPrimitive = tp.TypeVar("ChildPrimitive", bound="Primitive")


class Primitive(Node[NDArray[np.float64], ChildPrimitive]):
    """
    A geometric primitive

    A `Primitive` is represented by a parameter vector and child primitives.
    For example, a point in 2D is parameterized by a vector representing (x, y)
    coordinates.

    Subclasses "StaticPrimitive" and "DynamicPrimitive" provide more
    specific ways to create primitives.

    Parameters
    ----------
    value: NDArray with shape (n,)
        A parameter vector for the primitive
    child_keys: tp.List[str]
        Child primitive keys
    child_prims: tp.List[ChildPrimitive]
        Child primitives representing the topology
    """

    def __init__(
        self,
        value: NDArray,
        child_keys: tp.List[str],
        child_prims: tp.List[ChildPrimitive],
    ):
        children = {key: prim for key, prim in zip(child_keys, child_prims)}
        super().__init__(value, children)


class PrimitiveNode(Node[NDArray[np.float64], Primitive]):
    pass


PrimList = tp.List[Primitive]


class StaticPrimitive(Primitive):
    """
    A "static" geometric primitive

    Static primitives have predefined sizes and the signature below

    To define a static primitive, create a subclass and define the functions
    `default_value`, `default_prims` and `init_children`.

    Parameters
    ----------
    value: NDArray with shape (n,)
        A parameter vector for the primitive
    prims: tp.List[Primitive]
        Primitives used to parameterize the primitive

        In many cases, `prims` consists of the child primitives; however,
        this depends on the implementation of `init_children`.
    """

    def default_value(self) -> NDArray:
        """
        Return a default parameter vector
        """
        raise NotImplementedError()

    def default_prims(self) -> tp.List[Primitive]:
        """
        Return default parameterizing primitives
        """
        raise NotImplementedError()

    def init_children(
        self, prims: tp.List[Primitive]
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        """
        Return child primitives from parameterizing primitives

        In many cases the parameterizing primitives are child primitives.
        An exception is for `Polygon`.

        Parameters
        ----------
        prims: tp.List[Primitive]
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
        value: tp.Optional[NDArray] = None,
        prims: tp.Optional[tp.List[Primitive]] = None,
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

    Parameterized primitives have variable sizes based on keyword arguments
    To define a parameterized primitive, create a subclass and define the
    functions `default_value`, `default_prims` and `init_children`.

    Parameters
    ----------
    value: NDArray with shape (n,)
        A parameter vector for the primitive
    prims: tp.List[Primitive]
        Primitives used to parameterize the primitive

        In many cases, `prims` consists of the child primitives; however,
        this depends on the implementation of `init_children`.
    **kwargs
        Any additional arguments to parameterize the primitive
    """

    def default_value(self, **kwargs) -> NDArray:
        """
        Return a default parameter vector
        """
        raise NotImplementedError()

    def default_prims(self, **kwargs) -> tp.List[Primitive]:
        """
        Return default parameterizing primitives
        """
        raise NotImplementedError()

    def init_children(
        self, prims: tp.List[Primitive], **kwargs
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        """
        Return child primitives from parameterizing primitives

        In many cases the parameterizing primitives are child primitives.
        An exception is for `Polygon`.

        Parameters
        ----------
        prims: tp.List[Primitive]
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
        value: tp.Optional[NDArray] = None,
        prims: tp.Optional[tp.List[Primitive]] = None,
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

    A point has no child primitives.

    Parameters
    ----------
    value: NDArray (2,)
        The point coordinates
    prims: Tuple[]
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
    value: NDArray ()
        An empty array
    prims: Tuple[Point, Point]
        The start and end point
    """

    def default_value(self):
        return np.array([0, 0])

    def default_prims(self):
        return (Point([0, 0]), Point([0, 1]))

    def init_children(self, prims: tp.Tuple[Point, Point]):
        return ("Point0", "Point1"), prims


class Polygon(ParameterizedPrimitive):
    """
    A polygon through a given set of points

    Child primitives are:
    - `Polygon[f'Line{n}']` : the n'th line in the polygon

    Lines are directed in a clockwise fashion around a loop.

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: List[Point]
        A list of vertices the polygon passes through

        The final point in `prims` will automatically be connected to the first
        point in `prims`.
    size: int
        The number of points in the polygon
    """

    def default_value(self, size=3):
        return np.array([])

    def default_prims(self, size=3):
        # Generate points around circle
        ii = np.arange(size)
        xs = np.cos(2*np.pi/size * ii)
        ys = np.sin(2*np.pi/size * ii)
        return [Point((x, y)) for x, y in zip(xs, ys)]

    def init_children(
        self, prims: tp.List[Point], size=3
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        points = prims
        child_prims = [
            Line(np.array([]), [pointa, pointb])
            for pointa, pointb in zip(points[:], points[1:] + points[:1])
        ]
        child_keys = [f"Line{n}" for n, _ in enumerate(child_prims)]
        return child_keys, child_prims


class Quadrilateral(Polygon):
    """
    A quadrilateral (4 sided polygon)

    Child primitives are:
    - `quad['Line0']` : the first line in the quad
    - ...
    - `quad['Line3']` : the last line in the quad

    For modelling rectangle in matplotlib (axes, bbox, etc.) the lines are
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

    def __init__(
        self,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List[Point]] = None,
    ):
        super().__init__(value, children, size=4)


class Axes(ParameterizedPrimitive):
    """
    A collection of `Quadrilateral`s and `Point`s representing an axes

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
    prims: List[Point]
        A list of vertices the polygon passes through

        The final point in `prims` will automatically be connected to the first
        point in `prims`.
    xaxis, yaxis: bool
        Whether to include an x/y axis and corresponding label

        If false for a given axis, the corresponding child primitives will not
        be present.
    """

    def default_value(self, xaxis=False, yaxis=False):
        return np.array([])

    def default_prims(self, xaxis=False, yaxis=False):
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
        self, prims: tp.List[Primitive], xaxis=False, yaxis=False
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:

        if xaxis:
            xaxis_keys = ("XAxis", "XAxisLabel")
        else:
            xaxis_keys = ()

        if yaxis:
            yaxis_keys = ("YAxis", "YAxisLabel")
        else:
            yaxis_keys = ()
        return ("Frame",) + xaxis_keys + yaxis_keys, prims

    def __init__(self, value=None, prims=None, xaxis=False, yaxis=False):
        super().__init__(value, prims, xaxis=xaxis, yaxis=yaxis)


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [Primitive, Quadrilateral, Point, Line, Polygon]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = _make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )
