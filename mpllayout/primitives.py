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

    A `Primitive` can be parameterized by a parameter vector as well as
    other geometric primitives. For example, a point in 2D is parameterized by a
    vector representing (x, y) coordinates.

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
    """

    def default_value(self):
        raise NotImplementedError()

    def default_prims(self):
        raise NotImplementedError()

    def init_topology(
        self, value: NDArray, prims: tp.List[Primitive]
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
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

        super().__init__(value, *self.init_topology(value, prims))


class ParameterizedPrimitive(Primitive):
    """
    A "parameterized" geometric primitive

    Parameterized primitives have variable sizes based on keyword arguments
    """

    def default_value(self, **kwargs):
        raise NotImplementedError()

    def default_prims(self, **kwargs):
        raise NotImplementedError()

    def init_topology(
        self, value: NDArray, prims: tp.List[Primitive], **kwargs
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
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

        super().__init__(value, *self.init_topology(value, prims))


## Primitive definitions

class Point(StaticPrimitive):
    """
    A point
    """

    def default_value(self):
        return np.array([0, 0])

    def default_prims(self):
        return ()

    def init_topology(self, value, prims):
        return (), ()


class Line(StaticPrimitive):
    """
    A straight line segment between two points
    """

    def default_value(self):
        return np.array([0, 0])

    def default_prims(self):
        return (Point([0, 0]), Point([0, 1]))

    def init_topology(self, value, prims: tp.Tuple[Point, Point]):
        return ("Point0", "Point1"), prims


class Polygon(ParameterizedPrimitive):
    """
    A polygon through a given set of points
    """

    def default_value(self, size=3):
        return np.array([])

    def default_prims(self, size=3):
        # Generate points around circle
        ii = np.arange(size)
        xs = np.cos(2*np.pi/size * ii)
        ys = np.sin(2*np.pi/size * ii)
        return [Point((x, y)) for x, y in zip(xs, ys)]

    def init_topology(
        self, value: NDArray, prims: tp.List[Point], size=3
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
    A 4 sided closed polygon
    """

    def __init__(
        self,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List[Point]] = None,
    ):
        super().__init__(value, children, size=4)


class Axes(StaticPrimitive):

    def default_value(self):
        return np.array([])

    def default_prims(self):
        return (Quadrilateral(),)

    def init_topology(
        self, value: NDArray, prims: tp.List[Primitive]
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        return ("Frame", ), prims


class AxesX(StaticPrimitive):

    def default_value(self):
        return np.array([])

    def default_prims(self):
        return (
            Quadrilateral(), Quadrilateral(), Point()
        )

    def init_topology(
        self, value: NDArray, prims: tp.List[Primitive]
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        child_keys = ("Frame", "XAxis", "XAxisLabel")
        child_prims = prims
        return child_keys, child_prims


class AxesXY(StaticPrimitive):

    def default_value(self):
        return np.array([])

    def default_prims(self):
        return (
            Quadrilateral(), Quadrilateral(), Point(), Quadrilateral(), Point()
        )

    def init_topology(
        self, value: NDArray, prims: tp.List[Primitive]
    ) -> tp.Tuple[tp.List[str], tp.List[ChildPrimitive]]:
        child_keys = (
            "Frame", "XAxis", "XAxisLabel", "YAxis", "YAxisLabel"
        )
        child_prims = prims
        return child_keys, child_prims


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [Primitive, Quadrilateral, Point, Line, Polygon]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = _make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )
