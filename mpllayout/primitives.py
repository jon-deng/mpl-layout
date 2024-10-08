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
    vector representing (x, y) coordinates. Primitives can also contain implicit
    constraints to represent common use-cases. For example, an origin point may
    be explicitly constrained to have (0, 0) coordinates.

    To create a `Primitive` class, subclass `Primitive` and define the class
    attributes `PARAM_SHAPE`, `CHILD_TYPES`, `CHILD_KEYS`

    Parameters
    ----------
    value: NDArray with shape (n,)
        A parameter vector for the primitive
    children: PrimList
        A tuple of primitives parameterizing the primitive

    Attributes
    ----------
    value: NDArray[float] with shape (n,)
        A parameter vector for the primitive
    children: tp.List[Primitive]
        If non-empty, the primitive contains child geometric primitives in
        `self.children`
    keys: tp.List[str]

    PARAM_SHAPE: ArrayShape
        The shape of the parameter vector parameterizing the `Primitive`
    CHILD_TYPES: tp.Tuple[tp.Type['Primitive'], ...]
        The types of child primitives parameterizing the `Primitive`
    CHILD_KEYS: tp.Tuple[str, ...]
        Keys for the child primitives
    """

    ## Specific primitive classes should define these to represent different primitives
    PARAM_SHAPE: ArrayShape = (0,)
    CHILD_TYPES: tp.Tuple[tp.Type[ChildPrimitive], ...]
    CHILD_KEYS: tp.Tuple[str, ...]

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List["Primitive"]] = None,
    ):
        # NOTE: `Primitive` classes specify keys through `Primitive.CHILD_KEYS`
        # This is unlike `Node`, so `keys` is basically ignored!

        # Create default `value` if unspecified
        if value is None:
            value = np.zeros(cls.PARAM_SHAPE, dtype=float)
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        elif isinstance(value, (np.ndarray, jax.numpy.ndarray)):
            value = value
        else:
            raise TypeError(f"Invalid type {type(value)} for `value`")

        # Create default `children` if unspecified
        if children is None:
            children = tuple(Child.from_std() for Child in cls.CHILD_TYPES)

        # Validate the number of child primitives
        if len(children) != len(cls.CHILD_TYPES):
            raise ValueError(
                f"Expected {num_child} child primitives, got {len(children)}"
            )

        # Validate child primitive types
        type_comparisons = (
            type(child) == ref_type
            for child, ref_type in zip(children, cls.CHILD_TYPES)
        )
        if not all(type_comparisons):
            raise TypeError(f"Expected child types {ref_types} got {child_types}")

        # Create keys from class primitive labels
        children_map = {key: prim for key, prim in zip(cls.CHILD_KEYS, children)}

        return cls(value, children_map)


PrimList = tp.List[Primitive]


## Actual primitive classes


class Point(Primitive):
    """
    A point
    """

    PARAM_SHAPE = (2,)
    CHILD_TYPES = ()
    CHILD_KEYS = ()


class Line(Primitive[Point]):
    """
    A straight line segment between two points
    """

    PARAM_SHAPE = (0,)
    CHILD_TYPES = (Point, Point)
    CHILD_KEYS = ("Point0", "Point1")


class Polygon(Primitive[Line]):
    """
    A polygon through a given set of points
    """

    PARAM_SHAPE = (0,)
    CHILD_TYPES: tp.Tuple[Line, ...]
    CHILD_KEYS: tp.Tuple[str, ...]

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List[Point]] = None,
    ):
        if children is None:
            children = []

        if not hasattr(cls, "CHILD_TYPES"):
            cls.CHILD_TYPES = len(children) * (Line,)
        if not hasattr(cls, "CHILD_KEYS"):
            cls.CHILD_KEYS = tuple(f"Line{n}" for n in range(len(children)))

        # Polygons contain lines as `CHILD_TYPES` but it's easier to
        # pass the points the lines pass through as children instead.
        # The `from_std` constructor therefore accepts points instead of lines
        # but passes the actual lines to the `Primitive.from_std`
        if all(isinstance(prim, Point) for prim in children):
            lines = cls._points_to_lines(children)
        else:
            raise TypeError()

        return super().from_std(value, lines)

    @staticmethod
    def _points_to_lines(children: tp.List[Point]) -> tp.List[Line]:
        """
        Return a sequence of joined lines through a set of points
        """
        lines = [
            Line.from_std(np.array([]), [pointa, pointb])
            for pointa, pointb in zip(children[:], children[1:] + children[:1])
        ]

        return lines


class Quadrilateral(Polygon):
    """
    A 4 sided closed polygon
    """

    PARAM_SHAPE = (0,)
    CHILD_TYPES = (Line, Line, Line, Line)
    CHILD_KEYS = ("Line0", "Line1", "Line2", "Line3")

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List[Point]] = None,
    ):
        if children is None:
            children = [
                Point.from_std([0, 0]),
                Point.from_std([1, 0]),
                Point.from_std([1, 1]),
                Point.from_std([0, 1]),
            ]

        return super().from_std(value, children)


class Axes(Primitive[Quadrilateral]):

    PARAM_SHAPE = (0,)
    CHILD_TYPES = (Quadrilateral,)
    CHILD_KEYS = ("Frame",)


class StandardAxes(Primitive[Quadrilateral | Point]):

    PARAM_SHAPE = (0,)
    CHILD_TYPES = (Quadrilateral, Quadrilateral, Quadrilateral, Point, Point)
    CHILD_KEYS = ("Frame", "XAxis", "YAxis", "XAxisLabel", "YAxisLabel")


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [Primitive, Quadrilateral, Point, Line, Polygon]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = _make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )
