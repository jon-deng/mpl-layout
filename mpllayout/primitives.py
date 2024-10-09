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
    CHILD_TYPES: tp.Union[
            tp.Tuple[tp.Type['Primitive'], ...],
            tp.Type['Primitive']
        ]
        The types of child primitives parameterizing the `Primitive`
    CHILD_KEYS: tp.Optional[tp.Union[tp.Tuple[str, ...], str]]
        Optional labels for the child primitives
    """

    ## Specific primitive classes should define these to represent different primitives
    PARAM_SHAPE: ArrayShape = (0,)
    # `CHILD_TYPES` can either be a tuple of types, or a single type.
    # If it's a single type, then this implies a variable number of child primitives of that type
    # If it's a tuple of types, then this implies a set of child primitives of the corresponding type
    CHILD_TYPES: tp.Union[
        tp.Tuple[tp.Type[ChildPrimitive], ...], tp.Type[ChildPrimitive]
    ] = ()
    CHILD_KEYS: tp.Optional[tp.Union[tp.Tuple[str, ...], str]] = None

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List["Primitive"]] = None,
        keys: tp.Optional[tp.List[str]] = None,
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
            if isinstance(cls.CHILD_TYPES, tuple):
                children = tuple(PrimType.from_std() for PrimType in cls.CHILD_TYPES)
            else:
                children = ()
        else:
            # Validate the number of child primitives
            if isinstance(children, (list, tuple)) and isinstance(
                cls.CHILD_TYPES, tuple
            ):
                num_child = len(cls.CHILD_TYPES)
                if len(children) != num_child:
                    raise ValueError(
                        f"Expected {num_child} child primitives, got {len(children)}"
                    )

            # Validate child primitive types
            child_types = tuple(type(prim) for prim in children)
            if isinstance(cls.CHILD_TYPES, Primitive):
                ref_types = cls.CHILD_TYPES
            else:
                ref_types = cls.CHILD_TYPES

            type_comparisons = (
                child_type == ref_type
                for child_type, ref_type in zip(child_types, ref_types)
            )
            if not all(type_comparisons):
                raise TypeError(f"Expected child types {ref_types} got {child_types}")

        # Create keys from class primitive labels
        if cls.CHILD_KEYS is None:
            keys = [f"{type(prim).__name__}{n}" for n, prim in enumerate(children)]
        elif isinstance(cls.CHILD_KEYS, str):
            keys = [f"{cls.CHILD_KEYS}{n}" for n in range(len(children))]
        elif isinstance(cls.CHILD_KEYS, tuple):
            keys = cls.CHILD_KEYS
        else:
            raise TypeError(f"{cls.CHILD_KEYS}")

        return cls(value, {key: prim for key, prim in zip(keys, children)})


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

    CHILD_TYPES = (Point, Point)
    PARAM_SHAPE = (0,)


class Polygon(Primitive[Line]):
    """
    A polygon through a given set of points
    """

    PARAM_SHAPE = (0,)
    CHILD_TYPES = Line

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List["Primitive"]] = None,
        keys: tp.Optional[tp.List[str]] = None,
    ):
        if not isinstance(children, (tuple, list)):
            return super().from_std(value, children)
        else:
            # This allows you to input `children` as a series of points rather than lines
            if all((isinstance(prim, Point) for prim in children)):
                lines = cls._points_to_lines(children)
            else:
                lines = children

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
