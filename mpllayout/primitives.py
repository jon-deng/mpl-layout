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


class Primitive(Node[NDArray]):
    """
    A geometric primitive

    A `Primitive` can be parameterized by a parameter vector as well as
    other geometric primitives. For example, a point in 2D is parameterized by a
    vector representing (x, y) coordinates. Primitives can also contain implicit
    constraints to represent common use-cases. For example, an origin point may
    be explicitly constrained to have (0, 0) coordinates.

    To create a `Primitive` class, subclass `Primitive` and define the class
    attributes `_PARAM_SHAPE`, `_PRIM_TYPES`, `_PRIM_LABELS`

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

    _PARAM_SHAPE: ArrayShape
        The shape of the parameter vector parameterizing the `Primitive`
    _PRIM_TYPES: tp.Union[
            tp.Tuple[tp.Type['Primitive'], ...],
            tp.Type['Primitive']
        ]
        The types of child primitives parameterizing the `Primitive`
    _PRIM_LABELS: tp.Optional[tp.Union[tp.Tuple[str, ...], str]]
        Optional labels for the child primitives
    """

    ## Specific primitive classes should define these to represent different primitives
    _PARAM_SHAPE: ArrayShape = (0,)
    # `_PRIM_TYPES` can either be a tuple of types, or a single type.
    # If it's a single type, then this implies a variable number of child primitives of that type
    # If it's a tuple of types, then this implies a set of child primitives of the corresponding type
    _PRIM_TYPES: tp.Union[
        tp.Tuple[tp.Type["Primitive"], ...], tp.Type["Primitive"]
    ] = ()
    _PRIM_LABELS: tp.Optional[tp.Union[tp.Tuple[str, ...], str]] = None

    # def __init__(
    #     self,
    #     value: tp.Optional[NDArray] = None,
    #     children: tp.Optional[tp.List["Primitive"]] = None,
    #     keys: tp.Optional[tp.List[str]] = None,
    # ):


    #     super().__init__(value, children, keys)

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List["Primitive"]] = None,
        keys: tp.Optional[tp.List[str]] = None
    ):
        # NOTE: `Primitive` classes specify keys through `Primitive._PRIM_LABELS`
        # This is unlike `Node`, so `keys` is basically ignored!

        # Create default `value` if unspecified
        if value is None:
            value = np.zeros(cls._PARAM_SHAPE, dtype=float)
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        elif isinstance(value, (np.ndarray, jax.numpy.ndarray)):
            value = value
        else:
            raise TypeError(f"Invalid type {type(value)} for `value`")

        # Create default `children` if unspecified
        if children is None:
            if isinstance(cls._PRIM_TYPES, tuple):
                children = tuple(PrimType.from_std() for PrimType in cls._PRIM_TYPES)
            else:
                children = ()
        else:
            # Validate the number of child primitives
            if isinstance(children, (list, tuple)) and isinstance(
                cls._PRIM_TYPES, tuple
            ):
                num_child = len(cls._PRIM_TYPES)
                if len(children) != num_child:
                    raise ValueError(
                        f"Expected {num_child} child primitives, got {len(children)}"
                    )

            # Validate child primitive types
            child_types = tuple(type(prim) for prim in children)
            if isinstance(cls._PRIM_TYPES, Primitive):
                ref_types = cls._PRIM_TYPES
            else:
                ref_types = cls._PRIM_TYPES

            type_comparisons = (
                child_type == ref_type
                for child_type, ref_type in zip(child_types, ref_types)
            )
            if not all(type_comparisons):
                raise TypeError(f"Expected child types {ref_types} got {child_types}")

        # Create keys from class primitive labels
        if cls._PRIM_LABELS is None:
            keys = [f"{type(prim).__name__}{n}" for n, prim in enumerate(children)]
        elif isinstance(cls._PRIM_LABELS, str):
            keys = [f"{cls._PRIM_LABELS}{n}" for n in range(len(children))]
        elif isinstance(cls._PRIM_LABELS, tuple):
            keys = cls._PRIM_LABELS
        else:
            raise TypeError(f"{cls._PRIM_LABELS}")

        return cls(value, {key: prim for key, prim in zip(keys, children)})


PrimList = tp.List[Primitive]


## Actual primitive classes


class Point(Primitive):
    """
    A point
    """

    _PARAM_SHAPE = (2,)
    _PRIM_TYPES = ()
    _PRIM_LABELS = ()


class Line(Primitive):
    """
    A straight line segment between two points
    """

    _PRIM_TYPES = (Point, Point)
    _PARAM_SHAPE = (0,)


class Polygon(Primitive):
    """
    A polygon through a given set of points
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = Line

    @classmethod
    def from_std(
        cls,
        value: tp.Optional[NDArray] = None,
        children: tp.Optional[tp.List["Primitive"]] = None,
        keys: tp.Optional[tp.List[str]] = None
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

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Line, Line, Line, Line)


class Axes(Primitive):

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Quadrilateral, )
    _PRIM_LABELS = ("Frame",)

class StandardAxes(Primitive):

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Quadrilateral, Quadrilateral, Quadrilateral, Point, Point)
    _PRIM_LABELS = ("Frame", "XAxis", "YAxis", "XAxisLabel", "YAxisLabel")


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [Primitive, Quadrilateral, Point, Line, Polygon]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = _make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )
