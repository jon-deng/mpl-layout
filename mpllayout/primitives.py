"""
Geometry primitives
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp
import jax

from .containers import LabelledTuple


ArrayShape = typ.Tuple[int, ...]

## Generic primitive class/interface
# You can create specific primitive definitions by inheriting from these and
# defining appropriate class attributes


class Primitive:
    """
    A representation of a geometric primitive

    A `Primitive` can be parameterized by a parameter vector as well as
    other geometric primitives. For example, a point in 2D is parameterized by a
    vector representing (x, y) coordinates. Primitives can also contain implicit
    constraints to represent common use-cases. For example, an origin point may
    be explicitly constrained to have (0, 0) coordinates.

    To create a `Primitive` class, subclass `Primitive` and define the class
    attributes `_PARAM_SHAPE`, `_PRIM_TYPES`, `_PRIM_LABELS`

    Parameters
    ----------
    param: NDArray with shape (n,)
        A parameter vector for the primitive
    prims: PrimList
        A tuple of primitives parameterizing the primitive

    Attributes
    ----------
    param: NDArray[float] with shape (n,)
        A parameter vector for the primitive
    prims: LabelledTuple['Primitive']
        If non-empty, the primitive contains child geometric primitives in
        `self.prims`

    _PARAM_SHAPE: ArrayShape
        The shape of the parameter vector parameterizing the `Primitive`
    _PRIM_TYPES: typ.Union[
            typ.Tuple[typ.Type['Primitive'], ...],
            typ.Type['Primitive']
        ]
        The types of child primitives parameterizing the `Primitive`
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]]
        Optional labels for the child primitives
    """

    ## Specific primitive classes should define these to represent different primitives
    _PARAM_SHAPE: ArrayShape = (0,)
    # `_PRIM_TYPES` can either be a tuple of types, or a single type.
    # If it's a single type, then this implies a variable number of child primitives of that type
    # If it's a tuple of types, then this implies a set of child primitives of the corresponding type
    _PRIM_TYPES: typ.Union[
        typ.Tuple[typ.Type["Primitive"], ...], typ.Type["Primitive"]
    ] = ()
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]] = None

    def __init__(
        self,
        param: typ.Optional[NDArray] = None,
        prims: typ.Optional[typ.List["Primitive"]] = None,
    ):
        # Create default `param` if it's undefined
        if param is None:
            param = np.zeros(self._PARAM_SHAPE, dtype=float)
        elif not isinstance(param, (np.ndarray, jnp.ndarray)):
            param = np.array(param, dtype=float)

        self._param: NDArray = param

        # Create default `prims` if it's undefined
        if prims is None:
            if isinstance(self._PRIM_TYPES, tuple):
                prims = tuple(PrimType() for PrimType in self._PRIM_TYPES)
            else:
                prims = ()

        if self._PRIM_LABELS is None:
            keys = [f'{type(prim).__name__}{n}' for n, prim in enumerate(prims)]
        elif isinstance(self._PRIM_LABELS, str):
            keys = [f'{self._PRIM_LABELS}{n}' for n in range(len(prims))]
        elif isinstance(self._PRIM_LABELS, tuple):
            keys = self._PRIM_LABELS
        else:
            raise TypeError(f"{self._PRIM_LABELS}")

        self._prims = {key: prim for key, prim in zip(keys, prims)}

    @property
    def param(self):
        """
        Return the primitive's parameter vector
        """
        return self._param

    @property
    def prims(self):
        """
        Return the primitive's child primitives
        """
        return self._prims

    def __repr__(self):
        prim_tuple_repr = (
            "(" + str.join(", ", [prim.__repr__() for prim in self.prims.values()]) + ")"
        )
        return f"{type(self).__name__}({self.param}, {prim_tuple_repr})"

    def __str__(self):
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.prims)

    def __getitem__(self, key: str) -> "Primitive":
        return self.prims[key]


PrimList = typ.Tuple[Primitive, ...]
PrimTuple = typ.Tuple[Primitive, ...]


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

    def __init__(
        self, param: typ.Optional[NDArray] = None, prims: typ.Optional[PrimList] = None
    ):
        if not isinstance(prims, (tuple, list)):
            super().__init__(param, prims)
        else:
            # This allows you to input `prims` as a series of points rather than lines
            if all((isinstance(prim, Point) for prim in prims)):
                lines = self._points_to_lines(prims)
            else:
                lines = prims

            super().__init__(param, lines)

    @staticmethod
    def _points_to_lines(prims: PrimList):
        """
        Return a sequence of joined lines through a set of points
        """
        lines = [
            Line(np.array([]), [pointa, pointb])
            for pointa, pointb in zip(prims[:], prims[1:] + prims[:1])
        ]

        return lines


class Quadrilateral(Polygon):
    """
    A 4 sided closed polygon
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Line, Line, Line, Line)


def _make_flatten_unflatten(PrimitiveClass):

    def _flatten_primitive(prim):
        children = (prim.param, prim.prims)
        aux_data = None
        return (children, None)

    def _unflatten_primitive(aux_data, children):
        param, prims = children
        return PrimitiveClass(param, tuple(prims.values()))

    return _flatten_primitive, _unflatten_primitive

_PrimitiveClasses = [
    Quadrilateral, Point, Line, Polygon
]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = _make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass,
        _flatten_primitive,
        _unflatten_primitive
    )
