"""
Geometry primitives
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp

from .array import LabelledTuple

PrimList = typ.Tuple['Primitive', ...]
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
PrimTuple = typ.Tuple['Primitive', ...]

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
        typ.Tuple[typ.Type['Primitive'], ...],
        typ.Type['Primitive']
    ] = ()
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]] = None

    def __init__(
            self,
            param: typ.Optional[NDArray]=None,
            prims: typ.Optional[PrimList]=None
        ):
        # Create default `param` if it's undefined
        if param is None:
            param = np.zeros(self._PARAM_SHAPE, dtype=float)
        elif not isinstance(param, (np.ndarray, jnp.ndarray)):
            param = np.array(param, dtype=float)

        self._param: NDArray[float] = param

        # Create default `prims` if it's undefined
        if prims is None:
            if isinstance(self._PRIM_TYPES, tuple):
                prims = tuple(PrimType() for PrimType in self._PRIM_TYPES)
            else:
                # PrimType = self._PRIM_TYPES
                prims = ()

        if self._PRIM_LABELS is None:
            keys = None
        elif isinstance(self._PRIM_LABELS, tuple):
            keys = self._PRIM_LABELS
        else:
            keys = len(prims)*(self._PRIM_LABELS,)

        self._prims: LabelledTuple['Primitive'] = LabelledTuple(prims, keys)

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
            '('
            + str.join(', ', [prim.__repr__() for prim in self.prims])
            + ')'
        )
        return f'{type(self).__name__}({self.param}, {prim_tuple_repr})'

    def __str__(self):
        return self.__repr__()

    def __len__(self) -> int:
        return len(self.prims)

    def __getitem__(self, key: int) -> 'Primitive':
        return self.prims[key]


class PrimitiveArray(Primitive):
    """
    A representation of an array of geometric primitives

    Parameters
    ----------
    ... : see `Primitive`

    Attributes
    ----------
    ... : see `Primitive`
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        make_prim, child_prim_idxs = self.index_spec(key)
        return make_prim(tuple(self.prims[idx] for idx in child_prim_idxs))

    def index_spec(self, key) -> typ.Tuple[
            typ.Callable[[PrimTuple], Primitive],
            typ.Tuple[str, ...]
        ]:
        """
        Return a function and argument indices that form an indexed primitive

        Returns
        -------
        make_prim: typ.Callable[[PrimTuple], Primitive]
            A function that returns the indexed primitive from input primitives
        child_prim_idxs: typ.Tuple[str, ...]
            Indices of child primitives that are input to `make_prim` to get the
            indexed primitive
        """
        raise NotImplementedError


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
            self,
            param: typ.Optional[NDArray]=None,
            prims: typ.Optional[PrimList]=None
        ):
        # This allows you to input `prims` as a series of points rather than lines
        if not isinstance(self._PRIM_TYPES, tuple):
            prim_types = ()
        else:
            prim_types = self._PRIM_TYPES

        if prims is None:
            prims = tuple(Point() for _ in range(len(prim_types)))

        if all((isinstance(prim, Point) for prim in prims)):
            lines = [
                Line(np.array([]), [pointa, pointb])
                for pointa, pointb in zip(prims[:], prims[1:]+prims[:1])
            ]
        else:
            lines = prims

        super().__init__(param, lines)


class Quadrilateral(Polygon):
    """
    A 4 sided closed polygon
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Line, Line, Line, Line)
