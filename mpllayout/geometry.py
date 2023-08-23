"""
Geometric primitive and constraints
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp

from .array import LabelledTuple

Prims = typ.Tuple['Primitive', ...]
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
PrimTuple = typ.Tuple['Primitive', ...]

class PrimIdx:
    """
    An index to a child primitive

    Parameters
    ----------
    label: str
        The string identifier for the primitive.
        If `label` has no periods (e.g. '') this refers to the root primitive itself.
        If `label` is a period-prefixed string (e.g. '.Point0'), this refers to the named child
        primitive.
    sub_idx: int
        An integer representing the indexed primitive for `PrimitiveArray` types
    """

    def __init__(
            self,
            label: str,
            sub_idx: typ.Optional[int]=None
        ):

        self._label = label
        self._sub_idx = sub_idx

    @property
    def sub_idx(self):
        return self._sub_idx

    @property
    def label(self):
        return self._label

    def __repr__(self):
        return f'PrimIdx({self.label}, {self.sub_idx})'

    def __str__(self):
        return self.__repr__()

## Basic geometric primitives

class Primitive:
    """
    A basic geometric primitive

    Parameters
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive

    Attributes
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive
    prims: Tuple[Primitive, ...]
        If non-empty, the primitive contains child geometric primitives in `self.prims`
    constraints: Tuple[Constraint, ...]
        If non-empty, the primitive contains implicit geometric constraints in `self.constraints`
    constraint_graph: Tuple[Tuple[str, ...], ...]
        A graph representing which primitives a constraint applies to
    """

    _param: NDArray
    _prims: Prims

    ## Specific primitive classes should define these to represent different
    ## primitives
    _PARAM_SHAPE: ArrayShape = (0,)
    # `_PRIM_TYPES` can either be a tuple of types, or a single type.
    # If it's a single type, then this implies a variable number of child primitives of that type
    # If it's a tuple of types, then this implies a set of child primitives of the corresponding type
    _PRIM_TYPES: typ.Union[typ.Tuple[typ.Type['Primitive'], ...], typ.Type['Primitive']] = ()
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]] = None
    _CONSTRAINT_TYPES: typ.Tuple['Constraint', ...] = ()
    _CONSTRAINT_GRAPH: 'ConstraintGraph' = ()

    def __init__(
            self,
            param: typ.Optional[NDArray]=None,
            prims: typ.Optional[Prims]=None
        ):
        # Create default `param` and `prims` if they're undefined
        if param is None:
            param = np.zeros(self._PARAM_SHAPE, dtype=float)
        elif not isinstance(param, (np.ndarray, jnp.ndarray)):
            param = np.array(param, dtype=float)

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

        prims = LabelledTuple(prims, keys)

        # Create any internal constraints
        self._CONSTRAINTS = tuple(
            Constraint() for Constraint in self._CONSTRAINT_TYPES
        )

        # Check types and shapes are correct
        assert param.shape == self._PARAM_SHAPE
        prim_types = tuple(type(prim) for prim in prims)
        if isinstance(self._PRIM_TYPES, tuple):
            assert prim_types == self._PRIM_TYPES
        elif isinstance(self._PRIM_TYPES, type):
            assert prim_types == len(prim_types)*(self._PRIM_TYPES,)
        else:
            raise TypeError()

        self._param: NDArray = param
        self._prims = prims

    @property
    def param(self):
        """
        Return the primitive's parameter vector
        """
        return self._param

    @property
    def prims(self):
        return self._prims

    @property
    def constraints(self):
        return self._CONSTRAINTS

    @property
    def constraint_graph(self):
        return self._CONSTRAINT_GRAPH

    def __repr__(self):
        prim_tuple_repr = (
            '('
            + str.join(', ', [prim.__repr__() for prim in self.prims])
            + ')'
        )
        return f'{type(self).__name__}({self.param}, {prim_tuple_repr})'

class PrimitiveArray(Primitive):
    """
    A geometric primitive representing an array of primitives

    Parameters
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive

    Attributes
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive
    prims: Tuple[Primitive, ...]
        If non-empty, the primitive contains other geometric primitives in `self.prims`
    constraints: Tuple[Constraint, ...]
        If non-empty, the primitive contains implicit geometric constraints in `self.constraints`
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        make_prim, child_prim_idxs = self.index_spec(key)
        return make_prim(tuple(self.prims[idx] for idx in child_prim_idxs))

    def index_spec(self, key) -> typ.Tuple[
            typ.Callable[[PrimTuple], Primitive],
            typ.Tuple[int, ...]
        ]:
        """
        Return a specification for forming an indexed primitive from the array

        Returns
        -------
        make_prim:
            A function that returns a primitive from input primitives
        child_prim_idxs:
            Indices of child primitives that are input to `make_prim` to get the
            indexed primitive
        """
        raise NotImplementedError

class Constraint:
    """
    Constraint base class
    """

    primitive_types: typ.Tuple[typ.Type['Primitive'], ...]

    def __call__(self, prims: typ.Tuple['Primitive', ...]):
        # Check the input primitives are valid
        # assert len(prims) == len(self.primitive_types)
        # for prim, prim_type in zip(prims, self.primitive_types):
        #     assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))

    def assem_res(self, prims: typ.Tuple['Primitive', ...]) -> NDArray:
        raise NotImplementedError()


class Point(Primitive):

    _PARAM_SHAPE = (2,)
    _PRIM_TYPES = ()
    _PRIM_LABELS = ()
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

class PointToPointAbsDistance(Constraint):

    primitive_types = (Point, Point)

    def __init__(
            self, distance: float, direction: typ.Optional[NDArray]=None
        ):
        if direction is None:
            direction = np.array([1, 0])
        else:
            direction = np.array(direction)

        self._direction = direction
        self._distance = distance

    @property
    def distance(self):
        return self._distance

    @property
    def direction(self):
        return self._direction

    def assem_res(self, prims):
        return jnp.dot(prims[1].param - prims[0].param, self.direction) - self.distance

class PointLocation(Constraint):

    primitive_types = (Point, )

    def __init__(
            self,
            location: NDArray
        ):
        self._location = location

    def assem_res(self, prims):
        return prims[0].param - self._location

class CoincidentPoint(Constraint):

    primitive_types = (Point, Point)

    def assem_res(self, prims):
        return prims[0].param - prims[1].param


class LineSegment(Primitive):

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Point, Point)
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

class ClosedPolyline(PrimitiveArray):
    """
    A closed polyline passing through a given set of points
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = Point
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

    def __len__(self):
        return len(self.prims)

    def index_spec(self, key):
        def make_prim(prims):
            return LineSegment(prims=prims)

        if isinstance(key, int):
            idx1 = key % len(self)
            idx2 = (key+1) % len(self)
        else:
            raise TypeError("`key`, {key}, must be an integer")

        child_prim_idxs = (idx1, idx2)
        return make_prim, child_prim_idxs

class LineLength(Constraint):

    primitive_types = (LineSegment,)

    def __init__(
            self,
            length: float
        ):
        self._length = length

    def assem_res(self, prims):
        # This sets the length of a line
        vec = line_direction(prims[0])
        return jnp.linalg.norm(vec) - self._length

class Orthogonal(Constraint):
    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        line0, line1 = prims
        dir0 = line0.prims[1].param - line0.prims[0].param
        dir1 = line1.prims[1].param - line1.prims[0].param
        return jnp.dot(dir0, dir1)

class Vertical(Constraint):
    primitive_types = (LineSegment,)

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([1, 0]))

class Horizontal(Constraint):
    primitive_types = (LineSegment,)

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([0, 1]))

class Angle(Constraint):
    primitive_types = (LineSegment, LineSegment)

    def __init__(
            self,
            angle: NDArray
        ):
        self._angle = angle

    def assem_res(self, prims):
        line0, line1 = prims
        dir0 = line_direction(line0)
        dir1 = line_direction(line1)

        dir0 = dir0/jnp.linalg.norm(dir0)
        dir1 = dir1/jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self._angle

class Collinear(Constraint):
    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        line0, line1 = prims
        dir0 = line_direction(line0)
        dir1 = line_direction(line1)
        dir_inter = line1.prims[0].param - line0.prims[1].param
        return jnp.array([
            jnp.abs(jnp.dot(dir0, dir1)) - jnp.linalg.norm(dir0)*jnp.linalg.norm(dir1),
            jnp.abs(jnp.dot(dir0, dir_inter)) - jnp.linalg.norm(dir0)*jnp.linalg.norm(dir_inter)
        ])


class Box(ClosedPolyline):

    _PRIM_TYPES = (Point, Point, Point, Point)

    _CONSTRAINT_TYPES = (
        Horizontal,
        Vertical,
        Horizontal,
        Vertical
    )
    _CONSTRAINT_GRAPH = (
        (PrimIdx('', 0),),
        (PrimIdx('', 1),),
        (PrimIdx('', 2),),
        (PrimIdx('', 3),)
    )


def line_direction(line: LineSegment):
    return line.prims[1].param - line.prims[0].param

