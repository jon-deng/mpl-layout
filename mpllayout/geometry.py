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

class PrimitiveIndex:
    """
    An index to a collection of `Primitive`s

    A `PrimIdx` represents an index to a primitive within a collection where
    primitives are labelled by unique string labels.
    There are two main use cases:
        an index to a specific primitive within a collection
        or an index to a child primitive from a parent primitive.

    In either case the first argument represents the label of the desired
    primitive while the second argument is an integer index if the desired
    primitive is a `PrimitiveArray` type.
    Periods in the label denote a child primitive.
    For example, `PrimitiveIndex('MyBox.Point0')`, denotes the first point
    , `'Point0'`, of the primitive called `'MyBox'`.
    As another example, `PrimitiveIndex('MyBox', 0)`, denotes the first line
    segment of the primitive called `'MyBox'`, in the case the primitive is a
    `Polyline`.

    When indexing from a collection of primitives, the string label has the
    form:
    `'parent_prim_label.child_prim_label.etc'`.
    When indexing a child primitive, the string label has the form:
    `'.child_prim_label.etc'`.

    Parameters
    ----------
    label: str
        The string identifier for the primitive.
        When indexing from a collection of primitives, the string label has the
        form:
        `'parent_prim_label.child_prim_label.etc'`.
        When indexing a child primitive, the string label has the form:
        `'.child_prim_label.etc'`.
    sub_idx: int
        An integer representing an indexed primitive when `label` points to a
        `PrimitiveArray` type primitive.
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
    A representation of a geometric primitive

    Primitive can be parameterized by a parameter vector as well as
    other geometric primitives.
    For example, a point in 2D is parameterized by a vector representing x and y coordinates.
    Primitives can also contain implicit constraints to represent common use-cases.
    For example, an origin point may be explicitly constraint to have (0, 0) coordinates.

    Parameters
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive
    prims: Tuple[Primitive, ...]
        A tuple of primitives parameterizing the primitive

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

    ## Specific primitive classes should define these to represent different primitives
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
        """
        Return the primitive's child primitives
        """
        return self._prims

    @property
    def constraints(self):
        """
        Return the primitive's implicit constraints
        """
        return self._CONSTRAINTS

    @property
    def constraint_graph(self):
        """
        Return the primitive's implicit constraint graph
        """
        return self._CONSTRAINT_GRAPH

    def __repr__(self):
        prim_tuple_repr = (
            '('
            + str.join(', ', [prim.__repr__() for prim in self.prims])
            + ')'
        )
        return f'{type(self).__name__}({self.param}, {prim_tuple_repr})'

    def __str__(self):
        return self.__repr__()

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
            typ.Tuple[str, ...]
        ]:
        """
        Return a tuple to facilitate making an indexed primitive from the array

        Returns
        -------
        make_prim: typ.Callable[[PrimTuple], Primitive]
            A function that returns the indexed primitive from input primitives
        child_prim_idxs: typ.Tuple[str, ...]
            Indices of child primitives that are input to `make_prim` to get the
            indexed primitive
        """
        raise NotImplementedError

class Constraint:
    """
    A representation of a constraint on primitives

    A constraint represents a condition on the parameters of geometric
    primitive(s).
    The condition is specified through a residual function `assem_res`,
    specified with the `jax` library.
    Usage of `jax` in specifying the constraints allows automatic
    differentiation of constraint conditions which is used for solving them.

    Parameters
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive
    prims: Tuple[Primitive, ...]
        A tuple of primitives parameterizing the primitive

    Attributes
    ----------
    primitive_types: Tuple[typ.Type[Primitive], ...]
        Specifies the types of primitives that the constraint applies to.
        For example, if `primitive_types = (geo.Point, geo.LineSegment)`, then
        the constraint applies to a point and a line segment.
    """

    primitive_types: typ.Tuple[typ.Type['Primitive'], ...]

    def __call__(self, prims: typ.Tuple['Primitive', ...]):
        # Check the input primitives are valid
        # assert len(prims) == len(self.primitive_types)
        # for prim, prim_type in zip(prims, self.primitive_types):
        #     assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))

    def assem_res(self, prims: typ.Tuple['Primitive', ...]) -> NDArray:
        """
        Return a residual vector representing the constraint satisfaction

        Parameters
        ----------
        prims: Tuple[Primitive, ...]
            A tuple of primitives the constraint applies to

        Returns
        -------
        NDArray
            The residual representing whether the constraint is satisfied.
            The constraint is satisfied when the residual is 0.
        """
        raise NotImplementedError()


class Point(Primitive):
    """
    A point
    """

    _PARAM_SHAPE = (2,)
    _PRIM_TYPES = ()
    _PRIM_LABELS = ()
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

class PointToPointDirectedDistance(Constraint):
    """
    Constrain the distance between two points along a given direction
    """

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
        """
        Return the distance error between two points along a given direction

        The distance is measured from the first to the second point along a
        specified direction.
        """
        return jnp.dot(prims[1].param - prims[0].param, self.direction) - self.distance

class PointLocation(Constraint):
    """
    Constrain the location of a point
    """

    primitive_types = (Point, )

    def __init__(
            self,
            location: NDArray
        ):
        self._location = location

    def assem_res(self, prims):
        """
        Return the location error for a point
        """
        return prims[0].param - self._location

class CoincidentPoints(Constraint):
    """
    Constrain two points to coincide
    """

    primitive_types = (Point, Point)

    def assem_res(self, prims):
        """
        Return the coincident error between two points
        """
        return prims[0].param - prims[1].param


class LineSegment(Primitive):
    """
    A straight line segment between two points
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (Point, Point)
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

class ClosedPolyline(PrimitiveArray):
    """
    A closed polygon through a given set of points
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
            idx1 = f'Point{key % len(self)}'
            idx2 = f'Point{(key+1) % len(self)}'
        else:
            raise TypeError("`key`, {key}, must be an integer")

        child_prim_idxs = (idx1, idx2)
        return make_prim, child_prim_idxs

class LineLength(Constraint):
    """
    Constrain the length of a line
    """

    primitive_types = (LineSegment,)

    def __init__(
            self,
            length: float
        ):
        self._length = length

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec = line_direction(prims[0])
        return jnp.linalg.norm(vec) - self._length

class RelativeLineLength(Constraint):
    """
    Constrain a line's length relative to another line's length
    """

    primitive_types = (LineSegment, LineSegment)

    def __init__(
            self,
            length: float
        ):
        self._length = length

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec_a = line_direction(prims[0])
        vec_b = line_direction(prims[1])
        return jnp.linalg.norm(vec_a) - self._length*jnp.linalg.norm(vec_b)

class OrthogonalLines(Constraint):
    """
    Constrain two lines to be orthogonal
    """
    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        """
        Return the orthogonal error
        """
        line0, line1 = prims
        dir0 = line0.prims[1].param - line0.prims[0].param
        dir1 = line1.prims[1].param - line1.prims[0].param
        return jnp.dot(dir0, dir1)

class ParallelLines(Constraint):
    """
    Constrain two lines to be parallel
    """
    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        """
        Return the parallel error
        """
        line0, line1 = prims
        dir0 = line0.prims[1].param - line0.prims[0].param
        dir1 = line1.prims[1].param - line1.prims[0].param
        return jnp.cross(dir0, dir1)

class VerticalLine(Constraint):
    """
    Constrain a line to be vertical
    """
    primitive_types = (LineSegment,)

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        """
        Return the vertical error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([1, 0]))

class HorizontalLine(Constraint):
    """
    Constrain a line to be horizontal
    """
    primitive_types = (LineSegment,)

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        """
        Return the horizontal error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([0, 1]))

class Angle(Constraint):
    """
    Constrain the angle between two lines
    """
    primitive_types = (LineSegment, LineSegment)

    def __init__(
            self,
            angle: NDArray
        ):
        self._angle = angle

    def assem_res(self, prims):
        """
        Return the angle error
        """
        line0, line1 = prims
        dir0 = line_direction(line0)
        dir1 = line_direction(line1)

        dir0 = dir0/jnp.linalg.norm(dir0)
        dir1 = dir1/jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self._angle

class CollinearLines(Constraint):
    """
    Constrain two lines to be collinear
    """
    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        """
        Return the collinearity error
        """
        res_parallel = ParallelLines()
        line0, line1 = prims
        line2 = LineSegment(prims=(line1.prims[1], line0.prims[0]))
        line3 = LineSegment(prims=(line1.prims[0], line0.prims[1]))

        norm = jnp.linalg.norm
        return jnp.array([
            res_parallel.assem_res((line0, line1)),
            res_parallel.assem_res((line0, line2))
        ])


class Box(ClosedPolyline):
    """
    A box primitive with vertical walls and horizontal top and bottom
    """

    _PRIM_TYPES = (Point, Point, Point, Point)

    _CONSTRAINT_TYPES = (
        HorizontalLine,
        VerticalLine,
        HorizontalLine,
        VerticalLine
    )
    _CONSTRAINT_GRAPH = (
        (PrimitiveIndex('', 0),),
        (PrimitiveIndex('', 1),),
        (PrimitiveIndex('', 2),),
        (PrimitiveIndex('', 3),)
    )


def line_direction(line: LineSegment):
    return line.prims[1].param - line.prims[0].param

