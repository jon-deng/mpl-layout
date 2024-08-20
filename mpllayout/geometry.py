"""
Geometric primitive and constraints
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp

from .array import LabelledTuple

PrimList = typ.Tuple['Primitive', ...]
ConstraintList = typ.List['Constraint']
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
PrimTuple = typ.Tuple['Primitive', ...]

class PrimitiveIndex:
    """
    An index to a `Primitive` instance and, potentially, `PrimitiveArray` element(s)

    A `PrimitiveIndex` represents an index to a primitive within a collection
    where primitives are labelled by unique strings.
    There are two main use cases:
        - an index to a specific primitive within a collection
        - an index to a child primitive from a parent primitive

    In either case the first argument represents the label of the desired
    primitive while the second argument is an integer index if the desired
    primitive is a `PrimitiveArray` type. Periods in the label denote a child
    primitive. For example, `PrimitiveIndex('MyBox.Point0')`, denotes the first
    point, `'Point0'`, of the primitive called `'MyBox'`. As another example,
    `PrimitiveIndex('MyBox', 0)`, denotes the first line segment of the
    primitive called `'MyBox'`, in the case the primitive is a
    `Polyline`.

    Parameters
    ----------
    label: str
        The string identifier for the primitive.
        When indexing from a collection of primitives, the string label has the
        form:
        `'parent_prim_label.child_prim_label.etc'`.
        When indexing a child primitive, however, the string label has the form:
        `'.child_prim_label.etc'`.
    array_idx: int
        An integer representing an indexed primitive when `label` points to a
        `PrimitiveArray` type primitive.

    Attributes
    ----------
    label: str
        See 'Parameters'
    array_idx: int
        See 'Parameters'
    """

    def __init__(
            self,
            label: str,
            array_idx: typ.Optional[int]=None
        ):

        self._label = label
        self._array_idx = array_idx

    @property
    def array_idx(self):
        return self._array_idx

    @property
    def label(self):
        return self._label

    def __repr__(self):
        return f'PrimitiveIndex({self.label}, {self.array_idx})'

    def __str__(self):
        return self.__repr__()

## Basic geometric primitives

PrimIdxConstraintGraph = typ.List[typ.Tuple[PrimitiveIndex, ...]]
class Primitive:
    """
    A representation of a geometric primitive

    A `Primitive` can be parameterized by a parameter vector as well as
    other geometric primitives. For example, a point in 2D is parameterized by a
    vector representing (x, y) coordinates. Primitives can also contain implicit
    constraints to represent common use-cases. For example, an origin point may
    be explicitly constrained to have (0, 0) coordinates.

    To create a `Primitive` class, subclass `Primitive` and define the class
    attributes `_PARAM_SHAPE`, `_PRIM_TYPES`, `_PRIM_LABELS`,
    `_CONSTRAINT_TYPES`, and `_CONSTRAINT_GRAPH`.

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
    constraints: LabelledTuple['Constraint']
        If non-empty, the primitive contains implicit geometric constraints in
        `self.constraints`
    constraint_graph: PrimIdxConstraintGraph
        A graph representing which primitives a constraint applies to

    _PARAM_SHAPE: ArrayShape
        The shape of the parameter vector parameterizing the `Primitive`
    _PRIM_TYPES: typ.Union[
            typ.Tuple[typ.Type['Primitive'], ...],
            typ.Type['Primitive']
        ]
        The types of child primitives parameterizing the `Primitive`
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]]
        Optional labels for the child primitives
    _CONSTRAINT_TYPES: typ.Tuple['Constraint', ...]
        The types of any internal constraints on the `Primitive`
    _CONSTRAINT_GRAPH: 'PrimIdxConstraintGraph'
        The constraint graph for the internal constraints
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
    _CONSTRAINT_TYPES: typ.Tuple['Constraint', ...] = ()
    _CONSTRAINT_GRAPH: PrimIdxConstraintGraph = ()

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

        # Create any internal constraints
        self._constraints: LabelledTuple['Constraint', ...] = LabelledTuple(
            [Constraint() for Constraint in self._CONSTRAINT_TYPES]
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
        return self._constraints

    @property
    def constraint_graph(self) -> PrimIdxConstraintGraph:
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
    A representation of an array of geometric primitives

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
    constraints: LabelledTuple['Constraint']
        If non-empty, the primitive contains implicit geometric constraints in
        `self.constraints`
    constraint_graph: PrimIdxConstraintGraph
        A graph representing which primitives a constraint applies to

    _PARAM_SHAPE: ArrayShape
        The shape of the parameter vector parameterizing the `Primitive`
    _PRIM_TYPES: typ.Union[
            typ.Tuple[typ.Type['Primitive'], ...],
            typ.Type['Primitive']
        ]
        The types of child primitives parameterizing the `Primitive`
    _PRIM_LABELS: typ.Optional[typ.Union[typ.Tuple[str, ...], str]]
        Optional labels for the child primitives
    _CONSTRAINT_TYPES: typ.Tuple['Constraint', ...]
        The types of any internal constraints on the `Primitive`
    _CONSTRAINT_GRAPH: 'PrimIdxConstraintGraph'
        The constraint graph for the internal constraints
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

class Constraint:
    """
    A representation of a constraint on primitives

    A constraint represents a condition on the parameters of geometric
    primitive(s). The condition is specified through a residual function
    `assem_res`, specified with the `jax` library. Usage of `jax` in specifying
    the constraints allows automatic differentiation of constraint conditions,
    which is used for solving constraints.

    Parameters
    ----------
    *args, **kwargs :
        Specific parameters that control a constraint, for example, an angle or
        distance

        See `Constraint` subclasses for specific examples.

    Attributes
    ----------
    _PRIMITIVE_TYPES: typ.Tuple[typ.Type['Primitive'], ...]
        The types of primitives accepted by `assem_res`

        This is used for type checking.
    """

    _PRIMITIVE_TYPES: typ.Tuple[typ.Type['Primitive'], ...]

    def __init__(self, *args, **kwargs):
        self._res_args = args
        self._res_kwargs = kwargs

    def __call__(self, prims: typ.Tuple['Primitive', ...]):
        # Check the input primitives are valid
        # assert len(prims) == len(self._PRIMITIVE_TYPES)
        # for prim, prim_type in zip(prims, self._PRIMITIVE_TYPES):
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
    A constraint on distance between two point along a direction
    """

    _PRIMITIVE_TYPES = (Point, Point)

    def __init__(
            self, distance: float, direction: typ.Optional[NDArray]=None
        ):
        if direction is None:
            direction = np.array([1, 0])
        else:
            direction = np.array(direction)
        super().__init__(distance=distance, direction=direction)

    def assem_res(self, prims):
        """
        Return the distance error between two points along a given direction

        The distance is measured from the first to the second point along a
        specified direction.
        """
        distance = jnp.dot(
            prims[1].param - prims[0].param, self._res_kwargs['direction']
        )
        return distance - self._res_kwargs['distance']

class PointLocation(Constraint):
    """
    A constraint on the location of a point
    """

    _PRIMITIVE_TYPES = (Point, )

    def __init__(
            self,
            location: NDArray
        ):
        super().__init__(location=location)

    def assem_res(self, prims):
        """
        Return the location error for a point
        """
        return prims[0].param - self._res_kwargs['location']

class CoincidentPoints(Constraint):
    """
    A constraint on coincide of two points
    """

    _PRIMITIVE_TYPES = (Point, Point)

    def __init__(self):
        super().__init__()

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
    A constraint on the length of a line
    """

    _PRIMITIVE_TYPES = (LineSegment,)

    def __init__(
            self,
            length: float
        ):
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec = line_direction(prims[0])
        return jnp.sum(vec**2) - self._res_kwargs['length']**2

class RelativeLineLength(Constraint):
    """
    A constraint on relative length between two lines
    """

    _PRIMITIVE_TYPES = (LineSegment, LineSegment)

    def __init__(
            self,
            length: float
        ):
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec_a = line_direction(prims[0])
        vec_b = line_direction(prims[1])
        return jnp.sum(vec_a**2) - self._res_kwargs['length']**2 * jnp.sum(vec_b**2)

class OrthogonalLines(Constraint):
    """
    A constraint on orthogonality of two lines
    """
    _PRIMITIVE_TYPES = (LineSegment, LineSegment)

    def __init__(self):
        super().__init__()

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
    A constraint on parallelism of two lines
    """
    _PRIMITIVE_TYPES = (LineSegment, LineSegment)

    def __init__(self):
        super().__init__()

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
    A constraint that a line must be vertical
    """
    _PRIMITIVE_TYPES = (LineSegment,)

    def __init__(self):
        super().__init__()

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        """
        Return the vertical error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([1, 0]))

class HorizontalLine(Constraint):
    """
    A constraint that a line must be horizontal
    """
    _PRIMITIVE_TYPES = (LineSegment,)

    def __init__(self):
        super().__init__()

    def assem_res(self, prims: typ.Tuple[LineSegment]):
        """
        Return the horizontal error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([0, 1]))

class Angle(Constraint):
    """
    A constraint on the angle between two lines
    """
    _PRIMITIVE_TYPES = (LineSegment, LineSegment)

    def __init__(
            self,
            angle: NDArray
        ):
        super().__init__(angle=angle)

    def assem_res(self, prims):
        """
        Return the angle error
        """
        line0, line1 = prims
        dir0 = line_direction(line0)
        dir1 = line_direction(line1)

        dir0 = dir0/jnp.linalg.norm(dir0)
        dir1 = dir1/jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self._res_kwargs['angle']

class CollinearLines(Constraint):
    """
    A constraint on the collinearity of two lines
    """
    _PRIMITIVE_TYPES = (LineSegment, LineSegment)

    def __init__(self):
        super().__init__()

    def assem_res(self, prims: typ.Tuple[LineSegment, LineSegment]):
        """
        Return the collinearity error
        """
        res_parallel = ParallelLines()
        line0, line1 = prims
        line2 = LineSegment(prims=(line1.prims[1], line0.prims[0]))
        line3 = LineSegment(prims=(line1.prims[0], line0.prims[1]))

        return jnp.array([
            res_parallel.assem_res((line0, line1)),
            res_parallel.assem_res((line0, line2))
        ])


class Box(ClosedPolyline):
    """
    A box with vertical sides and horizontal top and bottom
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

