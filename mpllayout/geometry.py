"""
Geometric primitive and constraints 
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np

import jax.numpy as jnp

Prims = typ.Tuple['Primitive', ...]
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
Prims = typ.Tuple['Primitive', ...]

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
    """

    _param: NDArray
    _prims: Prims

    _PARAM_SHAPE: ArrayShape = (0,)
    _PRIM_TYPES: typ.Union[typ.Tuple[typ.Type['Primitive'], ...], typ.Type['Primitive']] = ()
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

class PrimitiveList(Primitive):
    """
    A geometric primitive representing a list of primitives

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
        raise NotImplementedError
    
    def _list_spec(self, key) -> typ.Tuple[typ.Callable, typ.Tuple[int, ...]]:
        """
        Return a helper function that lets you apply a constraint...
        """
        raise NotImplementedError

class Constraint:
    """
    Constraint base class
    """
    
    primitive_types: typ.Tuple[typ.Type['Primitive'], ...]

    def __call__(self, prims: typ.Tuple['Primitive', ...]):
        # Check the input primitives are valid
        assert len(prims) == len(self.primitive_types)
        for prim, prim_type in zip(prims, self.primitive_types):
            assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))
    
    def assem_res(self, prims: typ.Tuple['Primitive', ...]) -> NDArray:
        raise NotImplementedError()


class Point(Primitive):

    _PARAM_SHAPE = (2,)
    _PRIM_TYPES = ()
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

class ClosedPolyline(PrimitiveList):
    """
    A closed polyline passing through a given set of points
    """

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = Point
    _CONSTRAINT_TYPES = ()
    _CONSTRAINT_GRAPH = ()

    def __len__(self):
        return len(self.prims) - 1

    def __getitem__(self, key):
        make_prim, prim_idxs = self._list_spec(key)
        return make_prim(tuple(self.prims[idx] for idx in prim_idxs))
        
    def _list_spec(self, key):
        def make_prim(prims):
            return LineSegment(prims=prims)
        
        if isinstance(key, int):
            idx1 = key
            if key == -1 or key == len(self)-1:
                idx2 = 0
            else:
                idx2 = idx1+1
        else:
            raise TypeError("`key`, {key}, must be an integer")
        
        prim_idxs = (idx1, idx2)
        return make_prim, prim_idxs

class CoincidentLine(Constraint):

    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims):
        # This joins the start of the second line with the end of the first line
        return prims[1].prims[0].param - prims[0].prims[1].param

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
            jnp.dot(dir0, dir1) - jnp.linalg.norm(dir0)*jnp.linalg.norm(dir1), 
            jnp.dot(dir0, dir_inter) - jnp.linalg.norm(dir0)*jnp.linalg.norm(dir_inter)
        ])

class Box(Primitive):

    _PARAM_SHAPE = (0,)
    _PRIM_TYPES = (LineSegment, LineSegment, LineSegment, LineSegment)
    _CONSTRAINT_TYPES = (
        CoincidentLine, CoincidentLine, CoincidentLine, CoincidentLine,
        # Orthogonal, Orthogonal, Orthogonal, Orthogonal,
        Horizontal, Vertical, Horizontal, Vertical
    )
    _CONSTRAINT_GRAPH = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        # (0, 1), (1, 2), (2, 3), (3, 0),
        (0,), (1,), (2,), (3,)
    )


def line_direction(line: LineSegment):
    return line.prims[1].param - line.prims[0].param
