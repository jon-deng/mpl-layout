"""
Geometric constraints
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np

import jax.numpy as jnp

# from . import primitive as pri
from mpllayout.primitive import Primitive, Point, LineSegment

Prims = typ.Tuple[Primitive, ...]

Idxs = typ.Tuple[int]


class Constraint:
    """
    Constraint base class
    """
    
    primitive_types: typ.Tuple[typ.Type[Primitive], ...]

    def __call__(self, prims: typ.Tuple[Primitive, ...]):
        # Check the input primitives are valid
        assert len(prims) == len(self.primitive_types)
        for prim, prim_type in zip(prims, self.primitive_types):
            assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))
    
    def assem_res(self, prims: typ.Tuple[Primitive, ...]) -> NDArray:
        raise NotImplementedError()

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
    
class CoincidentLine(Constraint):

    primitive_types = (LineSegment, LineSegment)

    def assem_res(self, prims):
        # This joins the start of the second line with the end of the first line
        return prims[1][0].param - prims[0][1].param
