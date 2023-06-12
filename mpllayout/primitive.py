"""
Geometric primitives
"""

import typing as typ
if typ.TYPE_CHECKING:
    from .constraint import Constraint, ConstraintGraph

from numpy.typing import NDArray
import numpy as np
from jax import numpy as jnp

ArrayShape = typ.Tuple[int, ...]
Prims = typ.Tuple['Primitive', ...]

class Primitive:
    """
    Geometric primitive base class

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

    _param: NDArray
    _prims: Prims

    _PARAM_SHAPE: ArrayShape = ()
    _PRIM_TYPES: typ.Union[typ.Tuple[typ.Type['Primitive'], ...], type] = ()
    _CONSTRAINTS: typ.Tuple['Constraint', ...] = ()
    _CONSTRAINT_GRAPH: 'ConstraintGraph' = ()

    def __init__(
            self, 
            param: typ.Optional[NDArray]=None, 
            prims: typ.Optional[Prims]=()
        ):
        # Create default

        # Parameter vector for the primitive
        if not isinstance(param, (np.ndarray, jnp.ndarray)):
            param = np.array(param)

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
        prim_tuple_repr = tuple(prim.__repr__() for prim in self.prims)
        return f'{type(self).__name__}({self.param}, {prim_tuple_repr})'

class Point(Primitive):

    _PARAM_SHAPE = (2,)
    _PRIM_TYPES = ()
    _CONSTRAINTS = ()
    _CONSTRAINT_GRAPH = ()

class PolyLine(Primitive):

    _PARAM_SHAPE = ()
    _PRIM_TYPES = Point
    _CONSTRAINTS = ()
    _CONSTRAINT_GRAPH = ()
