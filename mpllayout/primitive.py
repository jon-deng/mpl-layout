"""
Geometric primitives
"""

from numpy.typing import NDArray
import numpy as np
from jax import numpy as jnp

class Primitive:
    """
    Geometric primitive base class

    Parameters
    ----------
    param: ArrayLike with shape (n,)
        A parameter vector for the primitive
    """

    def __init__(self, param: NDArray):

        # Parameter vector for the primitive
        if not isinstance(param, (np.ndarray, jnp.ndarray)):
            param = np.array(param)
        
        self._param: NDArray = param

    @property
    def param(self):
        """
        Return the primitive's parameter vector
        """
        return self._param

class Point(Primitive):

    def __init__(self, param: NDArray):
        super().__init__(param)

        assert self.param.shape == (2,)
    
    def __repr__(self):
        return f'Point({self.param})'
