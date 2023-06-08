

from numpy.typing import NDArray
import numpy as np

class Primitive:
    pass

class Point(Primitive):

    def __init__(self, coords: NDArray):
        self._coords = np.array(coords)

    @property
    def coords(self):
        return self._coords
    
    def __repr__(self):
        return f'Point({self.coords})'
