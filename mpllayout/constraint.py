
from typing import Optional, List
from numpy.typing import NDArray

import numpy as np

from .primitive import Primitive, Point

def solve(constraints: List['Constraint']):
    # Build a mapping for each primitive to it's associated constraints
    prim_to_constraints = {}
    for constraint in constraints:
        for prim in constraint.prims:
            if prim in prim_to_constraints:
                prim_to_constraints[prim].append(constraint)
            else:
                prim_to_constraints[prim] = [constraint]

    # Check that each primitive has a set of complete constraints
    for prim, constraints in prim_to_constraints.items():
        if isinstance(prim, Point):
            assert len(constraints) == 2

            assert not np.isclose(
                np.dot(constraints[0].dir, constraints[1].dir), 0
            )




class Constraint:
    
    def __init__(self, prims: List[Primitive]):
        self._prims = tuple(prims)

    @property
    def prims(self):
        return self._prims

class PointToPointAbsDistance(Constraint):

    def __init__(
            self, pointa: Point, pointb: Point, 
            distance: float, dir: Optional[NDArray]=None
        ):
        super().__init__((pointa, pointb))

        if dir is None:
            dir = np.array([1, 0])

        self._dir = dir
        self._distance = distance

    @property
    def distance(self):
        return self._distance
