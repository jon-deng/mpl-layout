from mpllayout import geometry as geo, solver

import numpy as np

## Create a quadrilateral
vertices = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1]
])

points = [geo.Point(vert_coords) for vert_coords in vertices]

quad = geo.ClosedPolyline(prims=points)

print(quad)
print(quad[0])
print(quad[1])

## Constrain the quadrilateral

layout = solver.ConstrainedPrimitiveManager()

layout.add_prim(quad)
layout.add_constraint()
