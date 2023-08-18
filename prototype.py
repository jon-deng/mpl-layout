from mpllayout import geometry as geo, solver

import numpy as np

## Create a quadrilateral
vertices = np.array([
    [0, 0], [1, -0.25], [1, 1], [0, 1]
])

points = [geo.Point(vert_coords) for vert_coords in vertices]
box = geo.ClosedPolyline(prims=points)

print("box:", box)
print("box[0]:", box[0])
print("box[1]:", box[1])

## Constrain the quadrilateral

layout = solver.Layout()
layout.add_prim(box, 'MyBox')
layout.add_constraint(geo.Horizontal(), ('MyBox',), (1,))

print("layout.constraints:", layout.constraints.values())
print("layout.constraint_graph:", layout.constraint_graph)

new_prims, info = solver.solve(layout.prims, layout.constraints, layout.constraint_graph)
breakpoint()
