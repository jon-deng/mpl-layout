from mpllayout import geometry as geo, solver

import numpy as np

## Create a quadrilateral
vertices = np.array([
    [0, 0], [1, -0.25], [1, 1], [0, 1]
])

points = [geo.Point(vert_coords) for vert_coords in vertices]
box = geo.ClosedPolyline(prims=points)

print("box:", box)
for ii in range(4):
    print(f"box[{ii}]:", box[ii])

## Constrain the quadrilateral

layout = solver.Layout()

layout.add_prim(geo.Point([0, 0]), 'Origin')
layout.add_prim(box, 'MyBox')
# layout.add_constraint(geo.CoincidentPoint(), ('MyBox.Point0', 'Origin'))
layout.add_constraint(geo.Horizontal(), ('MyBox',), (0,))
layout.add_constraint(geo.Vertical(), ('MyBox',), (1,))
layout.add_constraint(geo.Horizontal(), ('MyBox',), (2,))
# layout.add_constraint(geo.Vertical(), ('MyBox',), (3,))

print("layout.constraints:", layout.constraints.values())
print("layout.constraint_graph:", layout.constraint_graph)

new_prims, info = solver.solve(layout.prims, layout.constraints, layout.constraint_graph)
breakpoint()
