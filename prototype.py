from mpllayout import constraint as cons, primitive as primi

import numpy as np

a = primi.Point([0.0, 0])
b = primi.Point([1, 1.1])

print(a, b)

prims = [
    a, 
    b
]
constraints = [
    cons.PointLocation(np.array([0, 0])),
    cons.PointToPointAbsDistance(5, [0, 1]),
    cons.PointToPointAbsDistance(5, [1, 0])
]
constraint_graph = [
    (0,),
    (0, 1),
    (0, 1)
]

for constraint_idx, prim_idxs in enumerate(constraint_graph):
    local_constraint = constraints[constraint_idx]
    local_prims = tuple(prims[idx] for idx in prim_idxs)
    print(local_constraint(local_prims))

prim_params, info = cons.solve(prims, constraints, constraint_graph)
print(prim_params, info)
