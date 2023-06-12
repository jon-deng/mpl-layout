from mpllayout import constraint as con, primitive as pri

import numpy as np

a = pri.Point([0.0, 0])
b = pri.Point([1, 1.1])

sa, sb = pri.Point([10, 5.0]), pri.Point([11, 15.0])

print(a, b)

prims = [
    a, 
    b,
    pri.PolyLine(prims=(sa, sb))
]
constraints = [
    con.PointLocation(np.array([0, 0])),
    con.PointToPointAbsDistance(5, [0, 1]),
    con.PointToPointAbsDistance(5, [1, 0])
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

prim_params, info = con.solve(prims, constraints, constraint_graph)
print(prim_params, info)
