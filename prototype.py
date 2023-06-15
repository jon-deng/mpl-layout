from mpllayout import constraint as con, primitive as pri, solver

import numpy as np

a = pri.Point([0.0, 0])
b = pri.Point([1, 1.1])

sa, sb = pri.Point([10, 5.0]), pri.Point([11, 15.0])

print(a, b)

prims = [
    a, 
    b,
    pri.PolyLine(prims=(sa, sb)),
        sa, sb
]
constraints = [
    con.PointLocation(np.array([0, 0])),
    con.PointToPointAbsDistance(5, [0, 1]),
    con.PointToPointAbsDistance(5, [1, 0]),
    con.PointToPointAbsDistance(2.3, [0, 1])
]
constraint_graph = [
    (0,),
    (0, 1),
    (0, 1),
    (3, 4)
]

subprim_graph = []

for constraint_idx, prim_idxs in enumerate(constraint_graph):
    local_constraint = constraints[constraint_idx]
    local_prims = tuple(prims[idx] for idx in prim_idxs)
    print(local_constraint(local_prims))

prim_params, info = solver.solve(prims, constraints, constraint_graph, subprim_graph)
print(prim_params, info)


# Test that expanding then contracting a prim are inverses of each other
test = pri.PolyLine(prims=(sa, sb))
child_prims, child_constrs, child_constr_graph, prim_graph = solver.expand_prim(test)
new_test = solver.contract_prim(test, child_prims, prim_graph)

print(test, child_prims)
print(new_test)