from mpllayout import geometry as geo, solver

import numpy as np

## Manually create the primitives + constraints
a = geo.Point([0.0, 0])
b = geo.Point([1, 1.1])

sa, sb = geo.Point([10, 5.0]), geo.Point([11, 15.0])

print(a, b)

prims = [
    a, 
    b,
    geo.LineSegment(prims=(sa, sb)),
        sa, sb
]
constraints = [
    geo.PointLocation(np.array([0, 0])),
    geo.PointToPointAbsDistance(5, [0, 1]),
    geo.PointToPointAbsDistance(5, [1, 0]),
    geo.PointToPointAbsDistance(2.3, [0, 1])
]
constraint_graph = [
    (0,),
    (0, 1),
    (0, 1),
    (3, 4)
]

test = geo.LineSegment(prims=(sa, sb))
prims, constrs, constr_graph, prim_graph = solver.expand_prim(test)
# print(prim_graph)
prim_labels = solver.expand_prim_labels(test, 'Line0')

## Create the same primitives + constraints using `ConstrainedPrimitiveManager`
prim_coll = solver.ConstrainedPrimitiveManager()
 
_prims = [
    a, 
    b,
    geo.LineSegment(prims=(sa, sb))
]
_constraints = [
    geo.PointLocation(np.array([0, 0])),
    geo.PointToPointAbsDistance(5, [0, 1]),
    geo.PointToPointAbsDistance(5, [1, 0]),
    geo.PointToPointAbsDistance(2.3, [0, 1])
]
_constraint_graph = [
    ('Point0',),
    ('Point0', 'Point1'),
    ('Point0', 'Point1'),
    ('LineSegment0.Point0', 'LineSegment0.Point1')
]
for prim in _prims:
    prim_coll.add_prim(prim)

for constraint, prim_idxs in zip(_constraints, _constraint_graph):
    prim_coll.add_constraint(constraint, prim_idxs)

# for constraint_idx, prim_idxs in enumerate(prim_coll.constraint_graph):
#     local_constraint = constraints[constraint_idx]
#     local_prims = tuple(prims[idx] for idx in prim_idxs)
#     print(local_constraint(local_prims))

print(f"Primitive labels: {list(prim_coll.prims.keys())}")
print(prim_coll)
print(prim_coll.prims.keys())
print(prim_coll.prims.values())

## Try the solver
# prim_params, info = solver.solve(prims, constraints, constraint_graph)
constr_prims, info = solver.solve(prim_coll.prims, prim_coll.constraints, prim_coll.constraint_graph)
print(f"Constrained primitives: {constr_prims}")
print(f"Solver info: {info}")


# Test that expanding then contracting a prim are inverses of each other
test = geo.LineSegment(prims=(sa, sb))
child_prims, child_constrs, child_constr_graph, prim_graph = solver.expand_prim(test)
new_test, m = solver.contract_prim(test, child_prims)
# print(test.prims)
# print(m)

# print(test, child_prims)
print(new_test)
