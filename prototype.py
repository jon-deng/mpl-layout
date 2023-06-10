from mpllayout import constraint as cons, primitive as primi

a = primi.Point([0.0, 0])
b = primi.Point([1, 1.1])

print(a, b)

prims = [
    a, 
    b
]
constraints = [
    cons.PointToPointAbsDistance(5, [0, 1]),
    cons.PointToPointAbsDistance(5, [1, 0])
]
constraint_graph = [
    (0, (0, 1)),
    (1, (0, 1))
]

test = {
    a: 'a',
    b: 'b'
}

for (constraint_idx, prim_idxs) in constraint_graph:
    local_constraint = constraints[constraint_idx]
    local_prims = tuple(prims[idx] for idx in prim_idxs)
    print(local_constraint(local_prims))

jac = cons.solve(constraints, prims, constraint_graph)
print(jac)
