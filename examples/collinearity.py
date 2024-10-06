"""
Create a two axes figure
"""

from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from mpllayout import solver, geometry as geo, matplotlibutils as lplt

PrimIdx = geo.PrimitiveIndex

if __name__ == "__main__":
    layout = solver.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point.from_std([0, 0]), "Origin")
    layout.add_constraint(geo.PointLocation.from_std((np.array([0, 0]),)), (PrimIdx("Origin"),))

    ## Create two lines
    vert_coords = [[0, 0], [1, 0]]
    layout.add_prim(
        geo.Line.from_std(children=[geo.Point.from_std(coord) for coord in vert_coords]), "LineA"
    )

    vert_coords = [[2, 0], [3, 3]]
    layout.add_prim(
        geo.Line.from_std(children=[geo.Point.from_std(coord) for coord in vert_coords]), "LineB"
    )

    # Fix LineA
    layout.add_constraint(
        geo.CoincidentPoints.from_std({}), (PrimIdx("LineA.Point0"), PrimIdx("Origin"))
    )
    layout.add_constraint(geo.Horizontal.from_std({}), (PrimIdx("LineA"),))
    layout.add_constraint(
        geo.DirectedDistance.from_std(5, [1, 0]),
        (PrimIdx("LineA.Point0"), PrimIdx("LineA.Point1")),
    )

    # Fix LineB
    layout.add_constraint(
        geo.CoincidentPoints.from_std({}), (PrimIdx("LineA.Point1"), PrimIdx("LineB.Point0"))
    )
    layout.add_constraint(geo.Collinear.from_std({}), (PrimIdx("LineA"), PrimIdx("LineB")))
    layout.add_constraint(
        geo.DirectedDistance.from_std((5, [1, 0])),
        (PrimIdx("LineB.Point0"), PrimIdx("LineB.Point1")),
    )

    ## Solve the constraints and form the figure/axes layout
    prims, info = solver.solve(
        layout.root_prim, layout.constraints, layout.constraint_graph, max_iter=40
    )
    pprint(info)

    print("prims['LineA']:", prims["LineA"])
    print("prims['LineB']:", prims["LineB"])
