"""
Create a two axes figure
"""

from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from mpllayout import solver, geometry as geo, matplotlibutils as lplt

PrimIdx = geo.PrimIdx

if __name__ == '__main__':
    layout = solver.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point([0, 0]), 'Origin')
    layout.add_constraint(geo.PointLocation(np.array([0, 0])), (PrimIdx('Origin'),))

    ## Create the figure box
    verts = [
        [0, 0], [5, 0], [5, 5], [0, 5]
    ]
    box = geo.Box(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Figure')

    ## Create the two axes box
    verts = [
        [0, 0], [5, 0], [5, 5], [0, 5]
    ]
    box = geo.Box(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Axes1')

    verts = [
        [0, 0], [5, 0], [5, 5], [0, 5]
    ]
    box = geo.Box(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Axes2')

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    # layout.add_constraint(
    #     geo.PointToPointAbsDistance(fig_width, np.array([1, 0])),
    #     (PrimIdx('Figure.Point0'), PrimIdx('Figure.Point1'))
    # )
    layout.add_constraint(
        geo.LineLength(fig_width),
        (PrimIdx('Figure', 0),)
    )
    # layout.add_constraint(
    #     geo.PointToPointAbsDistance(fig_height, np.array([0, 1])),
    #     (PrimIdx('Figure.Point0'), PrimIdx('Figure.Point3'))
    # )
    layout.add_constraint(
        geo.LineLength(fig_height),
        (PrimIdx('Axes1', 1),)
    )

    layout.add_constraint(
        geo.CoincidentPoint(),
        (PrimIdx('Figure.Point0'), PrimIdx('Origin'))
    )

    ## Constrain 'Axes1' margins
    # Constrain left/right margins
    margin_left = 2.1
    margin_right = 1.1
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_left, np.array([-1, 0])),
        (PrimIdx('Axes1.Point0'), PrimIdx('Figure.Point0'))
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_right, np.array([1, 0])),
        (PrimIdx('Axes2.Point1'), PrimIdx('Figure.Point2'))
    )

    # Constrain the 'Axes1' width
    width = 0.5
    # layout.add_constraint(
    #     geo.PointToPointAbsDistance(width, np.array([1, 0])),
    #     (PrimIdx('Axes1.Point0'), PrimIdx('Axes1.Point1'))
    # )
    layout.add_constraint(
        geo.LineLength(width),
        (PrimIdx('Axes1', 0),)
    )

    # Constrain the inter-axis margin
    margin_inter = 0.5
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_inter, np.array([1, 0])),
        (PrimIdx('Axes1.Point1'), PrimIdx('Axes2.Point0'))
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_bottom, np.array([0, -1])),
        (PrimIdx('Axes1.Point0'), PrimIdx('Figure.Point0'))
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_top, np.array([0, 1])),
        (PrimIdx('Axes1.Point2'), PrimIdx('Figure.Point2'))
    )

    # layout.add_constraint(
    #     geo.PointToPointAbsDistance(margin_bottom, np.array([0, -1])),
    #     (PrimIdx('Axes2.Point0'), PrimIdx('Figure.Point0'))
    # )
    # layout.add_constraint(
    #     geo.PointToPointAbsDistance(margin_top, np.array([0, 1])),
    #     (PrimIdx('Axes2.Point2'), PrimIdx('Figure.Point2'))
    # )

    # Make axes line-up along the tops/bottoms
    layout.add_constraint(
        geo.Collinear(),
        (PrimIdx('Axes1', 0), PrimIdx('Axes2', 0))
    )
    layout.add_constraint(
        geo.Collinear(),
        (PrimIdx('Axes1', 2), PrimIdx('Axes2', 2))
    )

    ## Solve the constraints and form the figure/axes layout
    prims, info = solver.solve(
        layout.prims, layout.constraints, layout.constraint_graph,
        max_iter=40, rel_tol=1e-9
    )
    pprint(info)

    print('Figure:', prims['Figure'])
    print('Axes1:', prims['Axes1'])
    print('Axes2:', prims['Axes2'])

    fig, axs = lplt.subplots(prims)
    print(axs)
    # breakpoint()

    x = np.linspace(0, 1)
    axs['Axes1'].plot(x, 4*x)
    axs['Axes2'].plot(x, x**2)

    fig.savefig('out/two_axes.png')
