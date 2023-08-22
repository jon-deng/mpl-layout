"""
Create a single axes figure
"""

import numpy as np
from matplotlib import pyplot as plt

from mpllayout import solver, geometry as geo

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

    layout.add_constraint(
        geo.CoincidentPoint(),
        (PrimIdx('Figure.Point0'), PrimIdx('Origin'))
    )

    ## Create the axes box
    verts = [
        [0, 0], [5, 0], [5, 5], [0, 5]
    ]
    box = geo.Box(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Axes1')

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    layout.add_constraint(
        geo.PointToPointAbsDistance(fig_width, np.array([1, 0])),
        (PrimIdx('Figure.Point0'), PrimIdx('Figure.Point1'))
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(fig_height, np.array([0, 1])),
        (PrimIdx('Figure.Point0'), PrimIdx('Figure.Point3'))
    )

    ## Constrain 'Axes1' margins
    # Constrain left/right margins
    margin_left = 1.1
    margin_right = 1.1
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_left, np.array([-1, 0])),
        (PrimIdx('Axes1.Point0'), PrimIdx('Figure.Point0'))
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_right, np.array([1, 0])),
        (PrimIdx('Axes1.Point2'), PrimIdx('Figure.Point2'))
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

    ## Solve the constraints and form the figure/axes layout
    prims, info = solver.solve(
        layout.prims, layout.constraints, layout.constraint_graph
    )

    print('Figure:', prims['Figure'])
    print('Axes1:', prims['Axes1'])

    def wh_from_box(box: geo.Box):

        point_bottomleft = box.prims[0]
        xmin = point_bottomleft.param[0]
        ymin = point_bottomleft.param[1]

        point_topright = box.prims[2]
        xmax = point_topright.param[0]
        ymax = point_topright.param[1]

        return (xmax-xmin), (ymax-ymin)

    def rect_from_box(box: geo.Box, fig_size=(1, 1)):
        fig_w, fig_h = fig_size

        point_bottomleft = box.prims[0]
        xmin = point_bottomleft.param[0]/fig_w
        ymin = point_bottomleft.param[1]/fig_h

        point_topright = box.prims[2]
        xmax = point_topright.param[0]/fig_w
        ymax = point_topright.param[1]/fig_h

        return (xmin, ymin, (xmax-xmin), (ymax-ymin))
    
    width, height = wh_from_box(prims['Figure'])
    print((width, height))

    print(rect_from_box(prims['Axes1'], fig_size=(width, height)))

    fig = plt.Figure((width, height))
    # ax = fig.add_axes((0, 0, 1, 1))
    ax = fig.add_axes(rect_from_box(prims['Axes1'], (width, height)))
    # print(ax)
    ax.plot([0, 1], [0, 1])

    fig.savefig('out/test.png')
