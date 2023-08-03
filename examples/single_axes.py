"""
Create a single axes from a `bbox`
"""

import numpy as np
from matplotlib import pyplot as plt

from mpllayout import solver, geometry as geo

if __name__ == '__main__':
    layout = solver.Layout()

    # Create an origin point and the figure max extent point
    layout.add_prim(geo.Point([0, 0]), 'Origin')
    layout.add_constraint(geo.PointLocation(np.array([0, 0])), ('Origin',))

    layout.add_prim(geo.Point([0, 0]), 'TopRightCorner')
    layout.add_constraint(geo.PointLocation(np.array([5, 5])), ('TopRightCorner',))

    # Create a box representing the bbox for the axes
    xmin, xmax = 1, 4
    ymin, ymax = 1, 2
    box_points = [
        geo.Point([xmin+0.1, ymin+0.5]),
        geo.Point([xmax+0.6, ymin]),
        geo.Point([xmax, ymax]),
        geo.Point([xmin-0.1, ymax])
    ]
    box_lines = [
        geo.LineSegment(prims=(pointa, pointb))
        for pointa, pointb in zip(box_points, box_points[1:]+box_points[:1])
    ]
    layout.add_prim(geo.Box(prims=box_lines), 'Axes1')

    # Constrain the margins of the axes
    margin_x = 1.1
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_x, np.array([-1, 0])),
        ('Axes1.LineSegment0.Point0', 'Origin')
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_x, np.array([1, 0])),
        ('Axes1.LineSegment0.Point1', 'TopRightCorner')
    )

    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_bottom, np.array([0, -1])),
        ('Axes1.LineSegment0.Point0', 'Origin')
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_top, np.array([0, 1])),
        ('Axes1.LineSegment1.Point1', 'TopRightCorner')
    )

    prims, info = solver.solve(
        layout.prims, layout.constraints, layout.constraint_graph
    )

    width = prims['TopRightCorner'].param[0] - prims['Origin'].param[0]
    height = prims['TopRightCorner'].param[1] - prims['Origin'].param[1]

    print(prims['Axes1'])

    def rect_from_box(box: geo.Box, fig_size=(1, 1)):
        fig_w, fig_h = fig_size

        point_bottomleft = box.prims[0].prims[0]
        xmin = point_bottomleft.param[0]/fig_w
        ymin = point_bottomleft.param[1]/fig_h

        point_topright = box.prims[1].prims[1]
        xmax = point_topright.param[0]/fig_w
        ymax = point_topright.param[1]/fig_h

        return (xmin, ymin, (xmax-xmin), (ymax-ymin))

    print(rect_from_box(prims['Axes1'], fig_size=(width, height)))

    fig = plt.Figure((width, height))
    # ax = fig.add_axes((0, 0, 1, 1))
    ax = fig.add_axes(rect_from_box(prims['Axes1'], (width, height)))
    # print(ax)
    ax.plot([0, 1], [0, 1])

    fig.savefig('out/test.png')
