"""
Create a single axes figure
"""

import numpy as np
from matplotlib import pyplot as plt

from mpllayout import solver, geometry as geo

if __name__ == '__main__':
    layout = solver.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point([0, 0]), 'Origin')
    layout.add_constraint(geo.PointLocation(np.array([0, 0])), ('Origin',))

    ## Create the Figure box
    verts = np.array([
        [0, 0], [5, 0], [5, 5], [0, 5]
    ])
    box = geo.ClosedPolyline(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Figure')

    layout.add_constraint(
        geo.CoincidentPoint(),
        ('Figure.Point0', 'Origin')
    )

    # Make the box square
    layout.add_constraint(
        geo.Horizontal(), ('Figure',), (0,)
    )
    # layout.add_constraint(
    #     geo.Vertical(), ('Figure',), (1,)
    # )
    # layout.add_constraint(
    #     geo.Horizontal(), ('Figure',), (2,)
    # )
    # layout.add_constraint(
    #     geo.Vertical(), ('Figure',), (3,)
    # )

    ## Create the axes box
    verts = np.array([
        [0, 0], [5, 0], [5, 5], [0, 5]
    ])
    box = geo.ClosedPolyline(
        prims=[geo.Point(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, 'Axes1')

    # Constrain 'Axes1' margins
    margin_x = 1.1
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_x, np.array([-1, 0])),
        ('Axes1.Point0', 'Figure.Point0')
    )
    layout.add_constraint(
        geo.PointToPointAbsDistance(margin_x, np.array([1, 0])),
        ('Axes1.Point2', 'Figure.Point2')
    )

    prims, info = solver.solve(
        layout.prims, layout.constraints, layout.constraint_graph
    )

    print(prims['Axes1'])

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
