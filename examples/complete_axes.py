"""
Create a one axes figure with x/y axis labels and stuff too
"""

import numpy as np

import matplotlib as mpl
from mpllayout import solver, geometry as geo, layout as lay, matplotlibutils as lplt

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point([0, 0]), "Origin")
    layout.add_constraint(geo.PointLocation(np.array([0, 0])), ("Origin",))

    ## Create the figure box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    box = geo.Quadrilateral(children=[geo.Point(vert_coords) for vert_coords in verts])
    layout.add_prim(box, "Figure")
    layout.add_constraint(geo.Box(), ("Figure",))

    ## Create the axes box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    frame = geo.Quadrilateral(
        children=[geo.Point(vert_coords) for vert_coords in verts]
    )
    xaxis = geo.Quadrilateral(
        children=[geo.Point(vert_coords) for vert_coords in verts]
    )
    yaxis = geo.Quadrilateral(
        children=[geo.Point(vert_coords) for vert_coords in verts]
    )
    axes = geo.Axes(children=(frame, xaxis, yaxis, geo.Point(), geo.Point()))
    layout.add_prim(axes, "Axes1")
    layout.add_constraint(geo.Box(), ("Axes1/Frame",))

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    layout.add_constraint(
        geo.DirectedDistance(fig_width, np.array([1, 0])),
        ("Figure/Line0/Point0", "Figure/Line0/Point1"),
    )
    layout.add_constraint(
        geo.DirectedDistance(fig_height, np.array([0, 1])),
        ("Figure/Line1/Point0", "Figure/Line1/Point1"),
    )

    layout.add_constraint(geo.CoincidentPoints(), ("Figure/Line0/Point0", "Origin"))

    ## Constrain 'Axes1' elements
    # Constrain left/right margins
    margin_left = 1.1
    margin_right = 1.1
    layout.add_constraint(
        geo.DirectedDistance(margin_left, np.array([-1, 0])),
        ("Axes1/Frame/Line0/Point0", "Figure/Line0/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance(margin_right, np.array([1, 0])),
        ("Axes1/Frame/Line0/Point1", "Figure/Line0/Point1"),
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.DirectedDistance(margin_bottom, np.array([0, -1])),
        ("Axes1/Frame/Line1/Point0", "Figure/Line1/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance(margin_top, np.array([0, 1])),
        ("Axes1/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )

    # Constrain 'Axes1' x/y axis bboxes
    layout.add_constraint(
        geo.Box(),
        ("Axes1/XAxis",),
    )
    layout.add_constraint(
        geo.Box(),
        ("Axes1/YAxis",),
    )

    # Make the x/y axes align the with frame
    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/XAxis/Line1", "Axes1/Frame/Line1"),
    )
    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/XAxis/Line3", "Axes1/Frame/Line3"),
    )

    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/YAxis/Line0", "Axes1/Frame/Line0"),
    )
    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/YAxis/Line2", "Axes1/Frame/Line2"),
    )

    # Make the x/y axes align the with left/right of the frame
    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/XAxis/Line2", "Axes1/Frame/Line0"),
    )
    layout.add_constraint(
        geo.Collinear(),
        ("Axes1/YAxis/Line1", "Axes1/Frame/Line3"),
    )

    ## Constrain 'Axes1' x/y axis label positions
    # layout.add_constraint(
    #     geo.DirectedDistance(0.1, np.array([1, 0])),
    #     ("Axes1/Frame/Line0/Point0", "Axes1/XAxisLabel"),
    # )
    # layout.add_constraint(
    #     geo.DirectedDistance(0.5, np.array([0, 1])),
    #     ("Axes1/Frame/Line0/Point0", "Axes1/XAxisLabel"),
    # )

    # layout.add_constraint(
    #     geo.DirectedDistance(0.2, np.array([1, 0])),
    #     ("Axes1/Frame/Line0/Point0", "Axes1/YAxisLabel"),
    # )
    # layout.add_constraint(
    #     geo.DirectedDistance(0.5, np.array([0, 1])),
    #     ("Axes1/Frame/Line0/Point0", "Axes1/YAxisLabel"),
    # )

    # Align with frame
    # layout.add_constraint(
    #     geo.CoincidentPoints(),
    #     ("Axes1/Frame/Line0/Point1", "Axes1/XAxisLabel"),
    # )

    # layout.add_constraint(
    #     geo.CoincidentPoints(),
    #     ("Axes1/Frame/Line0/Point0", "Axes1/YAxisLabel"),
    # )

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(
        layout.root_prim, layout.constraints, layout.constraint_graph
    )

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    axs["Axes1"].xaxis.set_label_text("My x label")
    axs["Axes1"].yaxis.set_label_text("My y label")
    # plt.draw()

    ax = axs["Axes1"]
    # fig.savefig("out/complete_one_axes_temp.png")

    def get_axis_thickness(axis: mpl.axis.Axis):
        axes = axis.axes
        fig = axes.figure
        fig_width, fig_height = fig.get_size_inches()
        axes_bbox = axes.get_position()
        axis_bbox = axis.get_tightbbox().transformed(fig.transFigure.inverted())
        if isinstance(axis, mpl.axis.XAxis):
            if axis.get_ticks_position() == "bottom":
                return fig_height * (axes_bbox.ymin - axis_bbox.ymin)
            if axis.get_ticks_position() == "top":
                return fig_height * (axis_bbox.ymax - axes_bbox.ymax)
        elif isinstance(axis, mpl.axis.YAxis):
            if axis.get_ticks_position() == "left":
                return fig_width * (axes_bbox.xmin - axis_bbox.xmin)
            if axis.get_ticks_position() == "right":
                return fig_width * (axis_bbox.xmax - axes_bbox.xmax)
        else:
            raise TypeError

    fig_shape = fig.get_size_inches()
    for fig_dim, axis_key, axis, line_label in zip(
        (fig.get_size_inches()[::-1]),
        ("X", "Y"),
        (ax.xaxis, ax.yaxis),
        ("Line1", "Line0"),
    ):
        if axis.get_label().get_visible():
            axis.get_label().set_visible(False)
            thickness = get_axis_thickness(axis)
            axis.get_label().set_visible(True)
        else:
            thickness = get_axis_thickness(axis)

        print(f"{axis_key}Axis thickness: {thickness}")

        # layout.constraints

        # Make the x/y axes have a certain thickness
        layout.add_constraint(
            geo.Length(thickness),
            (f"Axes1/{axis_key}Axis/{line_label}",),
        )

    # Align with axis bboxes
    layout.add_constraint(
        geo.CoincidentPoints(),
        ("Axes1/XAxis/Line0/Point0", "Axes1/XAxisLabel"),
    )

    layout.add_constraint(
        geo.CoincidentPoints(),
        ("Axes1/YAxis/Line0/Point0", "Axes1/YAxisLabel"),
    )

    prim_tree_n, info = solver.solve(
        layout.root_prim, layout.constraints, layout.constraint_graph
    )

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    axs["Axes1"].xaxis.set_label_text("My x label", ha="left", va="top")
    axs["Axes1"].yaxis.set_label_text("My y label", ha="left", va="bottom")

    # fig_shape = fig.get_size_inches()
    # fig.add_axes(lplt.rect_from_box(prim_tree_n[f"Axes1/XAxis"], fig_shape))
    # fig.add_axes(lplt.rect_from_box(prim_tree_n[f"Axes1/YAxis"], fig_shape))

    fig.savefig("out/complete_one_axes.png")
