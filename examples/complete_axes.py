"""
Create a one axes figure with x/y axis labels and stuff too
"""

import numpy as np

import matplotlib as mpl

from mpllayout import (
    solver,
    geometry as geo,
    layout as lay,
    matplotlibutils as lplt,
    constraints as con,
)

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create constant constraints
    # (these have no parameters so are reused a fair bit)
    BOX = geo.Box.from_std(())
    COLLINEAR = geo.Collinear.from_std({})
    COINCIDENT = geo.Coincident.from_std({})

    ## Create an origin point
    layout.add_prim(geo.Point.from_std(), "Origin")
    layout.add_constraint(geo.Fix.from_std((np.array([0, 0]),)), ("Origin",))

    ## Create the Figure quad
    layout.add_prim(geo.Quadrilateral.from_std(), "Figure")
    layout.add_constraint(BOX, ("Figure",))

    ## Create the Axes quads
    layout.add_prim(geo.StandardAxes.from_std(), "Axes1")
    layout.add_constraint(BOX, ("Axes1/Frame",))

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    layout.add_constraint(
        geo.DirectedDistance.from_std((fig_width, np.array([1, 0]))),
        ("Figure/Line0/Point0", "Figure/Line0/Point1"),
    )
    layout.add_constraint(
        geo.DirectedDistance.from_std((fig_height, np.array([0, 1]))),
        ("Figure/Line1/Point0", "Figure/Line1/Point1"),
    )

    layout.add_constraint(
        geo.Coincident.from_std({}), ("Figure/Line0/Point0", "Origin")
    )

    ## Constrain 'Axes1' elements
    # Constrain left/right margins
    margin_left = 1.1
    margin_right = 1.1
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_left, np.array([-1, 0]))),
        ("Axes1/Frame/Line0/Point0", "Figure/Line0/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_right, np.array([1, 0]))),
        ("Axes1/Frame/Line0/Point1", "Figure/Line0/Point1"),
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_bottom, np.array([0, -1]))),
        ("Axes1/Frame/Line1/Point0", "Figure/Line1/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_top, np.array([0, 1]))),
        ("Axes1/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )

    # Constrain 'Axes1' x/y axis bboxes to be rectangles
    layout.add_constraint(BOX, ("Axes1/XAxis",))
    layout.add_constraint(BOX, ("Axes1/YAxis",))

    # Make the x/y axes align the with frame

    layout.add_constraint(COLLINEAR, ("Axes1/XAxis/Line1", "Axes1/Frame/Line1"))
    layout.add_constraint(COLLINEAR, ("Axes1/XAxis/Line3", "Axes1/Frame/Line3"))

    layout.add_constraint(COLLINEAR, ("Axes1/YAxis/Line0", "Axes1/Frame/Line0"))
    layout.add_constraint(COLLINEAR, ("Axes1/YAxis/Line2", "Axes1/Frame/Line2"))

    # Pin the x/y axis to the frame sie
    layout.add_constraint(COLLINEAR, ("Axes1/XAxis/Line2", "Axes1/Frame/Line0"))
    layout.add_constraint(COLLINEAR, ("Axes1/YAxis/Line1", "Axes1/Frame/Line3"))

    # Set temporary widths/heights for x/y axis
    dim_labels = ("Height", "Width")
    for axis_key, line_label, dim_label in zip(
        ("X", "Y"), ("Line1", "Line0"), dim_labels
    ):
        layout.add_constraint(
            geo.Length.from_std((0.0,)),
            (f"Axes1/{axis_key}Axis/{line_label}",),
            f"Axes1.{axis_key}Axis.{dim_label}",
        )

    # Align x/y axis labels with axis bboxes
    layout.add_constraint(COINCIDENT, ("Axes1/XAxis/Line0/Point0", "Axes1/XAxisLabel"))

    layout.add_constraint(COINCIDENT, ("Axes1/YAxis/Line0/Point0", "Axes1/YAxisLabel"))

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    axs["Axes1"].xaxis.set_label_text("My x label", ha="left")
    axs["Axes1"].yaxis.set_label_text("My y label", ha="left")

    ax = axs["Axes1"]

    lay.update_layout_constraints(layout.root_constraint, axs)
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())
    lplt.update_subplots(prim_tree_n, "Figure", fig, axs)

    fig.savefig("out/complete_axes.png")
