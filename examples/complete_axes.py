"""
Create a one axes figure with x/y axis labels and stuff too
"""

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from mpllayout import (
    solver,
    geometry as geo,
    layout as lay,
    matplotlibutils as lplt,
    constraints as con,
    ui
)

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create constant constraints
    # (these have no parameters so are reused a fair bit)
    BOX = geo.Box()
    COLLINEAR = geo.Collinear()
    COINCIDENT = geo.Coincident()

    ## Create the Figure quad
    layout.add_prim(geo.Quadrilateral(), "Figure")
    layout.add_constraint(BOX, ("Figure",), ())
    layout.add_constraint(geo.Fix(), ("Figure/Line0/Point0",), (np.array([0, 0]),))

    ## Create the Axes quads
    layout.add_prim(geo.AxesXY(), "Axes1")
    layout.add_constraint(BOX, ("Axes1/Frame",), ())

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    layout.add_constraint(geo.XLength(), ("Figure/Line0",), (fig_width,))
    layout.add_constraint(geo.YLength(), ("Figure/Line1",), (fig_height,))

    ## Constrain 'Axes1' elements
    # Constrain left/right margins
    margin_left = 1
    margin_right = 1.0
    layout.add_constraint(
        geo.XDistanceMidpoints(), ("Figure/Line3", "Axes1/Frame/Line3"), (margin_left,)
    )
    layout.add_constraint(
        geo.XDistanceMidpoints(), ("Axes1/Frame/Line1", "Figure/Line1"), (margin_right,)
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 1
    layout.add_constraint(
        geo.YDistanceMidpoints(), ("Figure/Line0", "Axes1/Frame/Line0"), (margin_bottom,)
    )
    layout.add_constraint(
        geo.YDistanceMidpoints(), ("Axes1/Frame/Line2", "Figure/Line2"), (margin_top,)
    )

    # Constrain 'Axes1' x/y axis bboxes to be rectangles
    layout.add_constraint(BOX, ("Axes1/XAxis",), ())
    layout.add_constraint(BOX, ("Axes1/YAxis",), ())

    # Make the x/y axes align the with frame
    layout.add_constraint(geo.PositionXAxis(bottom=True), ("Axes1", ), ())
    layout.add_constraint(geo.PositionYAxis(left=True), ("Axes1", ), ())

    # Link x/y axis width/height to axis sizes in matplotlib
    layout.add_constraint(
        geo.XAxisHeight(), ("Axes1/XAxis",), (None,), 'Axes1.XAxisHeight'
    )
    layout.add_constraint(
        geo.YAxisWidth(), ("Axes1/YAxis",), (None,), 'Axes1.YAxisWidth'
    )

    # Align x/y axis labels with axis bboxes
    layout.add_constraint(COINCIDENT, ("Axes1/XAxis/Line0/Point0", "Axes1/XAxisLabel"), ())
    layout.add_constraint(COINCIDENT, ("Axes1/YAxis/Line0/Point0", "Axes1/YAxisLabel"), ())
    # layout.add_constraint(COINCIDENT, ("Axes1/Frame/Line0/Point0", "Axes1/XAxisLabel"), ())
    # layout.add_constraint(COINCIDENT, ("Axes1/Frame/Line0/Point0", "Axes1/YAxisLabel"), ())

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    axs["Axes1"].xaxis.set_label_text("My x label", ha="left")
    axs["Axes1"].yaxis.set_label_text("My y label", ha="left")

    ax = axs["Axes1"]

    fig.savefig("out/complete_axes_1.png")
    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig('out/complete_axes_layout_1.png')

    lay.update_layout_constraints(layout.root_constraint, layout.root_constraint_param, axs)
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())
    lplt.update_subplots(prim_tree_n, "Figure", fig, axs)
    print(info)


    fig.savefig("out/complete_axes_2.png")
    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig('out/complete_axes_layout_2.png')
