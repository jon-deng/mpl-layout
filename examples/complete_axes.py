"""
Eample layout of a figure with a single axes

This example illustrates a single axes figure that includes specifications for
the x and y axis.
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
    # Create the layout to store constraints and primitives
    layout = lay.Layout()

    ## Create the Figure and Axes primitives
    layout.add_prim(geo.Quadrilateral(), "Figure")

    # The `Axes` primtive contains quadrilaterals to specify x and y axis
    # locations as well as axis labels
    layout.add_prim(geo.Axes(), "Axes")

    ## Make all quadrilaterals rectangular/boxes
    # NOTE: This step is needed because `Quadrilateral` by default don't have
    # to be rectangles
    layout.add_constraint(geo.Box(), ("Figure",), ())
    layout.add_constraint(geo.Box(), ("Axes/Frame",), ())
    layout.add_constraint(geo.Box(), ("Axes/XAxis",), ())
    layout.add_constraint(geo.Box(), ("Axes/YAxis",), ())

    ## Constrain the figure size + position
    # Fix the figure bottom left to the origin
    layout.add_constraint(geo.Fix(), ("Figure/Line0/Point0",), (np.array([0, 0]),))

    # Figure the figure width and height
    fig_width, fig_height = 6, 3
    layout.add_constraint(geo.XLength(), ("Figure/Line0",), (fig_width,))
    layout.add_constraint(geo.YLength(), ("Figure/Line1",), (fig_height,))

    ## Constrain margins around the axes to the figure
    # Constrain left/right margins
    margin_left = 0.1
    margin_right = 1/4

    layout.add_constraint(
        geo.InnerMargin(side='left'), ("Axes/Frame", "Figure"), (margin_left,)
    )
    layout.add_constraint(
        geo.InnerMargin(side='right'), ("Axes/YAxis", "Figure"), (margin_right,)
    )

    # Constrain top/bottom margins
    margin_top = 1/4
    margin_bottom = 0.1
    layout.add_constraint(
        geo.InnerMargin(side='bottom'), ("Axes/Frame", "Figure"), (margin_bottom,)
    )
    layout.add_constraint(
        geo.InnerMargin(side='top'), ("Axes/XAxis", "Figure"), (margin_top,)
    )

    ## Position the x axis on top and the y axis on the bottom
    # When creating axes from the primitives, `lplt.subplots` will detect axis
    # positions and set axis properties to reflect them.
    layout.add_constraint(geo.PositionXAxis(bottom=False, top=True), ("Axes", ), ())
    layout.add_constraint(geo.PositionYAxis(left=False, right=True), ("Axes", ), ())

    # Link x/y axis width/height to axis sizes in matplotlib.
    # Axis sizes change depending on the size of their tick labels so the
    # axis width/height must be linked to matplotlib and updated from plot
    # elements.
    layout.add_constraint(
        geo.XAxisHeight(), ("Axes/XAxis",), (None,), 'Axes.XAxisHeight'
    )
    layout.add_constraint(
        geo.YAxisWidth(), ("Axes/YAxis",), (None,), 'Axes.YAxisWidth'
    )

    ## Position the x/y axis label text anchors
    # When creating axes from the primitives, `lplt.subplots` will detect these
    # and set their locations
    on_line = geo.RelativePointOnLineDistance()
    to_line = geo.PointToLineDistance()

    ## Pad x/y axis label from the axis bbox
    pad = 1/16
    layout.add_constraint(to_line, ("Axes/XAxisLabel", "Axes/XAxis/Line2"), {"distance": pad, "reverse": True})
    layout.add_constraint(to_line, ("Axes/YAxisLabel", "Axes/YAxis/Line1"), {"distance": pad, "reverse": True})

    ## Center the axis labels halfway along the axes width/height
    layout.add_constraint(geo.PositionXAxisLabel(), ("Axes",), {"distance": 0.5})
    layout.add_constraint(geo.PositionYAxisLabel(), ("Axes",), {"distance": 0.5})

    # This is what `Position...AxisLabel` does under the hood
    # layout.add_constraint(on_line, ("Axes/XAxisLabel", "Axes/XAxis/Line2"), {"distance": 0.5, "reverse": True})
    # layout.add_constraint(on_line, ("Axes/YAxisLabel", "Axes/YAxis/Line1"), {"distance": 0.5, "reverse": True})

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, solve_info = solver.solve(
        layout.root_prim, *layout.flat_constraints()
    )
    print("First layout solve")
    print(f"Absolute errors: {solve_info['abs_errs']}")
    print(f"Relative errors: {solve_info['rel_errs']}")

    ## Plot into the generated figure and axes
    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes"].plot(x, x**2)

    axs["Axes"].xaxis.set_label_text("My x label", ha="center", va="bottom")
    axs["Axes"].yaxis.set_label_text("My y label", ha="center", va="bottom", rotation=-90)

    ax = axs["Axes"]

    # This figure illustrates the layout before the x/y axis width/height is
    # updated
    fig.savefig("complete_axes_before_axis_update.png")
    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig("complete_axes_layout_before_axis_update.png")

    # Using the generated axes and x/y axis contents, the layout constraints
    # can be updated with those matplotlib elements
    lay.update_layout_constraints(layout.root_constraint, layout.root_constraint_param, axs)
    prim_tree_n, solve_info = solver.solve(
        layout.root_prim, *layout.flat_constraints()
    )
    print("\nUpdated layout solve")
    print(f"Absolute errors: {solve_info['abs_errs']}")
    print(f"Relative errors: {solve_info['rel_errs']}")

    # This updates the figure and axes using the updated layout
    lplt.update_subplots(prim_tree_n, "Figure", fig, axs)

    # This figure illustrates the layout after the x/y axis width/height is
    # updated
    fig.savefig("complete_axes_after_update.png")
    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig("complete_axes_layout_after_update.png")
