"""
Create a grid of axes with widths decreasing in the fibonnaci sequence
"""

import numpy as np

from matplotlib import pyplot as plt
from mpllayout import (
    solver,
    geometry as geo,
    layout as lay,
    matplotlibutils as lplt,
    ui,
)

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point(), "Origin")
    layout.add_constraint(geo.Fix(), ("Origin",), (np.array([0, 0]),))

    ## Create the figure box
    layout.add_prim(geo.Quadrilateral(), "Figure")
    layout.add_constraint(geo.Box(), ("Figure",), ())

    ## Constrain the figure size and position
    fig_width, fig_height = 6, 3
    layout.add_constraint(geo.Length(), ("Figure/Line0",), (fig_width,))
    # layout.add_constraint(geo.Length((fig_height,)), ("Figure/Line1",))
    layout.add_constraint(geo.Coincident(), ("Figure/Line0/Point0", "Origin"), ())

    ## Create the axes boxes
    axes_shape = (4, 4)
    num_row, num_col = axes_shape
    num_axes = int(np.prod(axes_shape))
    for n in range(num_axes):
        layout.add_prim(geo.Axes(), f"Axes{n}")
        layout.add_constraint(geo.Box(), (f"Axes{n}/Frame",), ())

    ## Constrain the axes in a grid
    num_row, num_col = axes_shape
    grid_params = {
        "col_widths": (num_col - 1) * [1],
        "row_heights": (num_row - 1) * [1],
        "col_margins": (num_col - 1) * [1 / 16],
        "row_margins": (num_row - 1) * [1 / 16],
    }

    layout.add_constraint(
        geo.Grid(axes_shape),
        tuple(f"Axes{n}/Frame" for n in range(num_axes)),
        grid_params
    )

    # Constrain the first axis aspect ratio
    layout.add_constraint(
        geo.RelativeLength(), ("Axes0/Frame/Line0", "Axes0/Frame/Line1"), (1,)
    )

    # Constrain top/bottom margins
    margin_top = 0.5
    margin_bottom = 0.5
    layout.add_constraint(
        geo.MidpointYDistance(), ("Axes0/Frame/Line2", "Figure/Line2"), (margin_top,)
    )
    layout.add_constraint(
        geo.MidpointYDistance(), ("Figure/Line0", f"Axes{num_axes-1}/Frame/Line0"), (margin_bottom, )
    )

    # Constrain left/right margins
    margin_left = 0.5
    margin_right = 0.5
    layout.add_constraint(
        geo.MidpointXDistance(),
        ("Figure/Line3", "Axes0/Frame/Line3", ), (margin_left,)
    )
    layout.add_constraint(
        geo.MidpointXDistance(),
        (f"Axes{num_col-1}/Frame/Line1", "Figure/Line1"), (margin_right,)
    )

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())
    print(info)

    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig("grid_axes_layout.png")

    # print('Figure:', prim_tree_n['Figure'])
    # print('Axes1:', prim_tree_n['Axes1'])

    fig, axs = lplt.subplots(prim_tree_n)

    # x = np.linspace(0, 1)
    # axs['Axes1'].plot(x, x**2)

    fig.savefig("grid_axes.png")
