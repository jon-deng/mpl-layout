"""
Create a grid of axes with widths decreasing in the fibonnaci sequence
"""

import numpy as np

from mpllayout import solver
from mpllayout import primitives as pr
from mpllayout import constraints as co
from mpllayout import layout as lay
from mpllayout import matplotlibutils as lplt
from mpllayout import ui

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(pr.Point(), "Origin")
    layout.add_constraint(co.Fix(), ("Origin",), (np.array([0, 0]),))

    ## Create the figure box
    layout.add_prim(pr.Quadrilateral(), "Figure")
    layout.add_constraint(co.Box(), ("Figure",), ())

    ## Constrain the figure size and position
    fig_width, fig_height = 6, 3
    layout.add_constraint(co.Length(), ("Figure/Line0",), (fig_width,))
    # layout.add_constraint(co.Length((fig_height,)), ("Figure/Line1",))
    layout.add_constraint(co.Coincident(), ("Figure/Line0/Point0", "Origin"), ())

    ## Create the axes boxes
    axes_shape = (2, 1)
    num_row, num_col = axes_shape
    num_axes = int(np.prod(axes_shape))
    for n in range(num_axes):
        layout.add_prim(pr.Axes(), f"Axes{n}")
        layout.add_constraint(co.Box(), (f"Axes{n}/Frame",), ())

    ## Constrain the axes in a grid
    num_row, num_col = axes_shape
    grid_params = (
        (num_col - 1) * [1],
        (num_row - 1) * [1],
        (num_col - 1) * [0.5],
        (num_row - 1) * [0.5],
    )

    layout.add_constraint(
        co.Grid(axes_shape),
        tuple(f"Axes{n}/Frame" for n in range(num_axes)),
        grid_params
    )

    # Constrain the first axis aspect ratio
    layout.add_constraint(
        co.RelativeLength(), ("Axes0/Frame/Line0", "Axes0/Frame/Line1"), (1,)
    )

    # Constrain top/bottom margins
    margin_top = 1.0
    margin_bottom = 1.0
    layout.add_constraint(
        co.MidpointYDistance(), ("Axes0/Frame/Line2", "Figure/Line2"), (margin_top,)
    )
    layout.add_constraint(
        co.MidpointYDistance(), ("Figure/Line0", f"Axes{num_axes-1}/Frame/Line0"), (margin_bottom, )
    )

    # Constrain left/right margins
    margin_left = 1.0
    margin_right = 1.0
    layout.add_constraint(
        co.MidpointXDistance(),
        ("Figure/Line3", "Axes0/Frame/Line3", ), (margin_left,)
    )
    layout.add_constraint(
        co.MidpointXDistance(),
        (f"Axes{num_col-1}/Frame/Line1", "Figure/Line1"), (margin_right,)
    )

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(layout)
    print(info)

    _fig, _ = ui.figure_prims(prim_tree_n)
    _fig.savefig("grid_axes_layout.png", dpi=300)

    # print('Figure:', prim_tree_n['Figure'])
    # print('Axes1:', prim_tree_n['Axes1'])

    fig, axs = lplt.subplots(prim_tree_n)

    # x = np.linspace(0, 1)
    # axs['Axes1'].plot(x, x**2)

    fig.savefig("grid_axes.png")
