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

def plot_layout(root_prim, fig_path: str):

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # ax.set_xlim(-1, 10)
    # ax.set_ylim(-1, 10)
    # ax.set_xticks(np.arange(-1, 11, 1))
    # ax.set_yticks(np.arange(-1, 11, 1))
    ax.set_aspect(1)
    ax.grid()

    # for axis in (ax.xaxis, ax.yaxis)

    ax.set_xlabel("x [in]")
    ax.set_ylabel("y [in]")
    ui.plot_prims(ax, root_prim)

    fig.savefig(fig_path)

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
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Figure/Line0/Point0", "Figure/Line0/Point1"),
        (fig_width, np.array([1, 0]))
    )
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Figure/Line1/Point0", "Figure/Line1/Point1"),
        (fig_height, np.array([0, 1]))
    )

    ## Constrain 'Axes1' elements
    # Constrain left/right margins
    margin_left = 1
    margin_right = 1.0
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Axes1/Frame/Line0/Point0", "Figure/Line0/Point0"),
        (margin_left, np.array([-1, 0]))
    )
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Axes1/Frame/Line0/Point1", "Figure/Line0/Point1"),
        (margin_right, np.array([1, 0]))
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 1
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Axes1/Frame/Line1/Point0", "Figure/Line1/Point0"),
        (margin_bottom, np.array([0, -1]))
    )
    layout.add_constraint(
        geo.DirectedDistance(),
        ("Axes1/Frame/Line1/Point1", "Figure/Line1/Point1"),
        (margin_top, np.array([0, 1]))
    )

    # Constrain 'Axes1' x/y axis bboxes to be rectangles
    layout.add_constraint(BOX, ("Axes1/XAxis",), ())
    layout.add_constraint(BOX, ("Axes1/YAxis",), ())

    # Make the x/y axes align the with frame
    layout.add_constraint(
        geo.CoincidentLines(), ("Axes1/XAxis/Line2", "Axes1/Frame/Line0"), (True,)
    )
    layout.add_constraint(
        geo.CoincidentLines(), ("Axes1/YAxis/Line1", "Axes1/Frame/Line3"), (True,)
    )

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
    plot_layout(prim_tree_n, 'out/complete_axes_layout_1.png')

    lay.update_layout_constraints(layout.root_constraint, layout.root_constraint_param, axs)
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())
    lplt.update_subplots(prim_tree_n, "Figure", fig, axs)
    print(info)


    fig.savefig("out/complete_axes_2.png")

    plot_layout(prim_tree_n, 'out/complete_axes_layout_2.png')

    # breakpoint()