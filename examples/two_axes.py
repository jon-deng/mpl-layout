"""
Create a two axes figure
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


def plot_layout(layout: lay.Layout, fig_path: str):
    prim_tree_n, info = solver.solve(
        layout.prim_tree,
        layout.constraints,
        layout.constraint_graph_int,
        max_iter=40,
        rel_tol=1e-9,
    )
    root_prim_labels = [label for label in prim_tree_n.keys() if "." not in label]
    root_prims = [prim_tree_n[label] for label in root_prim_labels]

    fig, ax = plt.subplots(1, 1)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.set_xticks(np.arange(-1, 11, 1))
    ax.set_yticks(np.arange(-1, 11, 1))
    ax.set_aspect(1)

    ax.set_xlabel("x [in]")
    ax.set_ylabel("y [in]")
    ui.plot_prims(ax, prim_tree_n)

    fig.savefig(fig_path)


if __name__ == "__main__":
    # Create a layout object to handle the collection of primitives, and linking
    # of constraints with those primitives
    layout = lay.Layout()

    ## Create an origin point

    layout.add_prim(geo.Point([0, 0]), "Origin")
    # Constrain the origin to be at (0, 0)
    layout.add_constraint(geo.PointLocation(np.array([0, 0])), ("Origin",))

    plot_layout(layout, "out/2Axes--0.png")

    ## Create a box to represent the figure

    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    # Create the box with an initial size of 5 by 5 and call it 'Figure'
    layout.add_prim(
        geo.Quadrilateral(prims=[geo.Point(vert) for vert in verts]), "Figure"
    )
    layout.add_constraint(geo.Box(), ("Figure",))

    plot_layout(layout, "out/2Axes--1.png")

    ## Create another box to represent the left axes

    verts = [[1, 1], [4, 1], [4, 4], [1, 4]]
    # Call the box 'Axes1'
    layout.add_prim(
        geo.Quadrilateral(prims=[geo.Point(vert) for vert in verts]), "Axes1"
    )
    layout.add_constraint(geo.Box(), ("Axes1",))

    plot_layout(layout, "out/2Axes--2.png")

    ## Create another box to represent the right axes

    verts = [[2, 2], [5, 2], [5, 5], [2, 5]]
    # Call the box 'Axes2'
    layout.add_prim(
        geo.Quadrilateral(prims=[geo.Point(vert) for vert in verts]), "Axes2"
    )
    layout.add_constraint(geo.Box(), ("Axes2",))

    plot_layout(layout, "out/2Axes--3.png")

    ## Constrain the figure size
    fig_width, fig_height = 6, 3

    # Constrain the bottom edge of the figure box to have length `fig_width`
    layout.add_constraint(
        geo.Length(fig_width),
        ("Figure/Line0",),
    )

    # Constrain the right edge of the figure box to have length `fig_height`
    layout.add_constraint(geo.Length(fig_height), ("Figure/Line1",))

    # Constrain the bottom corner point of the figure box
    # to be coincident with the origin
    layout.add_constraint(geo.CoincidentPoints(), ("Figure/Line0/Point0", "Origin"))

    plot_layout(layout, "out/2Axes--4.png")

    ## Constrain the left margin to `Axes1`

    margin_left = 0.5
    layout.add_constraint(
        geo.DirectedDistance(margin_left, np.array([-1, 0])),
        ("Axes1/Line0/Point0", "Figure/Line0/Point0"),
    )

    plot_layout(layout, "out/2Axes--5.png")

    ## Constrain the right margin to `Axes2`

    margin_right = 0.5
    layout.add_constraint(
        geo.DirectedDistance(margin_right, np.array([1, 0])),
        ("Axes2/Line0/Point1", "Figure/Line0/Point1"),
    )

    plot_layout(layout, "out/2Axes--6.png")

    ## Constrain the width of 'Axes1' by setting the length of the bottom edge
    width = 2
    layout.add_constraint(geo.Length(width), ("Axes1/Line0",))

    plot_layout(layout, "out/2Axes--7.png")

    ## Constrain the gap between the left and right axes ('Axes1' and `Axes2`)
    margin_inter = 0.5
    layout.add_constraint(
        geo.DirectedDistance(margin_inter, np.array([1, 0])),
        ("Axes1/Line0/Point1", "Axes2/Line0/Point0"),
    )

    plot_layout(layout, "out/2Axes--8.png")

    ## Constrain the top/bottom margins on the left axes ('Axes1')
    margin_top = 1.0
    margin_bottom = 0.5
    layout.add_constraint(
        geo.DirectedDistance(margin_bottom, np.array([0, -1])),
        ("Axes1/Line0/Point0", "Figure/Line0/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance(margin_top, np.array([0, 1])),
        ("Axes1/Line1/Point1", "Figure/Line1/Point1"),
    )

    plot_layout(layout, "out/2Axes--9.png")

    ## Make the top/bottom edges of the right axes ('Axes2') line up with the
    # top/bottom edges of the left axes ('Axes1')
    layout.add_constraint(geo.Collinear(), ("Axes1/Line0", "Axes2/Line0"))
    layout.add_constraint(geo.Collinear(), ("Axes1/Line2", "Axes2/Line2"))

    plot_layout(layout, "out/2Axes--10.png")

    ## Solve for the constrained positions of the primitives
    prims, info = solver.solve(
        layout.prim_tree,
        layout.constraints,
        layout.constraint_graph_int,
        max_iter=40,
        rel_tol=1e-9,
    )
    print("Figure:", prims["Figure"])
    print("Axes1:", prims["Axes1"])
    print("Axes2:", prims["Axes2"])

    ## Create a figure and axes from the constrained primitives
    fig, axs = lplt.subplots(prims)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, 4 * x)
    axs["Axes2"].plot(x, x**2)

    fig.savefig("out/two_axes.png")

    ## Plot the constrained primitives in a figure
    root_prim_labels = [label for label in prims.keys() if "." not in label]
    root_prims = [prims[label] for label in root_prim_labels]
    print(root_prim_labels)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0, fig_width + 1)
    ax.set_ylim(0, fig_height + 1)
    ax.set_aspect(1)
    ui.plot_prims(ax, prims)
    fig.savefig("out/two_axes_layout.png")
