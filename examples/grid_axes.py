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


def plot_layout(layout: lay.Layout, fig_path: str):
    constraints, constraint_graph = layout.flat_constraints()
    prim_tree_n, info = solver.solve(
        layout.root_prim,
        constraints,
        constraint_graph,
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
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point.from_std([0, 0]), "Origin")
    layout.add_constraint(geo.Fix.from_std((np.array([0, 0]),)), ("Origin",))

    ## Create the figure box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    layout.add_prim(
        geo.Quadrilateral.from_std(
            children=[geo.Point.from_std(vert_coords) for vert_coords in verts]
        ),
        "Figure",
    )
    layout.add_constraint(geo.Box.from_std({}), ("Figure",))

    ## Constrain the figure size and position
    fig_width, fig_height = 6, 3
    layout.add_constraint(geo.Length.from_std((fig_width,)), ("Figure/Line0",))
    # layout.add_constraint(geo.Length.from_std((fig_height,)), ("Figure/Line1",))
    layout.add_constraint(
        geo.Coincident.from_std({}), ("Figure/Line0/Point0", "Origin")
    )

    ## Create the axes boxes
    axes_shape = (3, 4)
    num_row, num_col = axes_shape
    num_axes = int(np.prod(axes_shape))
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    for n in range(num_axes):
        layout.add_prim(
            geo.Axes.from_std(
                children=[
                    geo.Quadrilateral.from_std(
                        children=[
                            geo.Point.from_std(vert_coords) for vert_coords in verts
                        ]
                    )
                ]
            ),
            f"Axes{n}",
        )
        layout.add_constraint(geo.Box.from_std({}), (f"Axes{n}/Frame",))

    ## Constrain the axes in a grid
    num_row, num_col = axes_shape
    layout.add_constraint(
        geo.Grid.from_std(
            (
                axes_shape,
                (num_col - 1) * [1 / 16],
                (num_row - 1) * [1 / 16],
                (num_col - 1) * [1],
                (num_row - 1) * [1],
            )
        ),
        tuple(f"Axes{n}/Frame" for n in range(num_axes)),
    )

    # Constrain the first axis aspect ratio
    layout.add_constraint(
        geo.RelativeLength.from_std((2,)), ("Axes0/Frame/Line0", "Axes0/Frame/Line1")
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_top, np.array([0, 1]))),
        ("Axes0/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_bottom, np.array([0, -1]))),
        (f"Axes{num_axes-1}/Frame/Line1/Point0", "Figure/Line1/Point0"),
    )

    # Constrain left/right margins
    margin_left = 0.2
    margin_right = 0.3
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_left, np.array([-1, 0]))),
        ("Axes0/Frame/Line0/Point0", "Figure/Line0/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance.from_std((margin_right, np.array([1, 0]))),
        (f"Axes{num_col-1}/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(
        layout.root_prim, *layout.flat_constraints()
    )
    print(info)

    # plot_layout(layout, "grid_axes.png")

    # print('Figure:', prim_tree_n['Figure'])
    # print('Axes1:', prim_tree_n['Axes1'])

    fig, axs = lplt.subplots(prim_tree_n)

    # x = np.linspace(0, 1)
    # axs['Axes1'].plot(x, x**2)

    fig.savefig("grid_axes.png")
