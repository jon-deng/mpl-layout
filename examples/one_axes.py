"""
Create a one axes figure
"""

import numpy as np

from mpllayout import solver, geometry as geo, layout as lay, matplotlibutils as lplt

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point.from_std([0, 0]), "Origin")
    layout.add_constraint(geo.PointLocation.from_std((np.array([0, 0]),)), ("Origin",))

    ## Create the figure box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    box = geo.Quadrilateral.from_std(
        children=[geo.Point.from_std(vert_coords) for vert_coords in verts]
    )
    layout.add_prim(box, "Figure")
    layout.add_constraint(geo.Box.from_std({}), ("Figure",))

    ## Create the axes box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    box = geo.Axes.from_std(
        children=[
            geo.Quadrilateral.from_std(
                children=[geo.Point.from_std(vert_coords) for vert_coords in verts]
            )
        ]
    )
    layout.add_prim(box, "Axes1")
    layout.add_constraint(geo.Box.from_std({}), ("Axes1/Frame",))

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
        geo.CoincidentPoints.from_std({}), ("Figure/Line0/Point0", "Origin")
    )

    ## Constrain 'Axes1' margins
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

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(
        layout.root_prim, layout.constraints, layout.constraint_graph
    )

    print("Figure:", prim_tree_n["Figure"])
    print("Axes1:", prim_tree_n["Axes1"])

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    fig.savefig("out/one_axes.png")
