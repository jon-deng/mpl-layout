"""
Create a one axes figure
"""

import numpy as np

from mpllayout import solver, geometry as geo, layout as lay, matplotlibutils as lplt

if __name__ == "__main__":
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point([0, 0]), "Origin")
    layout.add_constraint(geo.Fix(), ("Origin",), (np.array([0, 0]),))

    ## Create the figure box
    layout.add_prim(geo.Quadrilateral(), "Figure")
    layout.add_constraint(geo.Box(), ("Figure",), ())

    ## Create the axes box
    box = geo.Axes()
    layout.add_prim(box, "Axes1")
    layout.add_constraint(geo.Box(), ("Axes1/Frame",), ())

    ## Constrain the figure size
    fig_width, fig_height = 6, 3
    layout.add_constraint(
        geo.XLength(), ("Figure/Line0",), (fig_width,)
    )
    layout.add_constraint(
        geo.YLength(), ("Figure/Line1",), (fig_height,)
    )

    layout.add_constraint(geo.Coincident(), ("Figure/Line0/Point0", "Origin"), ())

    ## Constrain 'Axes1' margins
    # Constrain left/right margins
    margin_left = 1.1
    margin_right = 1.1
    layout.add_constraint(
        geo.MidpointXDistance(), ("Figure/Line3", "Axes1/Frame/Line3", ), (margin_left,)
    )
    layout.add_constraint(
        geo.MidpointXDistance(), ("Axes1/Frame/Line1", "Figure/Line1"), (margin_right,)
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.MidpointYDistance(), ("Figure/Line0", "Axes1/Frame/Line0"), (margin_bottom,)
    )
    layout.add_constraint(
        geo.MidpointYDistance(), ("Axes1/Frame/Line2", "Figure/Line2"), (margin_top,)
    )

    ## Solve the constraints and form the figure/axes layout
    prim_tree_n, info = solver.solve(layout.root_prim, *layout.flat_constraints())

    print("Figure:", prim_tree_n["Figure"])
    print("Axes1:", prim_tree_n["Axes1"])

    fig, axs = lplt.subplots(prim_tree_n)

    x = np.linspace(0, 1)
    axs["Axes1"].plot(x, x**2)

    fig.savefig("out/one_axes.png")
