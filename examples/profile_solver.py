"""
Profile runtimes of key functions
"""

import typing as tp
from typing import Optional

import cProfile
import pstats

import numpy as np

from mpllayout import layout as lay, geometry as geo, solver


def gen_layout(axes_shape: Optional[tuple[int, ...]] = (3, 3)) -> lay.Layout:
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(geo.Point([0, 0]), "Origin")
    layout.add_constraint(geo.Fix((np.array([0, 0]),)), ("Origin",))

    ## Create the figure box
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    layout.add_prim(
        geo.Quadrilateral(children=[geo.Point(vert_coords) for vert_coords in verts]),
        "Figure",
    )
    layout.add_constraint(geo.Box({}), ("Figure",))

    ## Constrain the figure size and position
    fig_width, fig_height = 6, 3
    layout.add_constraint(geo.Length((fig_width,)), ("Figure/Line0",))
    # layout.add_constraint(geo.Length((fig_height,)), ("Figure/Line1",))
    layout.add_constraint(geo.Coincident({}), ("Figure/Line0/Point0", "Origin"))

    ## Create the axes boxes
    num_row, num_col = axes_shape
    num_axes = int(np.prod(axes_shape))
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    for n in range(num_axes):
        layout.add_prim(
            geo.Axes(
                prims=[
                    geo.Quadrilateral(
                        children=[geo.Point(vert_coords) for vert_coords in verts]
                    )
                ]
            ),
            f"Axes{n}",
        )
        layout.add_constraint(geo.Box({}), (f"Axes{n}/Frame",))

    ## Constrain the axes in a grid
    num_row, num_col = axes_shape
    layout.add_constraint(
        geo.Grid(
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
        geo.RelativeLength((2,)), ("Axes0/Frame/Line0", "Axes0/Frame/Line1")
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        geo.DirectedDistance((margin_top, np.array([0, 1]))),
        ("Axes0/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )
    layout.add_constraint(
        geo.DirectedDistance((margin_bottom, np.array([0, -1]))),
        (f"Axes{num_axes-1}/Frame/Line1/Point0", "Figure/Line1/Point0"),
    )

    # Constrain left/right margins
    margin_left = 0.2
    margin_right = 0.3
    layout.add_constraint(
        geo.DirectedDistance((margin_left, np.array([-1, 0]))),
        ("Axes0/Frame/Line0/Point0", "Figure/Line0/Point0"),
    )
    layout.add_constraint(
        geo.DirectedDistance((margin_right, np.array([1, 0]))),
        (f"Axes{num_col-1}/Frame/Line1/Point1", "Figure/Line1/Point1"),
    )
    return layout


if __name__ == "__main__":
    layout = gen_layout((12, 12))

    prim_graph, prims = lay.build_prim_graph(layout.root_prim)
    prim_values = [prim.value for prim in prims]

    constraints, constraint_graph = layout.flat_constraints()

    solver.assem_constraint_residual(
        layout.root_prim, prim_graph, prim_values, constraints, constraint_graph
    )
    stmt = "solver.assem_constraint_residual(layout.root_prim, prim_graph, prim_values, constraints, constraint_graph)"
    cProfile.run(stmt, "profile_wo_jax.prof")
    stats = pstats.Stats("profile_wo_jax.prof")
    stats.sort_stats("cumtime").print_stats(20)

    import jax

    constraints_jit = [jax.jit(c) for c in constraints]
    solver.assem_constraint_residual(
        layout.root_prim,
        prim_graph,
        prim_values,
        constraints_jit,
        constraint_graph,
    )
    stmt = "solver.assem_constraint_residual(layout.root_prim, prim_graph, prim_values, constraints_jit, constraint_graph)"
    cProfile.run(stmt, "profile_w_jax.prof")
    stats = pstats.Stats("profile_w_jax.prof")
    stats.sort_stats("cumtime").print_stats(20)
