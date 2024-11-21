"""
Profile runtimes of key functions
"""

import typing as tp
from typing import Optional

import cProfile
import pstats

import numpy as np

from mpllayout import layout as lay
from mpllayout import primitives as pr
from mpllayout import constraints as co
from mpllayout import containers as cn
from mpllayout import solver


def gen_layout(axes_shape: Optional[tuple[int, ...]] = (3, 3)) -> lay.Layout:
    layout = lay.Layout()

    ## Create an origin point
    layout.add_prim(pr.Point([0, 0]), "Origin")
    layout.add_constraint(co.Fix(), ("Origin",), {'location': np.array([0, 0])})

    ## Create the figure box
    layout.add_prim(pr.Quadrilateral(), "Figure")
    layout.add_constraint(co.Box(), ("Figure",), {})

    ## Constrain the figure size and position
    fig_width, fig_height = 6, 3
    layout.add_constraint(co.Length(), ("Figure/Line0",), {'length': fig_width})
    # layout.add_constraint(co.Length((fig_height,)), ("Figure/Line1",))
    layout.add_constraint(co.Coincident(), ("Figure/Line0/Point0", "Origin"), {})

    ## Create the axes boxes
    num_row, num_col = axes_shape
    num_axes = int(np.prod(axes_shape))
    verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
    for n in range(num_axes):
        layout.add_prim(pr.Axes(), f"Axes{n}")
        layout.add_constraint(co.Box(), (f"Axes{n}/Frame",), {})

    ## Constrain the axes in a grid
    num_row, num_col = axes_shape
    grid_kwargs = {
        'col_widths': (num_col - 1) * [1 / 16],
        'row_heights': (num_row - 1) * [1 / 16],
        'col_margins': (num_col - 1) * [1],
        'row_margins': (num_row - 1) * [1],
    }
    layout.add_constraint(
        co.Grid(axes_shape),
        tuple(f"Axes{n}/Frame" for n in range(num_axes)),
        grid_kwargs
    )

    # Constrain the first axis aspect ratio
    layout.add_constraint(
        co.RelativeLength(), ("Axes0/Frame/Line0", "Axes0/Frame/Line1"), {'length': 2}
    )

    # Constrain top/bottom margins
    margin_top = 1.1
    margin_bottom = 0.5
    layout.add_constraint(
        co.DirectedDistance(),
        ("Axes0/Frame/Line1/Point1", "Figure/Line1/Point1"),
        {'distance': margin_top, 'direction': np.array([0, 1])}
    )
    layout.add_constraint(
        co.DirectedDistance(),
        (f"Axes{num_axes-1}/Frame/Line1/Point0", "Figure/Line1/Point0"),
        {'distance': margin_bottom, 'direction': np.array([0, -1])}
    )

    # Constrain left/right margins
    margin_left = 0.2
    margin_right = 0.3
    layout.add_constraint(
        co.DirectedDistance(),
        ("Axes0/Frame/Line0/Point0", "Figure/Line0/Point0"),
        {'distance': margin_left, 'direction': np.array([-1, 0])}
    )
    layout.add_constraint(
        co.DirectedDistance(),
        (f"Axes{num_col-1}/Frame/Line1/Point1", "Figure/Line1/Point1"),
        {'distance': margin_right, 'direction': np.array([1, 0])}
    )
    return layout


if __name__ == "__main__":
    layout = gen_layout((12, 12))

    constraints, constraint_graph, constraint_params = layout.flat_constraints()

    solver.assem_constraint_residual(
        layout.root_prim, constraints, constraint_graph, constraint_params
    )
    stmt = "solver.assem_constraint_residual(layout.root_prim, constraints, constraint_graph, constraint_params)"
    cProfile.run(stmt, "profile_wo_jax.prof")
    stats = pstats.Stats("profile_wo_jax.prof")
    stats.sort_stats("cumtime").print_stats(20)

    import jax

    constraints_jit = [jax.jit(c) for c in constraints]
    solver.assem_constraint_residual(
        layout.root_prim,
        constraints_jit,
        constraint_graph,
        constraint_params
    )
    stmt = "solver.assem_constraint_residual(layout.root_prim, constraints_jit, constraint_graph, constraint_params)"
    cProfile.run(stmt, "profile_w_jax.prof")
    stats = pstats.Stats("profile_w_jax.prof")
    stats.sort_stats("cumtime").print_stats(20)
