"""
Test `solver`
"""

import pytest

import time
from pprint import pprint

import numpy as np

from mpllayout import geometry as geo, layout as lay, solver, containers as cn


class TestPrimitiveTree:

    @pytest.fixture()
    def layout(self):
        layout = lay.Layout()

        verts = np.array([[0.1, 0.2], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]])

        layout.add_prim(
            geo.Quadrilateral(children=[geo.Point(vert) for vert in verts]),
            "MyFavouriteBox",
        )
        layout.add_constraint(geo.Box(), ("MyFavouriteBox",), ())
        layout.add_constraint(
            geo.Fix(), ("MyFavouriteBox/Line0/Point0",), (np.array([0, 0]),)
        )

        layout.add_constraint(geo.Length(), ("MyFavouriteBox/Line0",), (5.0,))
        layout.add_constraint(geo.Length(), ("MyFavouriteBox/Line1",), (5.1,))
        return layout

    @pytest.fixture(params=[(5, 6)])
    def axes_shape(self, request):
        return request.param

    @pytest.fixture()
    def layout_grid(self, axes_shape):
        layout = lay.Layout()
        ## Create an origin point
        layout.add_prim(geo.Point(), "Origin")
        layout.add_constraint(geo.Fix(), ("Origin",), (np.array([0, 0]),))

        ## Create the figure box
        layout.add_prim(geo.Quadrilateral(), "Figure",)
        layout.add_constraint(geo.Box(), ("Figure",), ())

        ## Constrain the figure size and position
        fig_width, fig_height = 6, 3
        layout.add_constraint(
            geo.Length(), ("Figure/Line0",), (fig_width,)
        )
        layout.add_constraint(
            geo.Coincident(), ("Figure/Line0/Point0", "Origin"), ()
        )

        ## Create the axes boxes
        # axes_shape = (3, 4)
        num_row, num_col = axes_shape
        num_axes = int(np.prod(axes_shape))
        for n in range(num_axes):
            layout.add_prim(geo.Quadrilateral(), f"Axes{n}")
            layout.add_constraint(geo.Box(), (f"Axes{n}",), ())

        ## Constrain the axes in a grid
        num_row, num_col = axes_shape
        grid_param = {
            "col_margins": (num_col - 1) * [1 / 16],
            "row_margins": (num_row - 1) * [1 / 16],
            "col_widths": (num_col - 1) * [1],
            "row_heights": (num_row - 1) * [1],
        }
        layout.add_constraint(
            geo.Grid(axes_shape),
            tuple(f"Axes{n}" for n in range(num_axes)),
            grid_param
        )

        # Constrain the first axis aspect ratio
        layout.add_constraint(
            geo.RelativeLength(), ("Axes0/Line0", "Axes0/Line1"), (2,)
        )

        # Constrain top/bottom margins
        margin_top = 1.1
        margin_bottom = 0.5
        layout.add_constraint(
            geo.DirectedDistance(),
            ("Axes0/Line1/Point1", "Figure/Line1/Point1"),
            (margin_top, np.array([0, 1]))
        )
        layout.add_constraint(
            geo.DirectedDistance(),
            (f"Axes{num_axes-1}/Line1/Point0", "Figure/Line1/Point0"),
            (margin_bottom, np.array([0, -1]))
        )

        # Constrain left/right margins
        margin_left = 0.2
        margin_right = 0.3
        layout.add_constraint(
            geo.DirectedDistance(),
            ("Axes0/Line0/Point0", "Figure/Line0/Point0"),
            (margin_left, np.array([-1, 0]))
        )
        layout.add_constraint(
            geo.DirectedDistance(),
            (f"Axes{num_col-1}/Line1/Point1", "Figure/Line1/Point1"),
            (margin_right, np.array([1, 0]))
        )
        return layout

    def test_assem_constraint_residual(self, layout_grid: lay.Layout):
        layout = layout_grid

        prim_graph, prims = lay.build_prim_graph(layout.root_prim)
        prim_params = [prim.value for prim in prims]

        prim_tree = layout.root_prim
        # prim_graph = prim_tree.prim_graph()
        flat_constraints = layout.flat_constraints()

        # Plain call
        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                prim_tree, prim_graph, prim_params, *flat_constraints
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` individual constraint functions
        import jax

        constraints = flat_constraints[0]
        constraints_jit = [jax.jit(constraint) for constraint in constraints]
        flat_constraints_jit = (constraints_jit,) + flat_constraints[1:]
        solver.assem_constraint_residual(
            prim_tree, prim_graph, prim_params, *flat_constraints_jit
        )

        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                prim_tree, prim_graph, prim_params, *flat_constraints_jit
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` the overall function

        @jax.jit
        def assem_constraint_residual(prim_params):
            return solver.assem_constraint_residual(
                prim_tree, prim_graph, prim_params, *flat_constraints
            )

        assem_constraint_residual(prim_params)
        t0 = time.time()
        for i in range(50):
            assem_constraint_residual(prim_params)
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

    def test_solve(self, layout: lay.Layout):
        prim_tree_n, solve_info = solver.solve(layout)

        prim_keys_to_value = {
            key: prim.value for key, prim in cn.iter_flat("", prim_tree_n)
        }
        pprint(prim_keys_to_value)
        pprint(solve_info)
