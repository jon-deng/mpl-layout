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
            geo.Quadrilateral.from_std(
                children=[geo.Point.from_std(vert) for vert in verts]
            ),
            "MyFavouriteBox",
        )
        layout.add_constraint(geo.Box.from_std({}), ("MyFavouriteBox",))
        layout.add_constraint(
            geo.Fix.from_std((np.array([0, 0]),)),
            ("MyFavouriteBox/Line0/Point0",),
        )

        layout.add_constraint(geo.Length.from_std((5.0,)), ("MyFavouriteBox/Line0",))
        layout.add_constraint(geo.Length.from_std((5.1,)), ("MyFavouriteBox/Line1",))
        return layout

    @pytest.fixture(params=[(5, 6)])
    def axes_shape(self, request):
        return request.param

    @pytest.fixture()
    def layout_grid(self, axes_shape):
        layout = lay.Layout()
        ## Create an origin point
        layout.add_prim(geo.Point.from_std([0, 0]), "Origin")
        layout.add_constraint(
            geo.Fix.from_std((np.array([0, 0]),)), ("Origin",)
        )

        ## Create the figure box
        verts = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
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
        # axes_shape = (3, 4)
        num_row, num_col = axes_shape
        num_axes = int(np.prod(axes_shape))
        verts = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
        for n in range(num_axes):
            layout.add_prim(
                geo.Quadrilateral.from_std(
                    children=[geo.Point.from_std(vert_coords) for vert_coords in verts]
                ),
                f"Axes{n}",
            )
            layout.add_constraint(geo.Box.from_std({}), (f"Axes{n}",))

        ## Constrain the axes in a grid
        num_row, num_col = axes_shape
        layout.add_constraint(
            geo.Grid.from_std(
                {
                    "shape": axes_shape,
                    "col_margins": (num_col - 1) * [1 / 16],
                    "row_margins": (num_row - 1) * [1 / 16],
                    "col_widths": (num_col - 1) * [1],
                    "row_heights": (num_row - 1) * [1],
                }
            ),
            tuple(f"Axes{n}" for n in range(num_axes)),
        )

        # Constrain the first axis aspect ratio
        layout.add_constraint(
            geo.RelativeLength.from_std((2,)), ("Axes0/Line0", "Axes0/Line1")
        )

        # Constrain top/bottom margins
        margin_top = 1.1
        margin_bottom = 0.5
        layout.add_constraint(
            geo.DirectedDistance.from_std((margin_top, np.array([0, 1]))),
            ("Axes0/Line1/Point1", "Figure/Line1/Point1"),
        )
        layout.add_constraint(
            geo.DirectedDistance.from_std((margin_bottom, np.array([0, -1]))),
            (f"Axes{num_axes-1}/Line1/Point0", "Figure/Line1/Point0"),
        )

        # Constrain left/right margins
        margin_left = 0.2
        margin_right = 0.3
        layout.add_constraint(
            geo.DirectedDistance.from_std((margin_left, np.array([-1, 0]))),
            ("Axes0/Line0/Point0", "Figure/Line0/Point0"),
        )
        layout.add_constraint(
            geo.DirectedDistance.from_std((margin_right, np.array([1, 0]))),
            (f"Axes{num_col-1}/Line1/Point1", "Figure/Line1/Point1"),
        )
        return layout

    def test_assem_constraint_residual(self, layout_grid: lay.Layout):
        layout = layout_grid

        prim_graph, prims = lay.build_prim_graph(layout.root_prim)
        prim_params = [prim.value for prim in prims]

        prim_tree = layout.root_prim
        # prim_graph = prim_tree.prim_graph()
        constraints, constraint_graph_str = layout.flat_constraints()

        # Plain call

        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                prim_tree, prim_graph, prim_params, constraints, constraint_graph_str
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` individual constraint functions
        import jax

        constraints_jit = [jax.jit(constraint) for constraint in constraints]
        solver.assem_constraint_residual(
            prim_tree, prim_graph, prim_params, constraints_jit, constraint_graph_str
        )

        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                prim_tree,
                prim_graph,
                prim_params,
                constraints_jit,
                constraint_graph_str,
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` the overall function

        @jax.jit
        def assem_constraint_residual(prim_params):
            return solver.assem_constraint_residual(
                prim_tree, prim_graph, prim_params, constraints, constraint_graph_str
            )

        assem_constraint_residual(prim_params)
        t0 = time.time()
        for i in range(50):
            assem_constraint_residual(prim_params)
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

    def test_solve(self, layout: lay.Layout):
        prim_tree_n, solve_info = solver.solve(
            layout.root_prim, *layout.flat_constraints()
        )

        prim_keys_to_value = {
            key: prim.value for key, prim in cn.iter_flat("", prim_tree_n)
        }
        pprint(prim_keys_to_value)
        pprint(solve_info)
