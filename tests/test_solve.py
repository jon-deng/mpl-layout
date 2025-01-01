"""
Test `solver`
"""

import pytest

import time
from pprint import pprint

import numpy as np

from mpllayout import primitives as pr
from mpllayout import constraints as co
from mpllayout import layout as lay
from mpllayout import containers as cn
from mpllayout import solver


class TestPrimitiveTree:

    @pytest.fixture()
    def layout(self):
        layout = lay.Layout()

        verts = np.array([[0.1, 0.2], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]])

        layout.add_prim(
            pr.Quadrilateral(children=[pr.Point(vert) for vert in verts]),
            "MyFavouriteBox",
        )
        layout.add_constraint(co.Box(), ("MyFavouriteBox",), ())
        layout.add_constraint(
            co.Fix(), ("MyFavouriteBox/Line0/Point0",), (np.array([0, 0]),)
        )

        layout.add_constraint(co.Length(), ("MyFavouriteBox/Line0",), (5.0,))
        layout.add_constraint(co.Length(), ("MyFavouriteBox/Line1",), (5.1,))
        return layout

    @pytest.fixture(params=[(5, 6)])
    def axes_shape(self, request):
        return request.param

    @pytest.fixture()
    def layout_grid(self, axes_shape):
        layout = lay.Layout()
        ## Create an origin point
        layout.add_prim(pr.Point(), "Origin")
        layout.add_constraint(co.Fix(), ("Origin",), (np.array([0, 0]),))

        ## Create the figure box
        layout.add_prim(pr.Quadrilateral(), "Figure",)
        layout.add_constraint(co.Box(), ("Figure",), ())

        ## Constrain the figure size and position
        fig_width, fig_height = 6, 3
        layout.add_constraint(
            co.Length(), ("Figure/Line0",), (fig_width,)
        )
        layout.add_constraint(
            co.Coincident(), ("Figure/Line0/Point0", "Origin"), ()
        )

        ## Create the axes boxes
        # axes_shape = (3, 4)
        num_row, num_col = axes_shape
        num_axes = int(np.prod(axes_shape))
        for n in range(num_axes):
            layout.add_prim(pr.Quadrilateral(), f"Axes{n}")
            layout.add_constraint(co.Box(), (f"Axes{n}",), ())

        ## Constrain the axes in a grid
        num_row, num_col = axes_shape
        grid_param = (
            (num_col - 1) * [1],
            (num_row - 1) * [1],
            (num_col - 1) * [1 / 16],
            (num_row - 1) * [1 / 16],
        )
        layout.add_constraint(
            co.Grid(axes_shape),
            tuple(f"Axes{n}" for n in range(num_axes)),
            grid_param
        )

        # Constrain the first axis aspect ratio
        layout.add_constraint(
            co.RelativeLength(), ("Axes0/Line0", "Axes0/Line1"), (2,)
        )

        # Constrain top/bottom margins
        margin_top = 1.1
        margin_bottom = 0.5
        layout.add_constraint(
            co.DirectedDistance(),
            ("Axes0/Line1/Point1", "Figure/Line1/Point1"),
            (np.array([0, 1]), margin_top)
        )
        layout.add_constraint(
            co.DirectedDistance(),
            (f"Axes{num_axes-1}/Line1/Point0", "Figure/Line1/Point0"),
            (np.array([0, -1]), margin_bottom)
        )

        # Constrain left/right margins
        margin_left = 0.2
        margin_right = 0.3
        layout.add_constraint(
            co.DirectedDistance(),
            ("Axes0/Line0/Point0", "Figure/Line0/Point0"),
            (np.array([-1, 0]), margin_left)
        )
        layout.add_constraint(
            co.DirectedDistance(),
            (f"Axes{num_col-1}/Line1/Point1", "Figure/Line1/Point1"),
            (np.array([1, 0]), margin_right)
        )
        return layout

    def test_assem_constraint_residual(self, layout_grid: lay.Layout):
        layout = layout_grid

        root_prim = layout.root_prim
        flat_constraints = layout.flat_constraints()

        # Plain call
        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                root_prim, *flat_constraints
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` individual constraint functions
        import jax

        constraints = flat_constraints[0]
        constraints_jit = [jax.jit(constraint) for constraint in constraints]
        flat_constraints_jit = (constraints_jit,) + flat_constraints[1:]
        solver.assem_constraint_residual(
            root_prim, *flat_constraints_jit
        )

        t0 = time.time()
        for i in range(50):
            solver.assem_constraint_residual(
                root_prim, *flat_constraints_jit
            )
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

        # `jax.jit` the overall function

        @jax.jit
        def assem_constraint_residual(root_prim):
            return solver.assem_constraint_residual(
                root_prim, *flat_constraints
            )

        assem_constraint_residual(root_prim)
        t0 = time.time()
        for i in range(50):
            assem_constraint_residual(root_prim)
        t1 = time.time()
        print(f"Duration {t1-t0:.2e} s")

    @pytest.fixture(
        params=('newton', 'minimize')
    )
    def method(self, request):
        return request.param

    def test_solve(self, layout: lay.Layout, method: str):
        t0 = time.time()
        prim_tree_n, solve_info = solver.solve(
            layout, method=method, max_iter=100
        )
        t1 = time.time()
        print(f"Solve took {t1-t0:.2e} s")

        prim_keys_to_value = {
            key: prim.value for key, prim in cn.iter_flat("", prim_tree_n)
        }
        pprint(prim_keys_to_value)
        pprint(solve_info)
