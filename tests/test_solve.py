"""
Test `solver`
"""

import pytest

from pprint import pprint

import numpy as np

from mpllayout import geometry as geo, layout as lay, solver


class TestPrimitiveTree:

    @pytest.fixture()
    def layout(self):
        layout = lay.Layout()

        layout.add_prim(geo.Quadrilateral(), "MyFavouriteBox")
        layout.add_constraint(geo.Box(), ("MyFavouriteBox",))
        layout.add_constraint(
            geo.PointLocation(np.array([0, 0])), ("MyFavouriteBox/Line0/Point0",)
        )
        return layout

    def test_solve_linear(self, layout: lay.Layout):
        prim_tree_n, solve_info = solver.solve_linear(
            layout.prim_tree, layout.constraints, layout.constraint_graph_int
        )

        pprint(solve_info)
