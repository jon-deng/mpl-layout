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

        verts = [[0.1, 0.2], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]]

        layout.add_prim(geo.Quadrilateral(prims=[geo.Point(vert) for vert in verts]), "MyFavouriteBox")
        layout.add_constraint(geo.Box(), ("MyFavouriteBox",))
        layout.add_constraint(
            geo.PointLocation(np.array([0, 0])), ("MyFavouriteBox/Line0/Point0",)
        )

        layout.add_constraint(geo.Length(5.0), ("MyFavouriteBox/Line0",))
        layout.add_constraint(geo.Length(5.5), ("MyFavouriteBox/Line1",))
        return layout

    def test_solve(self, layout: lay.Layout):
        prim_tree_n, solve_info = solver.solve(
            layout.prim_tree, layout.constraints, layout.constraint_graph_int
        )
        pprint(prim_tree_n.keys(flat=True))
        pprint(solve_info)
