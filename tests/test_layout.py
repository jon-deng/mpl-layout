"""
Test `layout`
"""

import pytest

from pprint import pprint

import numpy as np

from mpllayout import primitives as pr
from mpllayout import constraints as co
from mpllayout import layout as lat
from mpllayout import containers as cn


class TestPrimitiveTree:

    @pytest.fixture()
    def prim_node(self):
        return cn.Node(np.array([]), {})

    def test_set_prim(self, prim_node):
        prim_node.add_child("MyBox", pr.Quadrilateral())

        pprint(f"Keys:")
        pprint(prim_node["MyBox"].keys())

    def test_build_primtree(self, prim_node):
        point_a = pr.Point([0, 0])
        point_b = pr.Point([1, 1])
        prim_node.add_child("PointA", point_a)
        prim_node.add_child("LineA", pr.Line([], (point_a, point_b)))
        prim_node.add_child("MySpecialBox", pr.Quadrilateral())

        prim_graph, prim_values = pr.filter_unique_values_from_prim(prim_node)

        params = prim_values

        new_params = [np.random.rand(*param.shape) for param in params]
        new_prim_node = pr.build_prim_from_unique_values(cn.flatten('', prim_node), prim_graph, new_params)
        # breakpoint()

        # rng = np.random.default_rng()

        # new_tree = lat.build_tree(prim_tree, prim_graph, new_params, {})

        # print("Old primitive graph:")
        # pprint(prim_tree.prim_graph())

        # print("Old primitive list")
        # pprint(prim_tree.prims())

        # print("Old primitive keys")
        # pprint(prim_tree.keys(flat=True))

        # print("New parameters")
        # pprint(new_params)

        # print("New primitive graph:")
        # pprint(new_tree.prim_graph())

        # print("New primitive list")
        # pprint(new_tree.prims())

        # print("New primitive keys")
        # pprint(new_tree.keys(flat=True))


class TestLayout:

    def test_layout(self):
        layout = lat.Layout()

        layout.add_prim(pr.Quadrilateral(), "MyBox")
        layout.add_constraint(co.Box(), ("MyBox",), ())

        layout.add_constraint(co.Fix(), ("MyBox/Line0/Point0",), ([0, 0],))

        pprint(layout.root_prim)
        constraints, constraints_argkeys, constraints_param = layout.flat_constraints()

        print("Flat constraints: ")
        print("Constraints:")
        pprint(constraints)
        print("Constraints argument keys:")
        pprint(constraints_argkeys)
        print("Constraints parameter vector:")
        pprint(constraints_param)

