"""
Test `layout`
"""

import pytest

from pprint import pprint

import numpy as np

from mpllayout import geometry as geo, layout as lat, containers as cn


class TestPrimitiveTree:

    @pytest.fixture()
    def prim_node(self):
        return cn.Node(np.array([]), {})

    def test_set_prim(self, prim_node):
        prim_node.add_child("MyBox", geo.Quadrilateral())

        pprint(f"Keys:")
        pprint(prim_node["MyBox"].keys())

    def test_build_primtree(self, prim_node):
        point_a = geo.Point([0, 0])
        point_b = geo.Point([1, 1])
        prim_node.add_child("PointA", point_a)
        prim_node.add_child("LineA", geo.Line([], (point_a, point_b)))
        prim_node.add_child("MySpecialBox", geo.Quadrilateral())

        prim_graph, prims = lat.build_prim_graph(prim_node)

        params = [prim.value for prim in prims]

        new_params = [np.random.rand(*param.shape) for param in params]
        new_prim_node = lat.build_tree(prim_node, prim_graph, new_params)
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

        layout.add_prim(geo.Quadrilateral(), "MyBox")
        layout.add_constraint(geo.Box(), ("MyBox",), ())

        layout.add_constraint(geo.Fix(), ("MyBox/Line0/Point0",), ([0, 0],))

        pprint(layout.root_prim)
        constraints, constraints_argkeys, constraints_param = layout.flat_constraints()

        print("Flat constraints: ")
        print("Constraints:")
        pprint(constraints)
        print("Constraints argument keys:")
        pprint(constraints_argkeys)
        print("Constraints parameter vector:")
        pprint(constraints_param)

