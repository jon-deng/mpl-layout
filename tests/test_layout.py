"""
Test `layout`
"""

import pytest

from pprint import pprint

import numpy as np

from mpllayout import geometry as geo, layout as lat

class TestPrimitiveTree:

    @pytest.fixture()
    def prim_tree(self):
        return lat.PrimitiveTree(None, {})

    def test_set_prim(self, prim_tree):
        prim_tree['MyBox'] = lat.convert_primitive_to_tree(geo.Quadrilateral())

        pprint(f"Keys:")
        pprint(prim_tree.children['MyBox'].keys())

        pprint("Flat keys:")
        pprint(prim_tree.keys(flat=True))

        pprint("Unique prims:")
        pprint(prim_tree.prims)

    def test_build_primtree(self, prim_tree):
        point_a = geo.Point([0, 0])
        point_b = geo.Point([1, 1])
        prim_tree['PointA'] = lat.convert_primitive_to_tree(point_a)
        prim_tree['LineA'] = lat.convert_primitive_to_tree(geo.Line([], (point_a, point_b)))

        prim_tree['MySpecialBox'] = lat.convert_primitive_to_tree(geo.Quadrilateral())

        prim_graph = prim_tree.prim_graph

        rng = np.random.default_rng()

        new_params = [rng.random(prim.param.shape) for prim in prim_tree.prims]

        new_tree = lat.build_tree(prim_tree, prim_graph, new_params, {})

        print("Old primitive graph:")
        pprint(prim_tree.prim_graph)

        print("Old primitive list")
        pprint(prim_tree.prims)

        print("Old primitive keys")
        pprint(prim_tree.keys(flat=True))

        print("New parameters")
        pprint(new_params)

        print("New primitive graph:")
        pprint(new_tree.prim_graph)

        print("New primitive list")
        pprint(new_tree.prims)

        print("New primitive keys")
        pprint(new_tree.keys(flat=True))

class TestLayout:

    def test_layout(self):
        layout = lat.Layout()

        layout.add_prim(geo.Quadrilateral(), 'MyBox')
        layout.add_constraint(geo.Box(), ('MyBox',))

        layout.add_constraint(geo.PointLocation((0, 0)), ('MyBox/Line0/Point0',))

        pprint(layout.prims)
        pprint(layout.constraints)
        pprint(layout.constraint_graph)
        pprint(layout.constraint_graph_int)

