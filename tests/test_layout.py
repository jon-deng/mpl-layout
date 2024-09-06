"""
Test `layout`
"""

import pytest

from pprint import pprint

from mpllayout import geometry as geo, layout as lat

class TestPrimitiveTree:

    @pytest.fixture()
    def prim_tree(self):
        return lat.PrimitiveTree(None, {})

    def test_set_prim(self, prim_tree):
        prim_tree['MyBox'] = lat.convert_prim_to_tree(geo.Quadrilateral())

        pprint(f"Keys:")
        pprint(prim_tree.children['MyBox'].keys())

        pprint("Flat keys:")
        pprint(prim_tree.keys(flat=True))

        pprint("Unique prims:")
        pprint(prim_tree.prims)

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

