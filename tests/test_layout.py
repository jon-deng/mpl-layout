"""
Test `layout`
"""


import pytest

from mpllayout import geometry as geo, layout as lat

class TestPrimitiveTree:

    @pytest.fixture()
    def prim_tree(self):
        return lat.PrimitiveTree(None, {})

    def test_set_prim(self, prim_tree):
        prim_tree['MyBox'] = geo.Quadrilateral()

        print("keys:", prim_tree.keys())
        print("flat keys:", prim_tree.keys(flat=True))
        print("prims", prim_tree.prims)

class TestLayout:

    def test_layout(self):
        layout = lat.Layout()

        layout.add_prim(geo.Quadrilateral(), 'MyBox')
        layout.add_constraint(geo.Box(), ('MyBox',))

        layout.add_constraint(geo.PointLocation((0, 0)), ('MyBox/Line0/Point0',))

        print(layout.prims)
        print(layout.constraints)
        print(layout.constraint_graph)
        print(layout.constraint_graph_int)

