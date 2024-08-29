"""
Test `layout`
"""


import pytest

from mpllayout import geometry as geo, layout

class TestPrimitiveTree:

    @pytest.fixture()
    def prim_tree(self):
        return layout.PrimitiveTree(None, {})

    def test_set_prim(self, prim_tree):
        prim_tree['MyBox'] = geo.Quadrilateral()

        print(prim_tree.keys())
