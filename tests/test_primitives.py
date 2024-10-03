"""
Test geometric primitive and constraints
"""

import pytest

import typing as tp

import itertools
from pprint import pprint

import numpy as np

from mpllayout import geometry as geo


class TestPrimitives:

    def test_Quadrilateral(self):
        quad = geo.Quadrilateral.from_std()

    def test_Line(self):
        line = geo.Line.from_std()

    def test_Point(self):
        point = geo.Point.from_std()

    def test_Polygon(self):
        poly = geo.Polygon.from_std()

    def test_Primitive_jax_pytree(self):
        # breakpoint()
        from jax import tree_util

        point = geo.Point.from_std()
        line = geo.Line.from_std()
        quad = geo.Quadrilateral.from_std()

        for prim in (point, line, quad):
            print(f"\nTesting primitive type {type(prim).__name__}")
            leaves = tree_util.tree_leaves(prim)
            print("Leaves:", leaves)

            value_flat, value_tree = tree_util.tree_flatten(prim)
            reconstructed_prim = tree_util.tree_unflatten(value_tree, value_flat)
            print("tree_util.tree_flatten:", value_flat, value_tree)
            print("tree_util.tree_unflatten:", reconstructed_prim)

            leaves = tree_util.tree_leaves([0, 1, 2, 3, [4, 5, [6, 7, [8]]]])
            print(leaves)
            print([type(leaf) for leaf in leaves])
