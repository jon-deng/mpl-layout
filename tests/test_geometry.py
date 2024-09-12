"""
Test geometric primitive and constraints
"""

import pytest

import typing as typ

import itertools
from pprint import pprint

import numpy as np

from mpllayout import geometry as geo


class TestPrimitives:

    def test_Quadrilateral(self):
        quad = geo.Quadrilateral()

    def test_Line(self):
        line = geo.Line()

    def test_Point(self):
        point = geo.Point()

    def test_Polygon(self):
        poly = geo.Polygon()


class TestConstraints:

    ## Constraints on points
    @pytest.fixture()
    def vertices(self):
        return np.array([[0, 0], [2, 2], [4, 4]])

    @pytest.fixture()
    def points(self, vertices):
        return [geo.Point(vert) for vert in vertices]

    @pytest.fixture()
    def direction(self):
        return np.array([0, 1], dtype=float)

    @pytest.fixture()
    def location(self):
        return np.array([5, 5], dtype=float)

    def test_PointToPointAbsDistance(self, points, direction):
        ans_ref = np.dot(points[1].param - points[0].param, direction)

        dist = geo.DirectedDistance(0)
        ans_com = dist(points[:2])
        assert np.isclose(ans_ref, ans_com)

    def test_PointLocation(self, points, location):
        ans_ref = points[0].param - location

        constraint = geo.PointLocation(location)
        ans_com = constraint(points[:2])
        assert np.all(np.isclose(ans_ref, ans_com))

    @pytest.fixture(params=[(1, 1), (2, 1), (1, 2), (2, 2)])
    def shape(self, request):
        return request.param

    @pytest.fixture()
    def quads(self, shape):
        num_quads = int(np.prod(shape))

        quads = []
        for ii, jj in itertools.product(*(range(size) for size in shape)):
            height = 1
            width = 2
            height_margin = 0.1
            width_margin = 0.1
            origin_topleft = np.array(
                [ii * (height + height_margin), jj * (width + width_margin)],
                dtype=float,
            )
            points = [
                origin_topleft - [0, height],
                origin_topleft - [0, height] + [width, 0],
                origin_topleft - [0, height] + [width, 0] + [0, height],
                origin_topleft,
            ]
            quads.append(geo.Quadrilateral([], [geo.Point(point) for point in points]))

        return quads

    def test_Grid(self, quads: typ.List[geo.Quadrilateral], shape: typ.Tuple[int, ...]):
        num_row, num_col = shape

        res = geo.Grid(
            shape,
            (num_col - 1) * [0.1],
            (num_row - 1) * [0.1],
            (num_col - 1) * [1],
            (num_row - 1) * [1],
        )(quads)

        print(res)

    # def test_CoincidentPoint(self, points):
    #     ans_ref = points[0].param

    #     dist = geo.PointToPointAbsDistance(0)
    #     ans_com = dist(points[:2])
    #     assert np.isclose(ans_ref, ans_com)

    ## Constraints on line segments

    @pytest.fixture()
    def lines(self, points):
        return [geo.Line(prims=(pa, pb)) for pa, pb in zip(points[:-1], points[1:])]

    @pytest.fixture()
    def orthogonal_lines(self):
        vec_a = np.random.rand(2) - 0.5
        vec_b = np.array([-vec_a[1], vec_a[0]])

        # Make the line starting point somewhere in a 10x10 box around the origin
        vert1_a = 10 * 2 * (np.random.rand(2) - 0.5)
        vert1_b = 10 * 2 * (np.random.rand(2) - 0.5)
        lines = tuple(
            geo.Line(prims=(geo.Point(vert1), geo.Point(vert1 + vec)))
            for vert1, vec in zip([vert1_a, vert1_b], [vec_a, vec_b])
        )
        return lines

    @pytest.fixture()
    def parallel_lines(self, points):
        return (geo.Line(prims=(pa, pb)) for pa, pb in zip(points[:-1], points[1:]))

    def test_LineLength(self, lines):
        line = lines[0]
        points = line.prims
        vec = points[1].param - points[0].param
        ans_ref = np.linalg.norm(vec) ** 2

        constraint = geo.Length(0)
        ans_com = constraint((line,))

        assert np.all(np.isclose(ans_ref, ans_com))

    def test_RelativeLineLength(self, lines):
        lines = tuple(lines[:2])
        vecs = tuple(line.prims[1].param - line.prims[0].param for line in lines)
        line_lengths = tuple(np.linalg.norm(vec) for vec in vecs)

        rel_length = 0.25
        ans_ref = (line_lengths[0]) ** 2 - (rel_length * line_lengths[1]) ** 2

        constraint = geo.RelativeLength(rel_length)
        ans_com = constraint((lines))

        assert np.all(np.isclose(ans_ref, ans_com))

    def test_Orthogonal(self, orthogonal_lines):
        lines = orthogonal_lines

        constraint = geo.Orthogonal()
        ans_com = constraint(lines)

        assert np.all(np.isclose([0, 0], ans_com))
