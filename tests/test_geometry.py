"""
Test geometric primitive and constraints 
"""

import pytest
from pprint import pprint

import numpy as np

from mpllayout import geometry as geo

class TestConstraints:

    ## Constraints on points
    @pytest.fixture()
    def vertices(self):
        return np.array([
            [0, 0], [2, 2], [4, 4]
        ])

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

        dist = geo.PointToPointDirectedDistance(0)
        ans_com = dist(points[:2])
        assert np.isclose(ans_ref, ans_com)

    def test_PointLocation(self, points, location):
        ans_ref = points[0].param - location

        constraint = geo.PointLocation(location)
        ans_com = constraint(points[:2])
        assert np.all(np.isclose(ans_ref, ans_com))

    # def test_CoincidentPoint(self, points):
    #     ans_ref = points[0].param

    #     dist = geo.PointToPointAbsDistance(0)
    #     ans_com = dist(points[:2])
    #     assert np.isclose(ans_ref, ans_com)

    ## Constraints on line segments

    @pytest.fixture()
    def lines(self, points):
        return [
            geo.LineSegment(prims=(pa, pb)) 
            for pa, pb in zip(points[:-1], points[1:])
        ]
    
    @pytest.fixture()
    def orthogonal_lines(self):
        vec_a = np.random.rand(2)-0.5
        vec_b = np.array([-vec_a[1], vec_a[0]])

        # Make the line starting point somewhere in a 10x10 box around the origin
        vert1_a = 10*2*(np.random.rand(2)-0.5)
        vert1_b = 10*2*(np.random.rand(2)-0.5)
        lines = tuple(
            geo.LineSegment(prims=(geo.Point(vert1), geo.Point(vert1+vec)))
            for vert1, vec in zip([vert1_a, vert1_b], [vec_a, vec_b])
        )
        return lines
    
    @pytest.fixture()
    def parallel_lines(self, points):
        return (
            geo.LineSegment(prims=(pa, pb)) 
            for pa, pb in zip(points[:-1], points[1:])
        )
    
    def test_LineLength(self, lines):
        line = lines[0]
        points = line.prims
        vec = points[1].param - points[0].param
        ans_ref = np.linalg.norm(vec)

        constraint = geo.LineLength(0)
        ans_com = constraint((line,))

        assert np.all(np.isclose(ans_ref, ans_com))

    def test_RelativeLineLength(self, lines):
        lines = tuple(lines[:2])
        vecs = tuple(line.prims[1].param - line.prims[0].param for line in lines)
        line_lengths = tuple(np.linalg.norm(vec) for vec in vecs)

        rel_length = 0.25
        ans_ref = line_lengths[0] - rel_length*line_lengths[1]

        constraint = geo.RelativeLineLength(rel_length)
        ans_com = constraint((lines))

        assert np.all(np.isclose(ans_ref, ans_com))

    def test_Orthogonal(self, orthogonal_lines):
        lines = orthogonal_lines

        constraint = geo.OrthogonalLines()
        ans_com = constraint(lines)

        assert np.all(np.isclose([0, 0], ans_com))
