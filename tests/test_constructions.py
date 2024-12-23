"""
Test geometric onstraints
"""

import pytest

from numpy.typing import NDArray

import numpy as np

from mpllayout import primitives as pr
from mpllayout import constructions as con

from tests.fixture_primitives import GeometryFixtures


class TestPoint(GeometryFixtures):
    """
    Test constructions with signature `[Point]`
    """

    @pytest.fixture()
    def coordinate(self):
        return np.random.rand(2)

    def test_Coordinate(self, coordinate):
        point = self.make_point(coordinate)
        res = con.Coordinate()((point,)) - coordinate
        assert np.all(np.isclose(res, 0))


class TestLine(GeometryFixtures):
    """
    Test constraints with signature `[Line]`
    """

    @pytest.fixture()
    def length(self):
        return np.random.rand()

    @pytest.fixture()
    def direction(self):
        unit_vec = np.random.rand(2)
        unit_vec = unit_vec / np.linalg.norm(unit_vec)
        return unit_vec

    @pytest.fixture()
    def linea(self, length, direction):
        origin = np.random.rand(2)
        return self.make_line(origin, direction * length)

    def test_Length(self, linea, length):
        res = con.Length()((linea,)) - length
        assert np.all(np.isclose(res, 0))

    def test_DirectedLength(self, length, direction):
        line_dir = np.random.rand(2)
        line_vec = length*line_dir
        line = self.make_line((0, 0), line_vec)

        dlength = np.dot(line_vec, direction)
        res = con.DirectedLength()((line,), direction) - dlength
        assert np.all(np.isclose(res, 0))

    @pytest.fixture(
        params=[
            ('x', np.array([1, 0])),
            ('y', np.array([0, 1]))
        ]
    )
    def axis_name_dir(self, request):
        return request.param

    @pytest.fixture()
    def axis_name(self, axis_name_dir):
        return axis_name_dir[0]

    @pytest.fixture()
    def axis_dir(self, axis_name_dir):
        return axis_name_dir[1]

    @pytest.fixture()
    def XYLength(self, axis_name):
        if axis_name == 'x':
            return con.XLength
        else:
            return con.YLength

    def test_XYLength(self, XYLength, axis_dir, length):
        line_dir = np.random.rand(2)
        line_vec = length*line_dir
        line = self.make_line((0, 0), line_vec)

        dlength = np.dot(line_vec, axis_dir)

        res = XYLength()((line,)) - dlength

        assert np.all(np.isclose(res, 0))


class TestQuadrilateral(GeometryFixtures):
    """
    Test constructions with signature `[Quadrilateral]`
    """

    @pytest.fixture()
    def quada(self):
        return self.make_quad(np.random.rand(2), np.random.rand(2, 2))

    @pytest.fixture()
    def aspect_ratio(self, quada):
        width = np.linalg.norm(con.LineVector.assem((quada['Line0'],)))
        height = np.linalg.norm(con.LineVector.assem((quada['Line1'],)))
        return width/height

    def test_AspectRatio(self, quada: pr.Quadrilateral, aspect_ratio: float):
        res = con.AspectRatio()((quada,)) - aspect_ratio
        assert np.all(np.isclose(res, 0))
