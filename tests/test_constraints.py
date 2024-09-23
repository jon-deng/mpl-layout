"""
Test geometric onstraints
"""

import pytest

import typing as tp
from numpy.typing import NDArray

import itertools

import numpy as np

from mpllayout import geometry as geo


class GeometryFixtures:

    ## Point pairs
    @pytest.fixture()
    def distance_pointa_pointb(self):
        return 2.4

    @pytest.fixture()
    def unit_pointa_pointb(self):
        vec = np.random.rand(2)
        return vec / np.linalg.norm(vec)

    @pytest.fixture()
    def pointa(self):
        return geo.Point(value=np.random.rand(2))

    def make_point(self, pointa: geo.Point, distance: float, unit_vec: NDArray):
        return geo.Point(value=pointa.value + distance * unit_vec)

    ## Line pairs
    @pytest.fixture()
    def unit_linea(self, unit_pointa_pointb):
        return unit_pointa_pointb

    @pytest.fixture()
    def length_linea(self):
        return 5.5

    @pytest.fixture()
    def linea(self, pointa, unit_linea, length_linea):
        return geo.Line(
            children=(pointa, geo.Point(pointa.value + unit_linea * length_linea))
        )

    def make_rotation(self, theta):
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return rot_mat

    def make_line(self, linea, translation, rotation, scale):
        lineb_origin = linea[0].value + translation
        lineb_vec = linea[1].value - linea[0].value
        lineb_vec = scale * self.make_rotation(rotation) @ lineb_vec
        return geo.Line(
            children=(geo.Point(lineb_origin), geo.Point(lineb_origin + lineb_vec))
        )

    ## Quadrilaterals


class TestPointConstraints(GeometryFixtures):

    @pytest.fixture()
    def direction(self):
        vec = np.random.rand(2)
        return vec/np.linalg.norm(vec)

    @pytest.fixture()
    def distance(self):
        return 2.9

    def test_DirectedDistance(
        self,
        pointa: geo.Point,
        distance: float,
        direction: NDArray
    ):
        pointb = self.make_point(pointa, distance, direction, )
        constraint = geo.DirectedDistance(distance, direction)
        res = constraint((pointa, pointb))
        assert np.all(np.isclose(res, 0))

    def test_PointLocation(self, pointa):
        constraint = geo.PointLocation(pointa.value)
        res = constraint((pointa,))
        assert np.all(np.isclose(res, 0))


class TestLineConstraints(GeometryFixtures):

    ## Constraints on line segments
    @pytest.fixture()
    def translation(self):
        return np.random.rand(2)

    @pytest.fixture()
    def parallel_lines(self, linea, translation):
        lineb = self.make_line(linea, translation, 0, 1.0)
        return (linea, lineb)

    def test_Parallel(self, parallel_lines):
        constraint = geo.Parallel()
        res = constraint(parallel_lines)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def orthogonal_lines(self, linea, translation):
        lineb = self.make_line(linea, translation, np.pi / 2, 1.0)
        return (linea, lineb)

    def test_Orthogonal(self, orthogonal_lines):
        constraint = geo.Orthogonal()
        res = constraint(orthogonal_lines)
        assert np.all(np.isclose(res, 0))

    def test_Length(self, linea, length_linea):
        constraint = geo.Length(length_linea)
        res = constraint((linea,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_length(self):
        return 1.3

    def test_RelativeLength(self, linea, translation, relative_length):
        lineb = self.make_line(linea, translation, np.random.rand(), relative_length)
        constraint = geo.RelativeLength(relative_length)
        res = constraint((lineb, linea))
        assert np.all(np.isclose(res, 0))


class TestQuadConstraints:
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

    def test_Box(self, quads: tp.List[geo.Quadrilateral], shape: tp.Tuple[int, ...]):
        num_row, num_col = shape

        res = geo.Grid(
            shape,
            (num_col - 1) * [0.1],
            (num_row - 1) * [0.1],
            (num_col - 1) * [1],
            (num_row - 1) * [1],
        )(quads)

        print(res)

    def test_Grid(self, quads: tp.List[geo.Quadrilateral], shape: tp.Tuple[int, ...]):
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
    #     ans_ref = points[0].value

    #     dist = geo.PointToPointAbsDistance(0)
    #     ans_com = dist(points[:2])
    #     assert np.isclose(ans_ref, ans_com)
