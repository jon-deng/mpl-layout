"""
Test geometric onstraints
"""

import pytest

from numpy.typing import NDArray

import numpy as np

from mpllayout import primitives as pr
from mpllayout import constructions as con

from tests.fixture_primitives import GeometryFixtures

class TestConstructionFunctions(GeometryFixtures):

    def test_transform_map(self):
        num_point = 2
        coords = np.random.rand(num_point, 2)
        points = tuple(pr.Point(value=coord) for coord in coords)

        coords_ref = np.concatenate([con.Coordinate()((point,)) for point in points])
        MapCoordinate = con.transform_MapType(con.Coordinate, num_point*[pr.Point])
        coords_map = MapCoordinate()(points)

        assert np.all(np.isclose(coords_ref, coords_map))

    def test_transform_constraint(self):

        construction = con.OuterMargin(side='right')
        quada = self.make_quad(np.zeros(2), np.diag(np.ones(2)))
        quadb = self.make_quad(np.array([1.5, 0]), np.diag(np.ones(2)))
        prims = (quada, quadb)

        value = construction(prims)

        constraint = con.transform_constraint(construction)
        res = constraint(prims, value)
        assert np.all(np.isclose(res, 0))

        construction = con.Coordinate()
        point = self.make_point(np.random.rand(2))
        prims = (point,)

        value = construction(prims)

        constraint = con.transform_constraint(construction)
        res = constraint(prims, value)
        assert np.all(np.isclose(res, 0))

    def test_transform_sum(self):
        construction = con.Coordinate()
        sum_construction = con.transform_sum(construction, construction)

        prims = (self.make_point(np.random.rand(2)),)
        params = ()

        res_a = construction(prims, *params) + construction(prims, *params)
        res_b = sum_construction(prims, *params)

        assert np.all(np.isclose(res_a, res_b))

    def test_transform_scalar_mul(self):
        construction = con.Coordinate()

        prims = (self.make_point(np.random.rand(2)),)
        cons_params = ()
        scalar = np.random.rand()
        res_a = scalar * construction(prims, *cons_params)

        # Test constant version
        mul_construction = con.transform_scalar_mul(construction, scalar)
        params = cons_params + ()
        res_b = mul_construction(prims, *params)

        assert np.all(np.isclose(res_a, res_b))

        # Test non constant version
        mul_construction = con.transform_scalar_mul(construction, None)
        params = cons_params + (scalar,)
        res_b = mul_construction(prims, *params)

        assert np.all(np.isclose(res_a, res_b))

    def test_transform_dummy_params(self):

        construction = con.Coordinate()

        prims = (self.make_point(np.random.rand(2)),)
        cons_params = ()
        res_a = construction(prims, *cons_params)

        # Test adding random set of dummy parameters
        n_dummy_params = 5
        dummy_param_types = n_dummy_params*(float,)
        dummy_params = tuple(np.random.rand(n_dummy_params))

        trans_construction = con.transform_dummy_params(construction, dummy_param_types)
        res_b = trans_construction(prims, *(cons_params + dummy_params))

        assert np.all(np.isclose(res_a, res_b))


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


class TestQuadrilateralQuadrilateral(GeometryFixtures):
    """
    Test constraints with signature `[Quadrilateral, Quadrilateral]`
    """

    @pytest.fixture()
    def margin(self):
        return np.random.rand()

    @pytest.fixture()
    def boxa(self):
        size = np.random.rand(2)
        origin = np.random.rand(2)
        return self.make_quad(origin, np.diag(size))

    @pytest.fixture()
    def boxb(self):
        size = np.random.rand(2)
        origin = np.random.rand(2)
        return self.make_quad(origin, np.diag(size))

    @pytest.fixture(params=('bottom', 'top', 'left', 'right'))
    def margin_side(self, request):
        return request.param

    @pytest.fixture()
    def outer_margin(self, boxa, boxb, margin_side):
        a_topright = boxa['Line1/Point1'].value
        b_topright = boxb['Line1/Point1'].value
        a_botleft = boxa['Line0/Point0'].value
        b_botleft = boxb['Line0/Point0'].value

        if margin_side == 'left':
            margin = (a_botleft - b_topright)[0]
        elif margin_side == 'right':
            margin = (b_botleft - a_topright)[0]
        elif margin_side == 'bottom':
            margin = (a_botleft - b_topright)[1]
        elif margin_side == 'top':
            margin = (b_botleft - a_topright)[1]
        return margin

    def test_OuterMargin(self, boxa, boxb, outer_margin, margin_side):
        res = con.OuterMargin(side=margin_side)((boxa, boxb)) - outer_margin
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def inner_margin(self, boxa, boxb, margin_side):
        a_topright = boxa['Line1/Point1'].value
        b_topright = boxb['Line1/Point1'].value
        a_botleft = boxa['Line0/Point0'].value
        b_botleft = boxb['Line0/Point0'].value

        if margin_side == 'left':
            margin = (a_botleft - b_botleft)[0]
        elif margin_side == 'right':
            margin = (b_topright - a_topright)[0]
        elif margin_side == 'bottom':
            margin = (a_botleft - b_botleft)[1]
        elif margin_side == 'top':
            margin = (b_topright-a_topright)[1]
        return margin

    def test_InnerMargin(self, boxa, boxb, inner_margin, margin_side):
        res = con.InnerMargin(side=margin_side)((boxa, boxb)) - inner_margin
        assert np.all(np.isclose(res, 0))

