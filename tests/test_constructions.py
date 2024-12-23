"""
Test geometric onstraints
"""

import pytest

import typing as tp
from numpy.typing import NDArray

import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from mpllayout import primitives as pr
from mpllayout import constructions as con
from mpllayout import ui
from mpllayout.containers import Node


class GeometryFixtures:
    """
    Utilities to help create primitives
    """

    ## Point creation
    def make_point(self, coord):
        """
        Return a `pr.Point` at the given coordinates
        """
        return pr.Point(value=coord)

    def make_relative_point(self, point: pr.Point, displacement: NDArray):
        """
        Return a `pr.Point` displaced from a given point
        """
        return pr.Point(value=point.value + displacement)

    ## Line creation
    def make_rotation(self, theta: float):
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return rot_mat

    def make_line(self, origin: NDArray, line_vec: NDArray):
        """
        Return a `pr.Line` with given origin and line vector
        """
        coords = (origin, origin + line_vec)
        return pr.Line(value=[], prims=tuple(pr.Point(x) for x in coords))

    def make_relative_line(
        self, line: pr.Line, translation: NDArray, deformation: NDArray
    ):
        """
        Return a `pr.Line` deformed about it's start point then translated
        """
        lineb_vec = line[1].value - line[0].value
        lineb_vec = deformation @ lineb_vec

        lineb_start = line[0].value + translation
        return self.make_line(lineb_start, lineb_vec)

    def make_relline_about_mid(
        self, line: pr.Line, translation: NDArray, deformation: NDArray
    ):
        """
        Return a `pr.Line` deformed about it's midpoint then translated
        """
        lineb_vec = line[1].value - line[0].value
        lineb_vec = deformation @ lineb_vec

        lineb_mid = 1/2*(line[0].value + line[1].value) + translation
        lineb_start = lineb_mid - lineb_vec/2
        return self.make_line(lineb_start, lineb_vec)

    ## Quadrilateral creation
    def make_quad(self, displacement, deformation):
        """
        Return a `pr.Quadrilateral` translated and deformed from a unit quadrilateral
        """
        # Specify vertices of a unit square, then deform it and translate it
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        verts = np.tensordot(verts, deformation, axes=(-1, -1))
        verts = verts + displacement

        return pr.Quadrilateral(
            value=[], children=tuple(pr.Point(vert) for vert in verts)
        )

    def make_quad_grid(
        self,
        translation: NDArray,
        col_margins: NDArray,
        row_margins: NDArray,
        col_widths: NDArray,
        row_heights: NDArray,
    ):
        """
        Return a grid of `Quadrilateral`s with shape (M, N)

        Rows and columns are numbered from top to bottom and left to right, respectively.

        Parameters
        ----------
        col_margins, row_margins: NDArray (N-1,), (M-1,)
            Column and row margins
        col_widths, row_heights: NDArray (N,), (M,)
            Column and row dimensions
        """
        # Determine translations/transformations needed for each quad
        col_defs = col_widths[:, None, None] * np.outer([1, 0], [1, 0]) + np.outer(
            [0, 1], [0, 1]
        )
        row_defs = np.outer([1, 0], [1, 0]) + row_heights[:, None, None] * np.outer(
            [0, 1], [0, 1]
        )

        cum_col_widths = np.cumsum(
            np.concatenate((translation[[0]], col_widths[:-1] + col_margins))
        )
        col_trans = np.stack([np.array([x, 0]) for x in cum_col_widths], axis=0)
        cum_row_heights = np.cumsum(
            np.concatenate((translation[[1]], -row_heights[1:] - row_margins))
        )
        row_trans = np.stack([np.array([0, y]) for y in cum_row_heights], axis=0)

        row_args = zip(row_trans, row_defs)
        col_args = zip(col_trans, col_defs)

        quads = tuple(
            self.make_quad(drow + dcol, row_def @ col_def)
            for (drow, row_def), (dcol, col_def) in itertools.product(
                row_args, col_args
            )
        )
        return quads


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
