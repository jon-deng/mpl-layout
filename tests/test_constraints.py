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
    """
    Utilities to help create primitives
    """

    ## Point creation
    def make_point(self, coord):
        """
        Return a `geo.Point` at the given coordinates
        """
        return geo.Point.from_std(value=coord)

    def make_relative_point(self, point: geo.Point, displacement: NDArray):
        """
        Return a `geo.Point` displaced from a given point
        """
        return geo.Point.from_std(value=point.value + displacement)

    ## Line creation
    def make_rotation(self, theta: float):
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return rot_mat

    def make_line(self, origin: NDArray, line_vec: NDArray):
        """
        Return a `geo.Line` with given origin and line vector
        """
        coords = (origin, origin + line_vec)
        return geo.Line.from_std(value=[], children=tuple(geo.Point.from_std(x) for x in coords))

    def make_relative_line(
        self, line: geo.Line, translation: NDArray, deformation: NDArray
    ):
        """
        Return a `geo.Line` translated and deformed from a given line
        """
        lineb_origin = line[0].value + translation
        lineb_vec = line[1].value - line[0].value
        lineb_vec = deformation @ lineb_vec
        return self.make_line(lineb_origin, lineb_vec)

    ## Quadrilateral creation
    def make_quad(self, displacement, deformation):
        """
        Return a `geo.Quadrilateral` translated and deformed from a unit quadrilateral
        """
        # Specify vertices of a unit square, then deform it and translate it
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        verts = np.tensordot(verts, deformation, axes=(-1, -1))
        verts = verts + displacement

        return geo.Quadrilateral.from_std(
            value=[], children=tuple(geo.Point.from_std(vert) for vert in verts)
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

        cumu_widths = np.cumsum(
            np.concatenate((translation[[0]], col_widths[:-1] + col_margins))
        )
        col_trans = np.stack([np.array([x, 0]) for x in cumu_widths], axis=0)
        cumu_heights = np.cumsum(
            np.concatenate((translation[[1]], -row_heights[1:] - row_margins))
        )
        row_trans = np.stack([np.array([0, y]) for y in cumu_heights], axis=0)

        row_args = zip(row_trans, row_defs)
        col_args = zip(col_trans, col_defs)

        quads = [
            self.make_quad(drow + dcol, row_def @ col_def)
            for (drow, row_def), (dcol, col_def) in itertools.product(
                row_args, col_args
            )
        ]
        return quads


class TestPointConstraints(GeometryFixtures):

    @pytest.fixture()
    def point(self):
        return self.make_point(np.random.rand(2))

    @pytest.fixture()
    def direction(self):
        vec = np.random.rand(2)
        return vec / np.linalg.norm(vec)

    @pytest.fixture()
    def distance(self):
        return np.random.rand()

    def test_DirectedDistance(
        self, point: geo.Point, distance: float, direction: NDArray
    ):
        pointb = self.make_relative_point(point, distance * direction)
        constraint = geo.DirectedDistance.init_from_constants({'distance':distance, 'direction':direction})
        res = constraint((point, pointb))
        assert np.all(np.isclose(res, 0))

    def test_PointLocation(self, point):
        constraint = geo.PointLocation.init_from_constants({'location': point.value})
        res = constraint((point,))
        assert np.all(np.isclose(res, 0))


class TestLineConstraints(GeometryFixtures):

    ## Constraints on line segments
    @pytest.fixture()
    def length(self):
        return np.random.rand()

    @pytest.fixture()
    def line(self, length):
        origin = np.random.rand(2)
        unit_vec = np.random.rand(2)
        unit_vec = unit_vec / np.linalg.norm(unit_vec)
        return self.make_line(origin, unit_vec * length)

    @pytest.fixture()
    def displacement(self):
        return np.random.rand(2)

    @pytest.fixture()
    def parallel_lines(self, line, displacement):
        lineb = self.make_relative_line(line, displacement, self.make_rotation(0))
        return (line, lineb)

    def test_Parallel(self, parallel_lines):
        constraint = geo.Parallel.init_from_constants({})
        res = constraint(parallel_lines)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def orthogonal_lines(self, line, displacement):
        lineb = self.make_relative_line(
            line, displacement, self.make_rotation(np.pi / 2)
        )
        return (line, lineb)

    def test_Orthogonal(self, orthogonal_lines):
        constraint = geo.Orthogonal.init_from_constants({})
        res = constraint(orthogonal_lines)
        assert np.all(np.isclose(res, 0))

    def test_Length(self, line, length):
        constraint = geo.Length.init_from_constants({'length': length})
        res = constraint((line,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_length(self):
        return np.random.rand()

    def test_RelativeLength(self, line, displacement, relative_length):
        scale = relative_length * np.diag(np.ones(2))
        theta = 2 * np.pi * np.random.rand()
        rotate = self.make_rotation(theta)

        lineb = self.make_relative_line(line, displacement, scale @ rotate)
        constraint = geo.RelativeLength.init_from_constants({'length': relative_length})
        res = constraint((lineb, line))
        assert np.all(np.isclose(res, 0))


class TestQuadConstraints(GeometryFixtures):
    # @pytest.fixture(params=[(2, 1)])
    @pytest.fixture(params=[(1, 1), (2, 1), (1, 2), (2, 2)])
    def grid_shape(self, request):
        return request.param

    @pytest.fixture()
    def grid_origin_dimensions(self):
        # width, height = np.random.rand(2)
        width, height = 1, 1
        return (width, height)

    @pytest.fixture()
    def rel_grid_dimensions(self, grid_shape: tp.Tuple[int, int]):
        num_row, num_col = grid_shape

        ## Random sizes and margins
        # widths = np.random.rand(num_col - 1)
        # col_margins = 0.1*np.random.rand(num_col - 1)

        # heights = np.random.rand(num_row - 1)
        # row_margins = 0.1*np.random.rand(num_row - 1)

        ## Specific sizes and margins
        rel_col_widths = 2 * np.ones(num_col - 1)
        col_margins = 1 * np.ones(num_col - 1)

        rel_row_heights = 6 * np.ones(num_row - 1)
        row_margins = 3 * np.ones(num_row - 1)

        return col_margins, row_margins, rel_col_widths, rel_row_heights

    @pytest.fixture()
    def quads(
        self,
        grid_origin_dimensions: tp.Tuple[float, float],
        rel_grid_dimensions: tp.Tuple[NDArray, NDArray, NDArray, NDArray],
    ):
        origin = np.random.rand(2)
        origin = np.zeros(2)

        col_margins, row_margins, rel_col_widths, rel_row_heights = rel_grid_dimensions

        origin_width, origin_height = grid_origin_dimensions
        col_widths = origin_width * np.concatenate(([1], rel_col_widths))
        row_heights = origin_height * np.concatenate(([1], rel_row_heights))

        return self.make_quad_grid(
            origin, col_margins, row_margins, col_widths, row_heights
        )

    def test_Grid(
        self,
        quads: tp.List[geo.Quadrilateral],
        grid_shape: tp.Tuple[int, int],
        rel_grid_dimensions: tp.Tuple[NDArray, NDArray, NDArray, NDArray],
    ):
        res = geo.Grid.init_from_constants({'shape': grid_shape, **rel_grid_dimensions})(quads)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def quad_box(self):
        translation = np.random.rand(2)
        deformation = np.diag(np.random.rand(2))
        return self.make_quad(translation, deformation)

    def test_Box(self, quad_box: geo.Quadrilateral):
        res = geo.Box.init_from_constants({})((quad_box,))
        assert np.all(np.isclose(res, 0))
