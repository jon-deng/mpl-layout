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

from mpllayout import geometry as geo
from mpllayout import ui
from mpllayout.containers import Node


class GeometryFixtures:
    """
    Utilities to help create primitives
    """

    ## Point creation
    def make_point(self, coord):
        """
        Return a `geo.Point` at the given coordinates
        """
        return geo.Point(value=coord)

    def make_relative_point(self, point: geo.Point, displacement: NDArray):
        """
        Return a `geo.Point` displaced from a given point
        """
        return geo.Point(value=point.value + displacement)

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
        return geo.Line(value=[], children=tuple(geo.Point(x) for x in coords))

    def make_relline_about_start(
        self, line: geo.Line, translation: NDArray, deformation: NDArray
    ):
        """
        Return a `geo.Line` deformed about it's start point then translated
        """
        lineb_vec = line[1].value - line[0].value
        lineb_vec = deformation @ lineb_vec

        lineb_start = line[0].value + translation
        return self.make_line(lineb_start, lineb_vec)

    def make_relline_about_mid(
        self, line: geo.Line, translation: NDArray, deformation: NDArray
    ):
        """
        Return a `geo.Line` deformed about it's midpoint then translated
        """
        lineb_vec = line[1].value - line[0].value
        lineb_vec = deformation @ lineb_vec

        lineb_mid = 1/2*(line[0].value + line[1].value) + translation
        lineb_start = lineb_mid - lineb_vec/2
        return self.make_line(lineb_start, lineb_vec)

    ## Quadrilateral creation
    def make_quad(self, displacement, deformation):
        """
        Return a `geo.Quadrilateral` translated and deformed from a unit quadrilateral
        """
        # Specify vertices of a unit square, then deform it and translate it
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        verts = np.tensordot(verts, deformation, axes=(-1, -1))
        verts = verts + displacement

        return geo.Quadrilateral(
            value=[], children=tuple(geo.Point(vert) for vert in verts)
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

    def test_Fix(self, point):
        constraint = geo.Fix()
        res = constraint((point,), (point.value,))
        assert np.all(np.isclose(res, 0))

    def test_Coincident(
        self, point: geo.Point,
    ):
        res = geo.Coincident()((point, point), ())
        assert np.all(np.isclose(res, 0))

    def test_DirectedDistance(
        self, point: geo.Point, distance: float, direction: NDArray
    ):
        pointb = self.make_relative_point(point, distance * direction)
        constraint = geo.DirectedDistance()
        res = constraint((point, pointb), (distance, direction))
        assert np.all(np.isclose(res, 0))

    def test_DirectedDistance(
        self, point: geo.Point, distance: float, direction: NDArray
    ):
        pointb = self.make_relative_point(point, distance * direction)
        constraint = geo.DirectedDistance()
        res = constraint((point, pointb), (distance, direction))
        assert np.all(np.isclose(res, 0))

    def test_XDistance(
        self, point: geo.Point, distance: float
    ):
        direction = np.array([1, 0])
        pointb = self.make_relative_point(point, distance * direction)
        res_a = geo.XDistance()((point, pointb), (distance,))
        res_b = geo.DirectedDistance()((point, pointb), (distance, direction))
        assert np.all(np.isclose(res_a, res_b))

    def test_YDistance(
        self, point: geo.Point, distance: float
    ):
        direction = np.array([0, 1])
        pointb = self.make_relative_point(point, distance * direction)
        res_a = geo.YDistance()((point, pointb), (distance,))
        res_b = geo.DirectedDistance()((point, pointb), (distance, direction))
        assert np.all(np.isclose(res_a, res_b))


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

    def test_Length(self, line, length):
        res = geo.Length()((line,), (length,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def displacement(self):
        return np.random.rand(2)

    @pytest.fixture()
    def parallel_lines(self, line, displacement):
        lineb = self.make_relline_about_start(line, displacement, self.make_rotation(0))
        return (line, lineb)

    def test_Parallel(self, parallel_lines):
        res = geo.Parallel()(parallel_lines, ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def orthogonal_lines(self, line, displacement):
        lineb = self.make_relline_about_start(
            line, displacement, self.make_rotation(np.pi / 2)
        )
        return (line, lineb)

    def test_Orthogonal(self, orthogonal_lines):
        res = geo.Orthogonal()(orthogonal_lines, ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_length(self):
        return np.random.rand()

    def test_RelativeLength(self, line, displacement, relative_length):
        scale = relative_length * np.diag(np.ones(2))
        theta = 2 * np.pi * np.random.rand()
        rotate = self.make_rotation(theta)

        lineb = self.make_relline_about_start(line, displacement, scale @ rotate)
        res = geo.RelativeLength()((lineb, line), (relative_length,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def angle(self):
        return np.pi*np.random.rand()

    def test_Angle(self, line, displacement, angle):
        scale = np.random.rand() * np.diag(np.ones(2))
        rotate = self.make_rotation(angle)

        lineb = self.make_relline_about_start(line, displacement, scale @ rotate)
        res = geo.Angle()((line, lineb), (angle,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def num_lines(self):
        return 5

    @pytest.fixture()
    def relative_lengths(self, num_lines: int):
        return np.random.rand(num_lines)

    @pytest.fixture()
    def displacements(self, num_lines: int):
        return np.random.rand(num_lines, 2)

    def test_RelativeLengthArray(self, line, displacements, relative_lengths):
        num_lines = len(relative_lengths)+1
        scales = relative_lengths[:, None, None] * np.diag(np.ones(2))
        thetas = 2 * np.pi * np.random.rand(num_lines)
        rotates = [self.make_rotation(theta) for theta in thetas]

        lines = [
            self.make_relline_about_start(line, displacement, scale @ rotate)
            for displacement, scale, rotate in zip(displacements, scales, rotates)
        ] + [line]
        constraint = geo.RelativeLengthArray(len(relative_lengths))
        res = constraint(lines, (relative_lengths,))
        assert np.all(np.isclose(res, 0))

    def test_Collinear(self, line):
        line_vec = geo.line_vector(line)

        lineb = self.make_relline_about_start(
            line, np.random.rand()*line_vec, np.diag(np.ones(2))
        )
        res = geo.Collinear()((line, lineb), ())
        assert np.all(np.isclose(res, 0))

    def test_CollinearArray(self, line):
        num_lines = 5
        line_vec = geo.line_vector(line)

        dists = np.random.rand(num_lines-1)
        lines = (line,) + tuple(
            self.make_relline_about_start(line, dist*line_vec, np.diag(np.ones(2)))
            for dist in dists
        )
        res = geo.CollinearArray(num_lines)(lines, ())
        assert np.all(np.isclose(res, 0))

    def test_CoincidentLines(self, line):
        res = geo.CoincidentLines()((line, line), (False,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def distance(self):
        return np.random.rand()

    @pytest.fixture()
    def unit_direction(self):
        vec = np.random.rand(2)
        return vec/np.linalg.norm(vec)

    def test_XDistanceMidpoints(self, line, distance, unit_direction):
        # Create the shifted line by rotating about the midpoint then translating
        theta = np.random.rand()
        displacement = distance/unit_direction[0] * unit_direction
        lineb = self.make_relline_about_mid(line, displacement, self.make_rotation(theta))

        res = geo.XDistanceMidpoints()((line, lineb), (distance,))
        assert np.all(np.isclose(res, 0))

    def test_XDistanceMidpointsArray(self, line, unit_direction):
        distances = np.random.rand(5)
        N = len(distances)

        # Create the shifted line by rotating about the midpoint then translating
        thetas = np.random.rand(N)
        rotations = [self.make_rotation(theta) for theta in thetas]
        displacements = [
            distance/unit_direction[0] * unit_direction for distance in distances
        ]
        lineas = [
            self.make_relline_about_mid(line, displacement, rotation)
            for displacement, rotation in zip(displacements, rotations)
        ]

        # Create the shifted line by rotating about the midpoint then translating
        thetas = np.random.rand(N)
        rotations = [self.make_rotation(theta) for theta in thetas]
        displacements = [
            distance/unit_direction[0] * unit_direction for distance in distances
        ]
        linebs = [
            self.make_relline_about_mid(linea, displacement, rotation)
            for linea, displacement, rotation in zip(lineas, displacements, rotations)
        ]

        lines = tuple(
            itertools.chain.from_iterable(
                (linea, lineb) for linea, lineb in zip(lineas, linebs)
            )
        )
        res = geo.XDistanceMidpointsArray(N)(lines, (distances,))
        assert np.all(np.isclose(res, 0))

    def test_YDistanceMidpoints(self, line, distance, unit_direction):
        # Create the shifted line by rotating about the midpoint then translating
        theta = np.random.rand()
        displacement = distance/unit_direction[1] * unit_direction
        lineb = self.make_relline_about_mid(line, displacement, self.make_rotation(theta))

        res = geo.YDistanceMidpoints()((line, lineb), (distance,))
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
        col_margins = 0.2 * np.ones(num_col - 1)

        rel_row_heights = 3 * np.ones(num_row - 1)
        row_margins = 0.1 * np.ones(num_row - 1)

        grid_kwargs = {
            "col_widths": rel_col_widths,
            "row_heights": rel_row_heights,
            "col_margins": col_margins,
            "row_margins": row_margins,
        }
        return grid_kwargs

    @pytest.fixture()
    def quads(
        self,
        grid_origin_dimensions: tp.Tuple[float, float],
        rel_grid_dimensions: tp.Mapping[str, NDArray],
    ):
        origin = np.random.rand(2)
        origin = np.zeros(2)

        col_margins = rel_grid_dimensions["col_margins"]
        row_margins = rel_grid_dimensions["row_margins"]
        rel_col_widths = rel_grid_dimensions["col_widths"]
        rel_row_heights = rel_grid_dimensions["row_heights"]

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
        root_prim = Node(None, {f'Quad{n}': quad for n, quad in enumerate(quads)})

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.grid(which='both', axis='both')
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_aspect(1)
        # ax.set_xlim(0, 10)
        # ax.set_ylim(0, 10)
        ui.plot_prims(ax, root_prim)

        fig.savefig(f"out/quad_grid_{grid_shape}.png")

        res = geo.Grid(grid_shape)(quads, rel_grid_dimensions)
        assert np.all(np.isclose(res, 0))

    def test_RectilinearGrid(
        self, quads: tp.List[geo.Quadrilateral], grid_shape: tp.Tuple[int, int]
    ):
        # {"shape": grid_shape}
        res = geo.RectilinearGrid(grid_shape)(quads, ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def quad_box(self):
        translation = np.random.rand(2)
        deformation = np.diag(np.random.rand(2))
        return self.make_quad(translation, deformation)

    def test_Box(self, quad_box: geo.Quadrilateral):
        res = geo.Box()((quad_box,), ())

        assert np.all(np.isclose(res, 0))


from matplotlib import pyplot as plt
class TestAxesConstraints(GeometryFixtures):

    @pytest.fixture()
    def size(self):
        return (1, 1)

    @pytest.fixture()
    def axes(self, size: tp.Tuple[float, float]):
        width, height = size
        scale = np.array([[width, 0], [0, height]])
        frame = self.make_quad(np.zeros(2), scale)

        squash_height = np.array([[1, 0], [0, 0]])
        squash_width = np.array([[0, 0], [0, 1]])
        xaxis = self.make_quad(np.zeros(2), squash_height@scale)
        yaxis = self.make_quad(np.zeros(2), squash_width@scale)

        xlabel_anchor = self.make_point([0, 0])
        ylabel_anchor = self.make_point([0, 0])
        return geo.AxesXY(children=(frame, xaxis, yaxis, xlabel_anchor, ylabel_anchor))

    @pytest.fixture()
    def axes_mpl(self, size: tp.Tuple[float, float]):
        width, height = size
        # NOTE: Unit dimensions of the figure are important because `fig.add_axes`
        # units are relative to figure dimensions
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_axes((0, 0, width, height))

        x = np.pi*2*np.linspace(0, 10)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_xlabel("My favourite xlabel")
        ax.set_ylabel("My favourite ylabel")

        return ax

    # NOTE: The `..AxisHeight` tests rely on them using `Distance` type constraints to
    # implement the residual
    def test_XAxisHeight(self, axes, axes_mpl):
        height = geo.get_xaxis_height(axes_mpl.xaxis)

        xaxis_height = geo.XAxisHeight()
        res = xaxis_height((axes,), (axes_mpl,))

        assert np.all(np.isclose(res + height, 0))

    def test_YAxisWidth(self, axes, axes_mpl):
        width = geo.get_yaxis_width(axes_mpl.yaxis)

        xaxis_height = geo.YAxisWidth()
        res = xaxis_height((axes,), (axes_mpl,))

        assert np.all(np.isclose(res + width, 0))
