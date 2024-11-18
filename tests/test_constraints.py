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
from mpllayout import constraints as co
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
    Test constraints with signature `[Point]`
    """

    @pytest.fixture()
    def pointa(self):
        return self.make_point(np.random.rand(2))

    @pytest.fixture()
    def direction(self):
        vec = np.random.rand(2)
        return vec / np.linalg.norm(vec)

    @pytest.fixture()
    def distance(self):
        return np.random.rand()

    def test_Fix(self, pointa):
        constraint = co.Fix()
        res = constraint((pointa,), (pointa.value,))
        assert np.all(np.isclose(res, 0))


class TestPointPoint(GeometryFixtures):
    """
    Test constraints with signature `[Point, Point]`
    """

    @pytest.fixture()
    def pointa(self):
        return self.make_point(np.random.rand(2))

    @pytest.fixture()
    def direction(self):
        vec = np.random.rand(2)
        return vec / np.linalg.norm(vec)

    @pytest.fixture()
    def distance(self):
        return np.random.rand()

    @pytest.fixture()
    def pointb(self, pointa, distance, direction):
        return self.make_relative_point(pointa, distance * direction)

    def test_DirectedDistance(
        self, pointa: pr.Point, pointb: pr.Point, distance: float, direction: NDArray
    ):
        constraint = co.DirectedDistance()
        res = constraint((pointa, pointb), (distance, direction))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def xdistance(self, distance, direction):
        return distance*direction[0]

    def test_XDistance(
        self, pointa: pr.Point, pointb: pr.Point, xdistance: float
    ):
        res = co.XDistance()((pointa, pointb), (xdistance,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def ydistance(self, distance, direction):
        return distance*direction[1]

    def test_YDistance(
        self, pointa: pr.Point, pointb: pr.Point, ydistance: float
    ):
        res = co.YDistance()((pointa, pointb), (ydistance,))
        assert np.all(np.isclose(res, 0))


    def test_Coincident(
        self, pointa: pr.Point,
    ):
        res = co.Coincident()((pointa, pointa), ())
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
        res = co.Length()((linea,), (length,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def xlength(self, length, direction):
        proj_dir = np.array([1, 0])
        proj_length = length * np.dot(direction, proj_dir)
        return proj_length

    def test_XLength(self, linea, xlength):
        res = co.XLength()((linea,), (xlength,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def ylength(self, length, direction):
        proj_dir = np.array([0, 1])
        proj_length = length * np.dot(direction, proj_dir)
        return proj_length

    def test_YLength(self, linea, ylength):
        res = co.YLength()((linea,), (ylength,))
        assert np.all(np.isclose(res, 0))


class TestLineLine(GeometryFixtures):
    """
    Test constraints with signature `[Line, Line]`
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

    @pytest.fixture()
    def displacement(self):
        return np.random.rand(2)

    @pytest.fixture()
    def line_parallel(self, linea, displacement):
        return self.make_relative_line(linea, displacement, self.make_rotation(0))

    def test_Parallel(self, linea, line_parallel):
        res = co.Parallel()((linea, line_parallel), ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def line_orthogonal(self, linea, displacement):
        return self.make_relative_line(
            linea, displacement, self.make_rotation(np.pi / 2)
        )

    def test_Orthogonal(self, linea, line_orthogonal):
        res = co.Orthogonal()((linea, line_orthogonal), ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture(params=[False, True])
    def reverse(self, request):
        return request.param

    @pytest.fixture()
    def line_coincident(self, linea, reverse):
        if reverse:
            return self.make_line(linea['Point1'].value, -1*co.line_vector(linea))
        else:
            return linea

    def test_CoincidentLines(self, linea, line_coincident, reverse):
        res = co.CoincidentLines()((linea, line_coincident), (reverse,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def lineb(self):
        origin = np.random.rand(2)
        return self.make_line(origin, np.random.rand(2))

    @pytest.fixture()
    def relative_length(self, linea, lineb):
        lengtha = np.linalg.norm(co.line_vector(linea))
        lengthb = np.linalg.norm(co.line_vector(lineb))
        return lengtha/lengthb

    def test_RelativeLength(self, linea, lineb, relative_length):
        res = co.RelativeLength()((linea, lineb), (relative_length,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def angle(self):
        return np.random.rand()

    @pytest.fixture()
    def line_angle(self, linea, angle, displacement):
        scale = np.random.rand() * np.diag(np.ones(2))
        rotate = self.make_rotation(angle)
        return self.make_relative_line(linea, displacement, scale @ rotate)

    def test_Angle(self, linea, line_angle, angle):
        res = co.Angle()((linea, line_angle), (angle,))
        assert np.all(np.isclose(res, 0))

    def test_Collinear(self, linea):
        line_vec = co.line_vector(linea)

        lineb = self.make_relative_line(
            linea, np.random.rand()*line_vec, np.diag(np.ones(2))
        )
        res = co.Collinear()((linea, lineb), ())
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def midpoint_distance(self, linea, lineb):
        midpointa = 1/2*(linea['Point0'].value + linea['Point1'].value)
        midpointb = 1/2*(lineb['Point0'].value + lineb['Point1'].value)
        return midpointb - midpointa

    def test_MidpointXDistance(self, linea, lineb, midpoint_distance):
        res = co.MidpointXDistance()((linea, lineb), (midpoint_distance[0],))
        assert np.all(np.isclose(res, 0))

    def test_MidpointYDistance(self, linea, lineb, midpoint_distance):
        res = co.MidpointYDistance()((linea, lineb), (midpoint_distance[1],))
        assert np.all(np.isclose(res, 0))


class TestLineArray(GeometryFixtures):
    """
    Test constraints with signature `[Line, ...]`
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

    @pytest.fixture()
    def num_lines(self):
        return 5

    @pytest.fixture()
    def lines_collinear(self, linea, num_lines):
        line_vec = co.line_vector(linea)
        dists = np.random.rand(num_lines-1)
        return tuple(
            self.make_relative_line(linea, dist*line_vec, np.diag(np.ones(2)))
            for dist in dists
        )

    def test_CollinearArray(self, linea, lines_collinear):
        res = co.CollinearArray(1+len(lines_collinear))(
            (linea,) + lines_collinear, ()
        )
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_lengths(self, num_lines: int):
        return np.random.rand(num_lines)

    @pytest.fixture()
    def displacements(self, num_lines: int):
        return np.random.rand(num_lines, 2)

    @pytest.fixture()
    def lines_relative(self, linea, relative_lengths, displacements):
        num_lines = len(relative_lengths)+1
        scales = relative_lengths[:, None, None] * np.diag(np.ones(2))
        thetas = 2 * np.pi * np.random.rand(num_lines)
        rotates = [self.make_rotation(theta) for theta in thetas]

        return tuple(
            self.make_relative_line(linea, displacement, scale @ rotate)
            for displacement, scale, rotate in zip(displacements, scales, rotates)
        )

    def test_RelativeLengthArray(self, linea, lines_relative, relative_lengths):

        constraint = co.RelativeLengthArray(len(relative_lengths))
        res = constraint(tuple(lines_relative) + (linea,), (relative_lengths,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def line_pairs(self, num_lines):
        lineas = [self.make_line(np.random.rand(2), np.random.rand(2)) for _ in range(num_lines)]
        linebs = [self.make_line(np.random.rand(2), np.random.rand(2)) for _ in range(num_lines)]
        return tuple(
            (linea, lineb) for linea, lineb in zip(lineas, linebs)
        )

    @pytest.fixture()
    def midpoint_distances(self, line_pairs):
        def midpoint_distance(linea, lineb):
            mida = 1/2*(linea['Point0'].value + linea['Point1'].value)
            midb = 1/2*(lineb['Point0'].value + lineb['Point1'].value)
            return midb - mida
        return np.array([midpoint_distance(*line_pair) for line_pair in line_pairs])

    @pytest.fixture()
    def lines_midpointsarray(self, line_pairs):
        return tuple(itertools.chain.from_iterable(line_pairs))

    def test_XDistanceMidpointsArray(self, lines_midpointsarray, midpoint_distances):
        num_pairs = len(lines_midpointsarray) // 2
        res = co.MidpointXDistanceArray(num_pairs)(
            lines_midpointsarray, (midpoint_distances[:, 0],)
        )
        assert np.all(np.isclose(res, 0))


class TestPointLine(GeometryFixtures):
    """
    Test constraints with signature `[Point, Line]`
    """

    @pytest.fixture()
    def point(self):
        return self.make_point(np.random.rand(2))

    @pytest.fixture()
    def line_length(self):
        return np.random.rand()

    @pytest.fixture()
    def line_unit_vec(self):
        line_vec = np.random.rand(2)
        return line_vec/np.linalg.norm(line_vec)

    @pytest.fixture()
    def line(self, line_length, line_unit_vec):
        return self.make_line(np.random.rand(2), line_length*line_unit_vec)

    @pytest.fixture(params=[False, True])
    def reverse(self, request):
        return request.param

    def calc_point_on_line_distance(self, point, line):
        origin = line['Point0'].value
        point_vec = point.value - origin

        line_vec = co.line_vector(line)
        line_unit_vec = line_vec / np.linalg.norm(line_vec)
        return np.dot(point_vec, line_unit_vec)

    @pytest.fixture()
    def distance_on(self, point, line, line_unit_vec, reverse):
        if reverse:
            rev_line = pr.Line(value=[], prims=line.children[::-1])
            return self.calc_point_on_line_distance(point, rev_line)
        else:
            return self.calc_point_on_line_distance(point, line)

    def test_DistanceOnLine(self, point, line, distance_on, reverse):
        res = co.PointOnLineDistance()((point, line), (distance_on, reverse))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_distance_on(self, distance_on, line_length):
        return distance_on/line_length

    def test_RelativeDistanceOnLine(self, point, line, relative_distance_on, reverse):
        res = co.RelativePointOnLineDistance()((point, line), (relative_distance_on, reverse))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def distance_to(self, point, line, line_unit_vec, reverse):
        zsign = 1 if reverse else -1
        orth_unit_vec = np.cross(line_unit_vec, [0, 0, zsign])[:2]
        return np.dot(point.value - line['Point0'].value, orth_unit_vec)

    def test_DistanceToLine(self, point, line, distance_to, reverse):
        res = co.PointToLineDistance()((point, line), (distance_to, reverse))
        assert np.all(np.isclose(res, 0))


class TestQuadrilateral(GeometryFixtures):
    """
    Test constraints with signature `[Quadrilateral]`
    """

    @pytest.fixture()
    def quada(self):
        return self.make_quad(np.random.rand(2), np.random.rand(2, 2))

    @pytest.fixture()
    def aspect_ratio(self, quada):
        width = np.linalg.norm(co.line_vector(quada['Line0']))
        height = np.linalg.norm(co.line_vector(quada['Line1']))
        return width/height

    def test_AspectRatio(self, quada: pr.Quadrilateral, aspect_ratio: float):
        res = co.AspectRatio()((quada,), (aspect_ratio,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def quad_box(self):
        translation = np.random.rand(2)
        deformation = np.diag(np.random.rand(2))
        return self.make_quad(translation, deformation)

    def test_Box(self, quad_box: pr.Quadrilateral):
        res = co.Box()((quad_box,), ())
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
        res = co.OuterMargin(side=margin_side)((boxa, boxb), (outer_margin,))
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
        res = co.InnerMargin(side=margin_side)((boxa, boxb), (inner_margin,))
        assert np.all(np.isclose(res, 0))


class TestQuadrilateralArray(GeometryFixtures):
    """
    Test constraints with signature `[Quadrilateral, ...]`
    """
    # @pytest.fixture(params=[(2, 1)])
    @pytest.fixture(params=[(1, 1), (2, 1), (1, 2), (2, 2)])
    def grid_shape(self, request):
        return request.param

    @pytest.fixture()
    def grid_origin_quad_size(self):
        """The size, (width, height), of the origin quad (top left) in a grid"""
        width, height = np.random.rand(2)
        return (width, height)

    @pytest.fixture()
    def grid_parameters(self, grid_shape: tuple[int, int]):
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
    def quads_grid(
        self,
        grid_origin_quad_size: tuple[float, float],
        grid_parameters: dict[str, NDArray],
    ):
        origin = np.random.rand(2)
        origin = np.zeros(2)

        col_margins = grid_parameters["col_margins"]
        row_margins = grid_parameters["row_margins"]
        rel_col_widths = grid_parameters["col_widths"]
        rel_row_heights = grid_parameters["row_heights"]

        origin_width, origin_height = grid_origin_quad_size
        col_widths = origin_width * np.concatenate(([1], rel_col_widths))
        row_heights = origin_height * np.concatenate(([1], rel_row_heights))

        return self.make_quad_grid(
            origin, col_margins, row_margins, col_widths, row_heights
        )

    def test_RectilinearGrid(
        self, quads_grid: list[pr.Quadrilateral], grid_shape: tuple[int, int]
    ):
        res = co.RectilinearGrid(grid_shape)(quads_grid, ())
        assert np.all(np.isclose(res, 0))

    def test_Grid(
        self,
        quads_grid: list[pr.Quadrilateral],
        grid_shape: tuple[int, int],
        grid_parameters: tuple[NDArray, NDArray, NDArray, NDArray],
    ):
        root_prim = Node(None, {f'Quad{n}': quad for n, quad in enumerate(quads_grid)})

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.grid(which='both', axis='both')
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_aspect(1)
        ui.plot_prims(ax, root_prim)

        fig.savefig(f"quad_grid_{grid_shape}.png")

        res = co.Grid(grid_shape)(quads_grid, grid_parameters)
        assert np.all(np.isclose(res, 0))


from matplotlib import pyplot as plt
class TestAxesConstraints(GeometryFixtures):
    """
    Test constraints with signature `[Axes, ...]`
    """

    @pytest.fixture()
    def axes_size(self):
        return np.random.rand(2)

    @pytest.fixture(
        params=[
            {'bottom': True, 'top': False},
            {'bottom': False, 'top': True},
        ]
    )
    def xaxis_position(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {'left': True, 'right': False},
            {'left': False, 'right': True},
        ]
    )
    def yaxis_position(self, request):
        return request.param

    @pytest.fixture()
    def xlabel_position(self):
        return np.random.rand()

    @pytest.fixture()
    def ylabel_position(self):
        return np.random.rand()

    @pytest.fixture()
    def axes(
        self,
        axes_size: tuple[float, float],
        xaxis_position,
        yaxis_position,
        xaxis_height,
        yaxis_width,
        xlabel_position,
        ylabel_position
    ):
        axes_width, axes_height = axes_size
        scale = np.diag([axes_width, axes_height])
        frame = self.make_quad(np.array([0, 0]), scale)

        squash_width = np.array([[0, 0], [0, 1]])

        scale = np.diag([axes_width, xaxis_height])
        if xaxis_position['bottom']:
            xaxis = self.make_quad(np.array([0, -xaxis_height]), scale)
        elif xaxis_position['top']:
            xaxis = self.make_quad(np.array([0, axes_height]), scale)
        else:
            raise ValueError()

        scale = np.diag([yaxis_width, axes_height])
        if yaxis_position['left']:
            yaxis = self.make_quad(np.array([-yaxis_width, 0]), scale)
        elif yaxis_position['right']:
            yaxis = self.make_quad(np.array([axes_width, 0]), scale)
        else:
            raise ValueError()

        def point_from_arclength(line: pr.Line, s: float):
            origin = line['Point0'].value
            line_vector = co.line_vector(line)
            return origin + s*line_vector

        xlabel_anchor = self.make_point(
            [point_from_arclength(xaxis['Line0'], xlabel_position)[0], np.random.rand()]
        )
        ylabel_anchor = self.make_point(
            [np.random.rand(), point_from_arclength(yaxis['Line1'], ylabel_position)[1]]
        )

        return pr.Axes(
            prims=(frame, xaxis, xlabel_anchor, yaxis, ylabel_anchor),
            xaxis=True,
            yaxis=True
        )

    def test_PositionXAxis(self, axes, xaxis_position):
        res = co.PositionXAxis(**xaxis_position)((axes,), ())
        assert np.all(np.isclose(res, 0))

    def test_PositionYAxis(self, axes, yaxis_position):
        res = co.PositionYAxis(**yaxis_position)((axes,), ())

        assert np.all(np.isclose(res, 0))

    def test_PositionXAxisLabel(self, axes, xlabel_position):
        res = co.PositionXAxisLabel()((axes,), (xlabel_position,))
        assert np.all(np.isclose(res, 0))

    def test_PositionYAxisLabel(self, axes, ylabel_position):
        res = co.PositionYAxisLabel()((axes,), (ylabel_position,))

        assert np.all(np.isclose(res, 0))

    # NOTE: The `..AxisHeight` constraints have signature [Quadrilateral] but are
    # specialized to axes so I included them here rather than with `TestQuadrilateral`
    @pytest.fixture()
    def axes_mpl(
            self,
            axes_size: tuple[float, float],
            xaxis_position,
            yaxis_position,
        ):
        width, height = axes_size
        # NOTE: Unit dimensions of the figure are important because `fig.add_axes`
        # units are relative to figure dimensions
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_axes((0, 0, width, height))

        x = np.pi*2*np.linspace(0, 10)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_xlabel("My favourite xlabel")
        ax.set_ylabel("My favourite ylabel")

        if xaxis_position['bottom']:
            ax.xaxis.tick_bottom()
        elif xaxis_position['top']:
            ax.xaxis.tick_top()

        if yaxis_position['left']:
            ax.yaxis.tick_left()
        elif yaxis_position['right']:
            ax.yaxis.tick_right()

        return ax

    @pytest.fixture()
    def xaxis_height(self, axes_mpl):
        return co.XAxisHeight.get_xaxis_height(axes_mpl.xaxis)

    def test_XAxisHeight(self, axes, axes_mpl):
        res = co.XAxisHeight()((axes['XAxis'],), (axes_mpl.xaxis,))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def yaxis_width(self, axes_mpl):
        return co.YAxisWidth.get_yaxis_width(axes_mpl.yaxis)

    def test_YAxisWidth(self, axes, axes_mpl):
        res = co.YAxisWidth()((axes['YAxis'],), (axes_mpl.yaxis,))
        assert np.all(np.isclose(res, 0))

