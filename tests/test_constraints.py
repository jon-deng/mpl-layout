"""
Test geometric onstraints
"""

import pytest

from numpy.typing import NDArray

import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from mpllayout import primitives as pr
from mpllayout import constraints as co
from mpllayout import constructions as con
from mpllayout import ui
from mpllayout.containers import Node

from tests.fixture_primitives import GeometryFixtures


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
        res = constraint((pointa,), pointa.value)
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
        res = constraint((pointa, pointb), direction, distance)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def XYDistance(self, axis_name: str):
        if axis_name == 'x':
            return co.XDistance
        else:
            return co.YDistance

    @pytest.fixture()
    def xy_distance(self, distance, direction, axis_dir):
        return np.dot(distance*direction, axis_dir)

    def test_XYDistance(
        self, XYDistance, pointa: pr.Point, pointb: pr.Point, xy_distance: float
    ):
        res = XYDistance()((pointa, pointb), xy_distance)
        assert np.all(np.isclose(res, 0))

    def test_Coincident(
        self, pointa: pr.Point,
    ):
        res = co.Coincident()((pointa, pointa))
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
        res = co.Length()((linea,), length)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def XYLength(self, axis_name:str):
        if axis_name == 'x':
            return co.XLength
        else:
            return co.YLength

    @pytest.fixture()
    def xy_length(self, length: float, direction: NDArray, axis_dir: NDArray):
        return np.dot(length * direction, axis_dir)

    def test_XYLength(self, XYLength, linea, xy_length):
        res = XYLength()((linea,), xy_length)
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
        res = co.Parallel()((linea, line_parallel))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def line_orthogonal(self, linea, displacement):
        return self.make_relative_line(
            linea, displacement, self.make_rotation(np.pi / 2)
        )

    def test_Orthogonal(self, linea, line_orthogonal):
        res = co.Orthogonal()((linea, line_orthogonal))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture(params=[False, True])
    def reverse(self, request):
        return request.param

    @pytest.fixture()
    def line_coincident(self, linea, reverse):
        if reverse:
            return self.make_line(linea['Point1'].value, -1*con.LineVector.assem((linea,)))
        else:
            return linea

    def test_CoincidentLines(self, linea, line_coincident, reverse):
        res = co.CoincidentLines()((linea, line_coincident), reverse)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def lineb(self):
        origin = np.random.rand(2)
        return self.make_line(origin, np.random.rand(2))

    @pytest.fixture()
    def relative_length(self, linea, lineb):
        lengtha = np.linalg.norm(con.LineVector.assem((linea,)))
        lengthb = np.linalg.norm(con.LineVector.assem((lineb,)))
        return lengtha/lengthb

    def test_RelativeLength(self, linea, lineb, relative_length):
        res = co.RelativeLength()((linea, lineb), relative_length)
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
        res = co.Angle()((linea, line_angle), angle)
        assert np.all(np.isclose(res, 0))

    def test_Collinear(self, linea):
        line_vec = con.LineVector.assem((linea,))

        lineb = self.make_relative_line(
            linea, np.random.rand()*line_vec, np.diag(np.ones(2))
        )
        res = co.Collinear()((linea, lineb))
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def MidpointXYDistance(self, axis_name: str):
        if axis_name == 'x':
            return co.MidpointXDistance
        else:
            return co.MidpointYDistance

    @pytest.fixture()
    def midpoint_xy_distance(self, linea, lineb, axis_dir: NDArray):
        midpointa = 1/2*(linea['Point0'].value + linea['Point1'].value)
        midpointb = 1/2*(lineb['Point0'].value + lineb['Point1'].value)
        return np.dot(midpointb - midpointa, axis_dir)

    def test_MidpointXYDistance(self, MidpointXYDistance, linea, lineb, midpoint_xy_distance):
        res = MidpointXYDistance()((linea, lineb), midpoint_xy_distance)
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
        line_vec = con.LineVector.assem((linea,))
        dists = np.random.rand(num_lines-1)
        return tuple(
            self.make_relative_line(linea, dist*line_vec, np.diag(np.ones(2)))
            for dist in dists
        )

    def test_CollinearArray(self, linea, lines_collinear):
        CollinearArray = con.transform_MapType(co.Collinear, (1+len(lines_collinear))*(pr.Line,))
        res = CollinearArray()((linea,) + lines_collinear)
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

        constraint = con.transform_map(
            co.RelativeLength(), (len(lines_relative)+1)*(pr.Line,)
        )
        res = constraint(tuple(lines_relative) + (linea,), *relative_lengths)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def line_pairs(self, num_lines):
        lineas = [self.make_line(np.random.rand(2), np.random.rand(2)) for _ in range(num_lines)]
        linebs = [self.make_line(np.random.rand(2), np.random.rand(2)) for _ in range(num_lines)]
        return tuple(
            (linea, lineb) for linea, lineb in zip(lineas, linebs)
        )


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

        line_vec = con.LineVector.assem((line,))
        line_unit_vec = line_vec / np.linalg.norm(line_vec)
        return np.dot(point_vec, line_unit_vec)

    @pytest.fixture()
    def distance_on(self, point, line, line_unit_vec, reverse):
        if reverse:
            rev_line = pr.Line(value=[], prims=line[::-1])
            return self.calc_point_on_line_distance(point, rev_line)
        else:
            return self.calc_point_on_line_distance(point, line)

    def test_DistanceOnLine(self, point, line, distance_on, reverse):
        res = co.PointOnLineDistance()((point, line), reverse, distance_on)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def relative_distance_on(self, distance_on, line_length):
        return distance_on/line_length

    def test_RelativeDistanceOnLine(self, point, line, relative_distance_on, reverse):
        res = co.RelativePointOnLineDistance()((point, line), reverse, relative_distance_on)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def distance_to(self, point, line, line_unit_vec, reverse):
        zsign = 1 if reverse else -1
        orth_unit_vec = np.cross(line_unit_vec, [0, 0, zsign])[:2]
        return np.dot(point.value - line['Point0'].value, orth_unit_vec)

    def test_DistanceToLine(self, point, line, distance_to, reverse):
        res = co.PointToLineDistance()((point, line), reverse, distance_to)
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
        width = np.linalg.norm(con.LineVector.assem((quada['Line0'],)))
        height = np.linalg.norm(con.LineVector.assem((quada['Line1'],)))
        return width/height

    def test_AspectRatio(self, quada: pr.Quadrilateral, aspect_ratio: float):
        res = co.AspectRatio()((quada,), aspect_ratio)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def width(self):
        return np.random.rand()

    @pytest.fixture()
    def height(self):
        return np.random.rand()

    @pytest.fixture()
    def quad_rectangle(self, width, height):
        translation = np.random.rand(2)
        deformation = np.diag([width, height])
        return self.make_quad(translation, deformation)

    def test_Box(self, quad_rectangle: pr.Quadrilateral):
        res = co.Box()((quad_rectangle,))
        assert np.all(np.isclose(res, 0))

    def test_Width(self, quad_rectangle: pr.Quadrilateral, width: float):
        res = co.Width()((quad_rectangle,), width)
        assert np.all(np.isclose(res, 0))

    def test_Height(self, quad_rectangle: pr.Quadrilateral, height: float):
        res = co.Height()((quad_rectangle,), height)
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
        res = co.OuterMargin(side=margin_side)((boxa, boxb), outer_margin)
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
        res = co.InnerMargin(side=margin_side)((boxa, boxb), inner_margin)
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

        return (rel_col_widths, rel_row_heights, col_margins, row_margins)

    @pytest.fixture()
    def quads_grid(
        self,
        grid_origin_quad_size: tuple[float, float],
        grid_parameters: dict[str, NDArray],
    ):
        origin = np.random.rand(2)
        origin = np.zeros(2)

        rel_col_widths = grid_parameters[0]
        rel_row_heights = grid_parameters[1]
        col_margins = grid_parameters[2]
        row_margins = grid_parameters[3]

        origin_width, origin_height = grid_origin_quad_size
        col_widths = origin_width * np.concatenate(([1], rel_col_widths))
        row_heights = origin_height * np.concatenate(([1], rel_row_heights))

        return self.make_quad_grid(
            origin, col_margins, row_margins, col_widths, row_heights
        )

    def test_RectilinearGrid(
        self, quads_grid: list[pr.Quadrilateral], grid_shape: tuple[int, int]
    ):
        res = co.RectilinearGrid(grid_shape)(quads_grid)
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

        res = co.Grid(grid_shape)(quads_grid, *grid_parameters)
        assert np.all(np.isclose(res, 0))


from matplotlib import pyplot as plt
class TestAxesConstraints(GeometryFixtures):
    """
    Test constraints with signature `[Axes, ...]`
    """

    @pytest.fixture()
    def axes_size(self):
        return np.random.rand(2)

    @pytest.fixture(params=(False, True))
    def twinx(self, request):
        return request.param

    @pytest.fixture(params=(False, True))
    def twiny(self, request):
        return request.param

    @pytest.fixture(
        params=['bottom', 'top']
    )
    def xaxis_side(self, request):
        return request.param

    @pytest.fixture(
        params=['left', 'right']
    )
    def yaxis_side(self, request):
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
        xaxis_side,
        yaxis_side,
        xaxis_height,
        yaxis_width,
        xlabel_position,
        ylabel_position
    ):
        axes_width, axes_height = axes_size
        scale = np.diag([axes_width, axes_height])
        frame = self.make_quad(np.array([0, 0]), scale)

        squash_width = np.array([[0, 0], [0, 1]])

        def make_quad_on_hor_side(bottom: bool):
            scale = np.diag([axes_width, xaxis_height])
            if bottom:
                return self.make_quad(np.array([0, -xaxis_height]), scale)
            else:
                return self.make_quad(np.array([0, axes_height]), scale)

        if xaxis_side == 'bottom':
            xbottom = True
        else:
            xbottom = False

        xaxis = make_quad_on_hor_side(xbottom)
        twin_xaxis = make_quad_on_hor_side(not xbottom)

        def make_quad_on_ver_side(left: bool):
            scale = np.diag([yaxis_width, axes_height])
            if left:
                return self.make_quad(np.array([-yaxis_width, 0]), scale)
            else:
                return self.make_quad(np.array([axes_width, 0]), scale)

        if yaxis_side == 'left':
            yleft = True
        else:
            yleft = False

        yaxis = make_quad_on_ver_side(yleft)
        twin_yaxis = make_quad_on_ver_side(not yleft)

        def point_from_arclength(line: pr.Line, s: float):
            origin = line['Point0'].value
            return origin + s*con.LineVector.assem((line,))

        xlabel_anchor = self.make_point(
            [point_from_arclength(xaxis['Line0'], xlabel_position)[0], np.random.rand()]
        )
        ylabel_anchor = self.make_point(
            [np.random.rand(), point_from_arclength(yaxis['Line1'], ylabel_position)[1]]
        )

        twin_xlabel_anchor = self.make_point(
            [point_from_arclength(xaxis['Line0'], xlabel_position)[0], np.random.rand()]
        )
        twin_ylabel_anchor = self.make_point(
            [np.random.rand(), point_from_arclength(yaxis['Line1'], ylabel_position)[1]]
        )

        xaxis_prims = (xaxis, xlabel_anchor)
        yaxis_prims = (yaxis, ylabel_anchor)
        twin_xaxis_prims = (twin_xaxis, twin_xlabel_anchor)
        twin_yaxis_prims = (twin_yaxis, twin_ylabel_anchor)

        prims = (frame, *xaxis_prims, *yaxis_prims, *twin_xaxis_prims, *twin_yaxis_prims)

        return pr.Axes(
            prims=prims, xaxis=True, yaxis=True, twinx=True, twiny=True
        )

    def test_PositionXAxis(self, axes, xaxis_side, twinx):
        res = co.PositionXAxis(side=xaxis_side, twin=twinx)((axes,))
        assert np.all(np.isclose(res, 0))

    def test_PositionYAxis(self, axes, yaxis_side):
        res = co.PositionYAxis(side=yaxis_side)((axes,))

        assert np.all(np.isclose(res, 0))

    def test_PositionXAxisLabel(self, axes, xlabel_position, twinx):
        res = co.PositionXAxisLabel(twin=twinx)((axes,), xlabel_position)
        assert np.all(np.isclose(res, 0))

    def test_PositionYAxisLabel(self, axes, ylabel_position, twiny):
        res = co.PositionYAxisLabel(twin=twiny)((axes,), ylabel_position)

        assert np.all(np.isclose(res, 0))

    # NOTE: The `..AxisThickness` constraints have signature [Quadrilateral] but are
    # specialized to axes so I included them here rather than with `TestQuadrilateral`
    @pytest.fixture()
    def axes_mpl(
            self,
            axes_size: tuple[float, float],
            xaxis_side,
            yaxis_side,
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

        if xaxis_side == 'bottom':
            ax.xaxis.tick_bottom()
        elif xaxis_side == 'top':
            ax.xaxis.tick_top()

        if yaxis_side == 'left':
            ax.yaxis.tick_left()
        elif yaxis_side == 'right':
            ax.yaxis.tick_right()

        return ax

    @pytest.fixture()
    def xaxis_height(self, axes_mpl):
        return co.XAxisThickness.get_axis_thickness(axes_mpl.xaxis)

    def test_XAxisThickness(self, axes, axes_mpl):
        res = co.XAxisThickness()((axes['XAxis'],), axes_mpl.xaxis)
        assert np.all(np.isclose(res, 0))

    @pytest.fixture()
    def yaxis_width(self, axes_mpl):
        return co.YAxisThickness.get_axis_thickness(axes_mpl.yaxis)

    def test_YAxisThickness(self, axes, axes_mpl):
        res = co.YAxisThickness()((axes['YAxis'],), axes_mpl.yaxis)
        assert np.all(np.isclose(res, 0))

