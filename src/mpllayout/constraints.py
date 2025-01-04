"""
Geometric constraints
"""

from typing import Optional, Any, Literal
from matplotlib.axis import XAxis, YAxis
from numpy.typing import NDArray

import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from . import constructions as con

PrimKeys = con.PrimKeys
Params = con.Params

PrimKeysNode = con.PrimKeysNode
ParamsNode = con.ParamsNode

Constraint = con.Construction
ConstraintNode = con.ConstructionNode
ArrayConstraint = con.ArrayCompoundConstruction

## Point constraints

# Argument type: tuple[Point,]

Fix = con.transform_ConstraintType(con.Coordinate)

# Argument type: tuple[Point, Point]

DirectedDistance = con.transform_ConstraintType(con.DirectedDistance)

XDistance = con.transform_ConstraintType(con.XDistance)

YDistance = con.transform_ConstraintType(con.YDistance)

class Coincident(con.LeafConstruction, con._PointPointSignature):
    """
    Constrain two points to be coincident

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Point]):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return con.Coordinate.assem((point1,)) - con.Coordinate.assem((point0,))


## Line constraints

# Argument type: tuple[Line,]

Length = con.transform_ConstraintType(con.Length)

DirectedLength = con.transform_ConstraintType(con.DirectedLength)

XLength = con.transform_ConstraintType(con.XLength)

YLength = con.transform_ConstraintType(con.YLength)


class Vertical(con.LeafConstruction, con._LineSignature):
    """
    Constrain a line to be vertical

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(con.LineVector.assem(prims), np.array([1, 0]))


class Horizontal(con.LeafConstruction, con._LineSignature):
    """
    Constrain a line to be horizontal

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(con.LineVector.assem(prims), np.array([0, 1]))


# Argument type: tuple[Line, Line]

class RelativeLength(con.ConstructionNode):
    """
    Constrain the length of a line relative to another line

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines

        The length of the first line is measured relative to the second line
    length: float
        The relative length
    """

    def __new__(cls):
        return con.transform_sum(
            con.Length(),
            con.transform_scalar_mul(
                con.transform_scalar_mul(con.Length(), -1), con.Scalar()
            )
        )

MidpointXDistance = con.transform_ConstraintType(con.MidpointXDistance)

MidpointYDistance = con.transform_ConstraintType(con.MidpointYDistance)

class Orthogonal(con.LeafConstruction, con._LineLineSignature):
    """
    Constrain two lines to be orthogonal

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        """
        Return the orthogonal error between two lines
        """
        line0, line1 = prims
        return jnp.dot(
            con.LineVector.assem((line0,)), con.LineVector.assem((line1,))
        )


class Parallel(con.LeafConstruction, con._LineLineSignature):
    """
    Return the parallel error between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        """
        Return the parallel error between two lines
        """
        line0, line1 = prims
        return jnp.cross(
            con.LineVector.assem((line0,)), con.LineVector.assem((line1,))
        )


Angle = con.transform_ConstraintType(con.Angle)


class Collinear(con.LeafConstruction, con._LineLineSignature):
    """
    Return the collinear error between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(self, prims: tuple[pr.Line, pr.Line]):
        """
        Return the collinearity error between two lines
        """
        line0, line1 = prims
        line2 = pr.Line(prims=(line1[0], line0[0]))

        return jnp.array(
            [Parallel.assem((line0, line1)), Parallel.assem((line0, line2))]
        )


class CoincidentLines(con.LeafConstruction, con._LineLineSignature):
    """
    Return coincident error between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    reverse: bool
        A boolean indicating whether lines are reversed
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2, (bool,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], reverse: bool):
        """
        Return the coincident error between two lines
        """
        line0, line1 = prims
        if reverse:
            point0_err = Coincident.assem((line1['Point0'], line0['Point1']))
            point1_err = Coincident.assem((line1['Point1'], line0['Point0']))
        else:
            point0_err = Coincident.assem((line1['Point0'], line0['Point0']))
            point1_err = Coincident.assem((line1['Point1'], line0['Point1']))
        return jnp.concatenate([point0_err, point1_err])

# Argument type: tuple[Line, ...]

## Point and Line constraints

# TODO: class BoundPointsByLine(DynamicConstraint)
# A class that ensures all points have orthogonal distance to a line > offset
# The orthogonal direction should be rotated 90 degrees from the line direction
# (or some other convention)
# You should also have some convention to specify whether you bound for positive
# distance or negative distance
# This would be like saying the points all lie to the left or right of the line
# + and offset
# This would be useful for aligning axis labels

PointOnLineDistance = con.transform_ConstraintType(con.PointOnLineDistance)


PointToLineDistance = con.transform_ConstraintType(con.PointToLineDistance)


class RelativePointOnLineDistance(con.LeafConstruction, con._PointLineSignature):
    """
    Constrain the projected distance of a point along a line

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Line]
        The point and line
    reverse: bool
        A boolean indicating whether to reverse the line direction

        The distance of the point on the line is measured either from the start or end
        point of the line based on `reverse`. If `reverse=False` then the start point is
        used.
    distance: float
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1, (bool, float))

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Point, pr.Line],
        reverse: bool,
        distance: float
    ):
        """
        Return the projected distance error of a point along a line
        """
        point, line = prims
        if reverse:
            origin = con.Coordinate.assem((line['Point1'],))
            unit_vec = -con.UnitLineVector.assem((line,))
        else:
            origin = con.Coordinate.assem((line['Point0'],))
            unit_vec = con.UnitLineVector.assem((line,))
        line_length = con.Length.assem((line,))

        proj_dist = jnp.dot(point.value-origin, unit_vec)
        return jnp.array([proj_dist - distance*line_length])


## Quad constraints

# Argument type: tuple[Quadrilateral]

class Box(con.StaticCompoundConstruction, con._QuadrilateralSignature):
    """
    Constrain a quadrilateral to be rectangular

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The quad
    """

    @classmethod
    def init_children(cls):
        keys = (
            "HorizontalBottom", "HorizontalTop", "VerticalLeft", "VerticalRight"
        )
        constraints = (Horizontal(), Horizontal(), Vertical(), Vertical())
        prim_keys = (
            ("arg0/Line0",), ("arg0/Line2",), ("arg0/Line3",), ("arg0/Line1",)
        )
        def child_params(params: Params) -> tuple[Params, ...]:
            return ((), (), (), ())

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)


AspectRatio = con.transform_ConstraintType(con.AspectRatio)


def get_axis_dim(axis: XAxis | YAxis, side: str):

    # Ignore the axis label in the height by temporarily making it invisible
    label_visibility = axis.label.get_visible()
    axis.label.set_visible(False)

    axis_bbox = axis.get_tightbbox()

    if axis_bbox is None:
        dim = 0
    else:
        axis_bbox = axis_bbox.transformed(axis.axes.figure.transFigure.inverted())
        fig_width, fig_height = axis.axes.figure.get_size_inches()
        axes_bbox = axis.axes.get_position()

        if axis.get_ticks_position() == "bottom":
            dim = fig_height * (axes_bbox.ymin - axis_bbox.ymin)
        elif axis.get_ticks_position() == "top":
            dim = fig_height * (axis_bbox.ymax - axes_bbox.ymax)
        elif axis.get_ticks_position() == "left":
            dim = fig_width * (axes_bbox.xmin - axis_bbox.xmin)
        elif axis.get_ticks_position() == "right":
            dim = fig_width * (axis_bbox.xmax - axes_bbox.xmax)
        else:
            raise ValueError()

    axis.label.set_visible(label_visibility)

    return dim

class XAxisHeight(con.StaticCompoundConstruction, con._QuadrilateralSignature):
    """
    Return the x-axis height for an axes

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The axes
    axis: XAxis
        The XAxis
    """

    @staticmethod
    def get_xaxis_height(axis: XAxis):
        return get_axis_dim(axis, axis.get_ticks_position())

    @classmethod
    def init_children(cls):
        keys = ("Height",)
        constraints = (YDistance(),)
        prim_keys = (("arg0/Line1/Point0", "arg0/Line1/Point1"),)

        def child_params(params: Params) -> tuple[Params, ...]:
            xaxis: XAxis | None = params[0]
            if xaxis is None:
                return ((0,),)
            else:
                return ((cls.get_xaxis_height(xaxis),),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0, (XAxis,))


class YAxisWidth(con.StaticCompoundConstruction, con._QuadrilateralSignature):
    """
    Constrain the y-axis width for an axes

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The axes
    axis: YAxis
        The YAxis
    """

    @staticmethod
    def get_yaxis_width(axis: YAxis):
        return get_axis_dim(axis, axis.get_ticks_position())

    @classmethod
    def init_children(cls):
        keys = ("Width",)
        constraints = (XDistance(),)
        prim_keys = (("arg0/Line0/Point0", "arg0/Line0/Point1"),)

        def child_params(params: Params) -> tuple[Params, ...]:
            yaxis: YAxis | None = params[0]
            if yaxis is None:
                return ((0,),)
            else:
                return ((cls.get_yaxis_width(yaxis),),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0, (YAxis,))


# Argument type: tuple[Quadrilateral, Quadrilateral]

OuterMargin = con.transform_ConstraintType(con.OuterMargin)

InnerMargin = con.transform_ConstraintType(con.InnerMargin)

class AlignRow(con.StaticCompoundConstruction, con._QuadrilateralQuadrilateralSignature):
    """
    Constrain two quadrilaterals to lie in a row

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quadrilaterals
    """

    @classmethod
    def init_children(cls):
        keys = ("AlignBottom", "AlignTop")
        constraints = (Collinear(), Collinear())
        prim_keys = (("arg0/Line0", "arg1/Line0"), ("arg0/Line2", "arg1/Line2"))

        def child_params(params: Params) -> tuple[Params, ...]:
            return 2*((),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)

class AlignColumn(con.StaticCompoundConstruction, con._QuadrilateralQuadrilateralSignature):
    """
    Constrain two quadrilaterals to lie in a column

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quadrilaterals
    """

    @classmethod
    def init_children(cls):
        keys = ("AlignLeft", "AlignRight")
        constraints = (Collinear(), Collinear())
        prim_keys = (("arg0/Line3", "arg1/Line3"), ("arg0/Line1", "arg1/Line1"))

        def child_params(params: Params) -> tuple[Params, ...]:
            return 2*((),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)

class AlignOutside(con.StaticCompoundConstruction, con._QuadrilateralQuadrilateralSignature):
    """
    Constrain the outside sides of two quadrilaterals to coincide

    The behaviour depends on the `side` keyword argument. If side is 'left', the
    left side of the first quad is coincident with the right side of the second
    quad. If side is 'bottom', the bottom side of the first quad is coincident
    with the top side of the second quad. Behaviour for the 'top' and 'right'
    follows the same pattern.

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quadrilaterals
    """

    @classmethod
    def init_children(cls, side=Literal['bottom', 'top', 'left', 'right']):
        keys = ("CoincidentLines",)
        constraints = (CoincidentLines(),)
        if side == 'bottom':
            prim_keys = (('arg0/Line0', f'arg1/Line2'),)
        elif side == 'top':
            prim_keys = (('arg0/Line2', f'arg1/Line0'),)
        elif side == 'left':
            prim_keys = (('arg0/Line3', f'arg1/Line1'),)
        elif side == 'right':
            prim_keys = (('arg0/Line1', f'arg1/Line3'),)
        else:
            raise ValueError(
                "`side` must be one of 'bottom', 'top', 'left', or 'right'"
            )

        def child_params(params: Params) -> tuple[Params, ...]:
            return ((True,),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)

# Argument type: tuple[Quadrilateral, ...]

def idx_1d(multi_idx: tuple[int, ...], shape: tuple[int, ...]):
    """
    Return a 1D array index from a multi-dimensional array index
    """
    strides = shape[1:] + (1,)
    return sum(axis_idx * stride for axis_idx, stride in zip(multi_idx, strides))

class RectilinearGrid(ArrayConstraint, con._QuadrilateralsSignature):
    """
    Constrain a set of quads to lie on a rectilinear grid

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, ...]
        The quadrilaterals
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        # size = np.prod(shape)
        num_row, num_col = shape

        def idx(i, j):
            return idx_1d((i, j), shape)

        # There are 2 child constraints that line up all rows and all columns

        keys = tuple(
            [f"AlignRow{nrow}" for nrow in range(num_row)]
            + [f"AlignColumn{ncol}" for ncol in range(num_col)]
        )

        align_rows = num_row * [
            con.transform_map(AlignRow(), num_col*(pr.Quadrilateral,))
        ]
        align_cols = num_col * [
            con.transform_map(AlignColumn(), num_row*(pr.Quadrilateral,)),
        ]
        constraints = tuple(align_rows + align_cols)

        align_row_args = [
            tuple(f"arg{idx(nrow, ncol)}" for ncol in range(num_col))
            for nrow in range(num_row)
        ]
        align_col_args = [
            tuple(f"arg{idx(nrow, ncol)}" for nrow in range(num_row))
            for ncol in range(num_col)
        ]
        prim_keys = tuple(align_row_args + align_col_args)

        def child_params(params: Params) -> tuple[Params, ...]:
            return 4 * num_row*num_col * ((),)
        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls, shape: tuple[int, ...]):
        prim_types = np.prod(shape) * (pr.Quadrilateral,)
        param_types = ()
        value_size = 0
        return con.ConstructionSignature(prim_types, param_types, value_size)


class Grid(ArrayConstraint, con._QuadrilateralsSignature):
    """
    Constrain a set of quads to lie on a dimensioned rectilinear grid

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, ...]
        The quadrilaterals
    col_widths: NDArray
        Column widths (from left to right) relative to the left-most column
    row_heights: NDArray
        Row height (from top to bottom) relative to the top-most row
    col_margins: NDArray
        Absolute column margins (from left to right)
    row_margins: NDArray
        Absolute row margins (from top to bottom)
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        num_row, num_col = shape
        size = np.prod(shape)

        # Children constraints do:
        # 1. Align all quads in a grid
        # 2. Set relative column widths relative to column 0
        # 3. Set relative row heights relative to row 0
        # 4. Set margins between columns
        # 5. Set margins between rows
        keys = (
            "RectilinearGrid", "ColWidths", "RowHeights", "ColMargins", "RowMargins"
        )
        constraints = (
            RectilinearGrid(shape=shape),
            con.transform_map(RelativeLength(), num_col*(pr.Line,)),
            con.transform_map(RelativeLength(), num_row*(pr.Line,)),
            con.transform_map(OuterMargin(side='right'), num_col*(pr.Quadrilateral,)),
            con.transform_map(OuterMargin(side='bottom'), num_row*(pr.Quadrilateral,))
        )

        def idx(i, j):
            return idx_1d((i, j), shape)
        rows = list(range(shape[0]))
        cols = list(range(shape[1]))

        rect_grid_args = tuple(f"arg{n}" for n in range(size))

        col_width_args = tuple(
            f"arg{idx(0, col)}/Line0" for col in cols[1:] + cols[:1]
        )
        row_height_args = tuple(
            f"arg{idx(row, 0)}/Line1" for row in rows[1:] + rows[:1]
        )

        col_margin_args = tuple(f"arg{idx(0, col)}" for col in cols)
        row_margin_args = tuple(f"arg{idx(row, 0)}" for row in rows)

        prim_keys = (
            rect_grid_args,
            col_width_args,
            row_height_args,
            col_margin_args,
            row_margin_args,
        )

        def child_params(params: Params) -> tuple[Params, ...]:
            col_widths, row_heights, col_margins, row_margins = params
            return ((), tuple(col_widths), tuple(row_heights), tuple(col_margins), tuple(row_margins))

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls, shape: tuple[int, ...]):
        prim_types = np.prod(shape) * (pr.Quadrilateral,)
        param_types = (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        value_size = 0
        return con.ConstructionSignature(prim_types, param_types, value_size)


## Axes constraints

# Argument type: tuple[Axes]

# TODO: Handle more specialized x/y axes combos (i.e. twin x/y axes)
# The below axis constraints are made for a single x and y axis

class PositionXAxis(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the x-axis to the top or bottom of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    def __init__(
        self,
        side: Literal['bottom', 'top']='bottom',
        twinx: bool=False
    ):
        return super().__init__(side=side, twinx=twinx)

    @classmethod
    def init_children(cls,
        side: Literal['bottom', 'top']='bottom',
        twinx: bool=False
    ):
        if side not in {'bottom', 'top'}:
            raise ValueError("`side` must be 'bottom' or 'top'")

        if side == 'bottom':
            bottom = True
        else:
            bottom = False

        def coincident_line_keys(bottom: bool, twin: bool=False):
            if twin:
                twin_prefix = 'Twin'
            else:
                twin_prefix = ''

            if bottom:
                return (('arg0/Frame/Line0', f'arg0/{twin_prefix}XAxis/Line2'),)
            else:
                return (('arg0/Frame/Line2', f'arg0/{twin_prefix}XAxis/Line0'),)

        keys = ('CoincidentLines',)
        constraints = (CoincidentLines(),)
        prim_keys = coincident_line_keys(bottom)

        if twinx:
            keys = keys + ('TwinCoincidentLines',)
            constraints = constraints + (CoincidentLines(),)
            prim_keys = prim_keys + coincident_line_keys(not bottom, twin=True)

        def child_params(params: Params) -> tuple[Params, ...]:
            child_params = ((True,),)
            if twinx:
                child_params = child_params + ((True,),)
            return child_params

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(
        cls,
        side: Literal['bottom', 'top']='bottom',
        twinx: bool=False
    ):
        return cls.make_signature(0)


class PositionYAxis(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the y-axis to the left or right of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    def __init__(
        self,
        side: Literal['left', 'right']='left',
        twiny: bool=False
    ):
        return super().__init__(side=side, twiny=twiny)

    @classmethod
    def init_children(
        cls,
        side: Literal['left', 'right']='left',
        twiny: bool=False
    ):

        if side not in {'left', 'right'}:
            raise ValueError("`side` must be 'left' or 'right'")

        if side == 'left':
            left = True
        else:
            left = False

        def coincident_line_keys(left: bool, twin: bool=False):
            if twin:
                twin_prefix = 'Twin'
            else:
                twin_prefix = ''

            if left:
                return (('arg0/Frame/Line3', f'arg0/{twin_prefix}YAxis/Line1'),)
            else:
                return (('arg0/Frame/Line1', f'arg0/{twin_prefix}YAxis/Line3'),)

        keys = ('CoincidentLines',)
        constraints = (CoincidentLines(),)
        prim_keys = coincident_line_keys(left)

        if twiny:
            keys = keys + ('TwinCoincidentLines',)
            constraints = constraints + (CoincidentLines(),)
            prim_keys = prim_keys + coincident_line_keys(not left, twin=True)

        def child_params(params: Params) -> tuple[Params, ...]:
            child_params = ((True,),)
            if twiny:
                child_params = child_params + ((True,),)
            return child_params

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(
        cls,
        side: Literal['left', 'right']='left',
        twiny: bool=False
    ):
        return cls.make_signature(0)


class PositionXAxisLabel(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the x-axis label horizontal distance (left to right) relative to axes width

    Parameters
    ----------
    prims: tuple[pr.AxesX | pr.Axes]
        The axes
    distance: float
        The axes fraction from the left to position the label
    """

    @classmethod
    def init_children(cls):

        keys = ('RelativePointOnLineDistance',)
        constraints = (RelativePointOnLineDistance(),)
        prim_keys = (('arg0/XAxisLabel', 'arg0/XAxis/Line0'),)

        def child_params(params: Params) -> tuple[Params, ...]:
            distance, = params
            return ((False, distance),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)


class PositionYAxisLabel(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the y-axis label vertical distance (bottom to top) relative to axes height

    Parameters
    ----------
    prims: tuple[pr.AxesX | pr.Axes]
        The axes
    distance: float
        The axes fraction from the bottom to position the label
    """

    @classmethod
    def init_children(cls):
        keys = ('RelativePointOnLineDistance',)
        constraints = (RelativePointOnLineDistance(),)
        prim_keys = (('arg0/YAxisLabel', 'arg0/YAxis/Line1'),)

        def child_params(params: Params) -> tuple[Params, ...]:
            distance, = params
            return ((False, distance),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls):
        return cls.make_signature(0)
