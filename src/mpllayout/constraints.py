"""
Geometric constraints
"""

from typing import Optional, Any
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

Fix = con.generate_constraint(con.Coordinate)

# Argument type: tuple[Point, Point]

DirectedDistance = con.generate_constraint(con.DirectedDistance)

XDistance = con.generate_constraint(con.XDistance)

YDistance = con.generate_constraint(con.YDistance)

class Coincident(con.LeafConstruction, con._PointPointSignature):
    """
    Constrain two points to be coincident

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points
    """

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(2)

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Point]):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return con.Coordinate.assem((point1,)) - con.Coordinate.assem((point0,))


## Line constraints

# Argument type: tuple[Line,]

Length = con.generate_constraint(con.Length)

DirectedLength = con.generate_constraint(con.DirectedLength)

XLength = con.generate_constraint(con.XLength)

YLength = con.generate_constraint(con.YLength)


class Vertical(con.LeafConstruction, con._LineSignature):
    """
    Constrain a line to be vertical

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(1)

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
    def init_aux_data(cls):
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(con.LineVector.assem(prims), np.array([0, 1]))


# Argument type: tuple[Line, Line]

class RelativeLength(con.LeafConstruction, con._LineLineSignature):
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

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(1, (float,))

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], length: float):
        """
        Return the length error of line `prims[0]` relative to line `prims[1]`
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = con.LineVector.assem((line0,))
        vec_b = con.LineVector.assem((line1,))
        return jnp.sum(vec_a**2) - length**2 * jnp.sum(vec_b**2)

MidpointXDistance = con.generate_constraint(con.MidpointXDistance)

MidpointYDistance = con.generate_constraint(con.MidpointYDistance)

class Orthogonal(con.LeafConstruction, con._LineLineSignature):
    """
    Constrain two lines to be orthogonal

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(1)

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
    def init_aux_data(cls):
        return cls.aux_data(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line]):
        """
        Return the parallel error between two lines
        """
        line0, line1 = prims
        return jnp.cross(
            con.LineVector.assem((line0,)), con.LineVector.assem((line1,))
        )


Angle = con.generate_constraint(con.Angle)


class Collinear(con.LeafConstruction, con._LineLineSignature):
    """
    Return the collinear error between two lines

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(2)

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
    def init_aux_data(cls):
        return cls.aux_data(2, (bool,))

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

class RelativeLengthArray(ArrayConstraint, con._LinesSignature):
    """
    Constrain the lengths of a set of lines relative to the last

    Parameters
    ----------
    prims: tuple[pr.Line, ...]
        The lines

        The length of the lines are measured relative to the last line
    lengths: NDArray
        The relative lengths
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        size = np.prod(shape)

        child_keys = tuple(f"RelativeLength{n}" for n in range(size))
        child_constraint_types = size * (RelativeLength,)
        child_constraint_kwargs = size * ({},)
        child_prim_keys = tuple((f"arg{n}", f"arg{size}") for n in range(size))

        def child_params(parameters):
            lengths, = parameters
            return tuple(
                (length,) for length in lengths
            )
        return child_keys, child_constraint_types, child_constraint_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Line,) + (pr.Line,),
            'RES_PARAMS_TYPE': size * (float,),
            'RES_SIZE': 0
        }


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

PointOnLineDistance = con.generate_constraint(con.PointOnLineDistance)


PointToLineDistance = con.generate_constraint(con.PointToLineDistance)


class RelativePointOnLineDistance(con.LeafConstruction, con._PointLineSignature):
    """
    Constrain the projected distance of a point along a line

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Line]
        The point and line
    distance: float
    reverse: bool
        A boolean indicating whether to reverse the line direction

        The distance of the point on the line is measured either from the start or end
        point of the line based on `reverse`. If `reverse=False` then the start point is
        used.
    """

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(1, (bool, float))

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
        child_keys = ("HorizontalBottom", "HorizontalTop", "VerticalLeft", "VerticalRight")
        child_constraint_types = (Horizontal, Horizontal, Vertical, Vertical)
        child_constraint_type_kwargs = ({}, {}, {}, {})
        child_prim_keys = (("arg0/Line0",), ("arg0/Line2",), ("arg0/Line3",), ("arg0/Line1",))
        def child_params(params):
            return [(), (), (), ()]
        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0)


AspectRatio = con.generate_constraint(con.AspectRatio)


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
        child_keys = ("Height",)
        child_constraint_types = (YDistance,)
        child_constraint_type_kwargs = ({},)
        child_prim_keys = (("arg0/Line1/Point0", "arg0/Line1/Point1"),)

        def child_params(parameters):
            xaxis: XAxis | None
            xaxis, = parameters
            if xaxis is None:
                return [(0,)]
            else:
                return [(cls.get_xaxis_height(xaxis),)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0, (XAxis,))


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
        child_keys = ("Width",)
        child_constraint_types = (XDistance,)
        child_constraint_type_kwargs = ({},)
        child_prim_keys = (("arg0/Line0/Point0", "arg0/Line0/Point1"),)

        def child_params(parameters):
            yaxis: YAxis | None
            yaxis, = parameters
            if yaxis is None:
                return [(0,)]
            else:
                return [(cls.get_yaxis_width(yaxis),)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0, (YAxis,))


# Argument type: tuple[Quadrilateral, Quadrilateral]

OuterMargin = con.generate_constraint(con.OuterMargin)

InnerMargin = con.generate_constraint(con.InnerMargin)

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
        child_keys = ("AlignBottom", "AlignTop")
        child_constraint_types = (Collinear, Collinear)
        child_constraint_type_kwargs = ({}, {})
        child_prim_keys = (("arg0/Line0", "arg1/Line0"), ("arg0/Line2", "arg1/Line2"))

        def child_params(parameters):
            return 2*((),)

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0)

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
        child_keys = ("AlignLeft", "AlignRight")
        child_constraint_types = (Collinear, Collinear)
        child_constraint_type_kwargs = ({}, {})
        child_prim_keys = (("arg0/Line3", "arg1/Line3"), ("arg0/Line1", "arg1/Line1"))

        def child_params(parameters):
            return 2*((),)

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0)

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
        size = np.prod(shape)
        num_row, num_col = shape

        def idx(i, j):
            return idx_1d((i, j), shape)

        # Specify child constraints given the grid shape
        # Line up bottom/top and left/right
        child_constraint_types = (
            num_row * (con.map(AlignRow, num_col*(pr.Quadrilateral,)),)
            + num_col * (con.map(AlignColumn, num_row*(pr.Quadrilateral,)),)
        )
        child_constraint_type_kwargs = (
            num_row * ({},)
            + num_col * ({},)
        )
        align_rows = [
            tuple(f"arg{idx(nrow, ncol)}" for ncol in range(num_col))
            for nrow in range(num_row)
        ]
        align_cols = [
            tuple(f"arg{idx(nrow, ncol)}" for nrow in range(num_row))
            for ncol in range(num_col)
        ]

        child_prim_keys = align_rows + align_cols
        child_keys = (
            [f"AlignRow{nrow}" for nrow in range(num_row)]
            + [f"AlignColumn{ncol}" for ncol in range(num_col)]
        )
        def child_params(params):
            return len(child_keys)*[()]
        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': (),
            'RES_SIZE': 0
        }


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
        num_args = np.prod(shape)
        num_row, num_col = shape

        # Children constraints do:
        # 1. Align all quads in a grid
        # 2. Set relative column widths relative to column 0
        # 3. Set relative row heights relative to row 0
        child_keys = (
            "RectilinearGrid",
            "ColumnWidths",
            "RowHeights",
            "ColumnMargins",
            "RowMargins",
        )
        child_constraint_types = (
            RectilinearGrid,
            RelativeLengthArray,
            RelativeLengthArray,
            con.map(OuterMargin, num_col*(pr.Quadrilateral,)),
            con.map(OuterMargin, num_row*(pr.Quadrilateral,))
        )
        child_constraint_type_kwargs = (
            {'shape': shape},
            {'shape': num_col-1},
            {'shape': num_row-1},
            {'side': 'right'},
            {'side': 'bottom'},
        )

        def idx(i, j):
            return idx_1d((i, j), shape)
        rows, cols = list(range(shape[0])), list(range(shape[1]))

        rectilineargrid_args = tuple(f"arg{n}" for n in range(num_args))

        colwidth_args = tuple(
            f"arg{idx(row, col)}/Line0"
            for row, col in itertools.product([0], cols[1:] + cols[:1])
        )
        rowheight_args = tuple(
            f"arg{idx(row, col)}/Line1"
            for row, col in itertools.product(rows[1:] + rows[:1], [0])
        )
        col_margin_args = tuple(f"arg{idx(0, col)}" for col in cols)
        row_margin_args = tuple(f"arg{idx(row, 0)}" for row in rows)

        child_prim_keys = (
            rectilineargrid_args,
            colwidth_args,
            rowheight_args,
            tuple(col_margin_args),
            tuple(row_margin_args),
        )

        def child_params(params):
            col_widths, row_heights, col_margins, row_margins = params
            return ((),) + ((col_widths,), (row_heights,)) + (col_margins, row_margins)

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params


    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': (np.ndarray, np.ndarray, np.ndarray, np.ndarray),
            'RES_SIZE': 0
        }


## Axes constraints

# Argument type: tuple[Axes]

# TODO: Handle more specialized x/y axes combos? (i.e. twin x/y axes)
# The below axis constraints are made for single x and y axises

class PositionXAxis(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the x-axis to the top or bottom of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    def __init__(self, bottom: bool=True, top: bool=False):
        return super().__init__(bottom=bottom, top=top)

    @classmethod
    def init_children(cls, bottom: bool, top: bool):

        child_keys = ('CoincidentLines',)
        child_constraint_types = (CoincidentLines,)
        child_constraint_type_kwargs = ({},)
        if bottom:
            child_prim_keys = (('arg0/Frame/Line0', 'arg0/XAxis/Line2'),)
        elif top:
            child_prim_keys = (('arg0/Frame/Line2', 'arg0/XAxis/Line0'),)
        else:
            raise ValueError(
                "Currently, 'bottom' and 'top' can't both be true"
            )

        def child_params(params):
            return [(True,)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls, bottom: bool, top: bool):
        return cls.aux_data(0)


class PositionYAxis(con.CompoundConstruction, con._AxesSignature):
    """
    Constrain the y-axis to the left or right of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    def __init__(self, left: bool=True, right: bool=False):
        return super().__init__(left=left, right=right)

    @classmethod
    def init_children(cls, left: bool=True, right: bool=False):

        child_keys = ('CoincidentLines',)
        child_constraint_types = (CoincidentLines,)
        child_constraint_type_kwargs = ({},)
        if left:
            child_prim_keys = (('arg0/Frame/Line3', 'arg0/YAxis/Line1'),)
        elif right:
            child_prim_keys = (('arg0/Frame/Line1', 'arg0/YAxis/Line3'),)
        else:
            raise ValueError(
                "Currently, 'left' and 'right' can't both be true"
            )

        def child_params(params):
            return [(True,)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls, left: bool=True, right: bool=False):
        return cls.aux_data(0)


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

        child_keys = ('RelativePointOnLineDistance',)
        child_constraint_types = (RelativePointOnLineDistance,)
        child_constraint_type_kwargs = ({},)
        child_prim_keys = (('arg0/XAxisLabel', 'arg0/XAxis/Line0'),)

        def child_params(params):
            distance, = params
            return [(False, distance)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0)


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
        child_keys = ('RelativePointOnLineDistance',)
        child_constraint_types = (RelativePointOnLineDistance,)
        child_constraint_type_kwargs = ({},)
        child_prim_keys = (('arg0/YAxisLabel', 'arg0/YAxis/Line1'),)

        def child_params(params):
            distance, = params
            return [(False, distance)]

        return child_keys, child_constraint_types, child_constraint_type_kwargs, child_prim_keys, child_params

    @classmethod
    def init_aux_data(cls):
        return cls.aux_data(0)
