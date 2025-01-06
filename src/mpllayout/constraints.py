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

# Argument type: tuple[Point]

Fix = con.transform_ConstraintType(con.Coordinate)

# Argument type: tuple[Point, Point]

DirectedDistance = con.transform_ConstraintType(con.DirectedDistance)

XDistance = con.transform_ConstraintType(con.XDistance)

YDistance = con.transform_ConstraintType(con.YDistance)

class Coincident(con.LeafConstruction, con._PointPointSignature):
    """
    Return coincident error between two points

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Point, pr.Point])
        Return the difference between two point coordinates
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

# Argument type: tuple[Line]

Length = con.transform_ConstraintType(con.Length)


DirectedLength = con.transform_ConstraintType(con.DirectedLength)

XLength = con.transform_ConstraintType(con.XLength)


YLength = con.transform_ConstraintType(con.YLength)


class Vertical(con.LeafConstruction, con._LineSignature):
    """
    Return the vertical error of a line

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
        Return the dot-product between a line vector and the x-axis
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(con.LineVector.assem(prims), np.array([1, 0]))


class Horizontal(con.LeafConstruction, con._LineSignature):
    """
    Return the horizontal error of a line

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line])
        Return the dot-product between a line vector and the y-axis
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(1)

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(con.LineVector.assem(prims), np.array([0, 1]))


# Argument type: tuple[Line, Line]

# TODO: Refactor as derived constraint
class RelativeLength(con.ConstructionNode):
    """
    Return the length error of a line relative to another line

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line], value: float)

        `value` is the desired relative length.
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
    Return the orthogonal error between two lines

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line])
        Return the dot product between the two lines
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
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line])
        Return the cross product between the two lines
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
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line])
    """

    @classmethod
    def init_signature(cls):
        return cls.make_signature(2)

    @classmethod
    def assem(self, prims: tuple[pr.Line, pr.Line]):
        line0, line1 = prims
        line2 = pr.Line(prims=(line1[0], line0[0]))

        return jnp.array(
            [Parallel.assem((line0, line1)), Parallel.assem((line0, line2))]
        )


class CoincidentLines(con.LeafConstruction, con._LineLineSignature):
    """
    Return the coincident error between two lines

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Line, pr.Line], reverse: bool)
        Return the difference between endpoint coordinates of the two lines

        `reverse` controls how the line endpoint errors are computed.
        If `reverse = False`, then the start to start difference and end to end
        difference is returned.
        If `reverse = True`, then the start to end and end to start difference
        is returned.
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


RelativePointOnLineDistance = con.transform_ConstraintType(
    con.RelativePointOnLineDistance
)


## Quad constraints

# Argument type: tuple[Quadrilateral]

class Box(con.StaticCompoundConstruction, con._QuadrilateralSignature):
    """
    Return the rectangularity error of a quadrilateral

    This assumes an orientation for rectangle where the first line of the
    quadrilateral is the bottom of the rectangle, the second line is the right,
    and so on.

    Parameters
    ----------
    None

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral])
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

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral]):
        return super().assem(prims)


Width = con.transform_ConstraintType(con.Width)


Height = con.transform_ConstraintType(con.Height)


AspectRatio = con.transform_ConstraintType(con.AspectRatio)


def get_axis_thickness(axis: XAxis | YAxis, side: str):

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


class AxisThickness(
    con.CompoundConstruction, con._QuadrilateralSignature
):
    """
    Return the thickness error between the axis primitive and `matplotlib` axis

    Parameters
    ----------
    axis: Literal['x', 'y']
        The axis

        The thickness depends on the axis. If `axis = 'x'` then thickness is
        the axis height. If `axis = 'y'` then thickness is the axis width.

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral], mpl_axis: XAxis | YAxis)
    """

    def __init__(self, axis: Literal['x', 'y'] = 'x'):
        super().__init__(axis=axis)

    @staticmethod
    def get_axis_thickness(mpl_axis: XAxis | YAxis):
        return get_axis_thickness(mpl_axis, mpl_axis.get_ticks_position())

    @classmethod
    def init_children(cls, axis: Literal['x', 'y'] = 'x'):
        keys = ("Height",)
        if axis == 'x':
            constraints = (YLength(),)
            prim_keys = (("arg0/Line1",),)
        else:
            constraints = (XLength(),)
            prim_keys = (("arg0/Line0",),)

        def child_params(params: Params) -> tuple[Params, ...]:
            mpl_axis: XAxis | YAxis | None = params[0]
            if mpl_axis is None:
                return ((0,),)
            else:
                return ((cls.get_axis_thickness(mpl_axis),),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls, axis: Literal['x', 'y'] = 'x'):
        if axis == 'x':
            return cls.make_signature(0, (XAxis,))
        else:
            return cls.make_signature(0, (YAxis,))

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral], mpl_axis: XAxis):
        return super().assem(prims, mpl_axis)


class XAxisThickness(AxisThickness):
    """
    Return the thickness error between a x axis primitive and a `matplotlib` axis

    See `AxisThickness` with fixed `axis='x'` for more details.
    """

    def __init__(self):
        super().__init__(axis='x')


class YAxisThickness(AxisThickness):
    """
    Return the thickness error between a y axis primitive and a `matplotlib` axis

    See `AxisThickness` with fixed `axis='y'` for more details.
    """

    def __init__(self):
        super().__init__(axis='y')


# Argument type: tuple[Quadrilateral, Quadrilateral]

OuterMargin = con.transform_ConstraintType(con.OuterMargin)

InnerMargin = con.transform_ConstraintType(con.InnerMargin)

class Align(
    con.CompoundConstruction, con._QuadrilateralQuadrilateralSignature
):
    """
    Return the row or column alignment error of two quadrilaterals

    Parameters
    ----------
    row: bool
        Whether to return the row or column alignment error

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral, pr.Quadrilateral])
        Return the collinearity error between quadrilateral sides

        If `row=True` the top and bottom sides are used. If `row=False the left
        and right sides are used.
    """

    def __init__(self, row: bool = True):
        super().__init__(row=row)

    @classmethod
    def init_children(cls, row: bool = True):
        if row:
            keys = ("AlignBottom", "AlignTop")
            constraints = (Collinear(), Collinear())
            prim_keys = (
                ("arg0/Line0", "arg1/Line0"),
                ("arg0/Line2", "arg1/Line2")
            )
        else:
            keys = ("AlignLeft", "AlignRight")
            constraints = (Collinear(), Collinear())
            prim_keys = (
                ("arg0/Line3", "arg1/Line3"),
                ("arg0/Line1", "arg1/Line1")
            )

        def child_params(params: Params) -> tuple[Params, ...]:
            return 2*((),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(cls, row: bool = True):
        return cls.make_signature(0)

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral, pr.Quadrilateral]):
        return super().assem(prims)


class AlignRow(Align):
    """
    Return the row-alignment error of two quadrilaterals

    See `Align` with fixed `row=True` for more details.
    """

    def __init__(self):
        super().__init__(row=True)


class AlignColumn(Align):
    """
    Return the column-alignment error of two quadrilaterals

    See `Align` with fixed `row=False` for more details.
    """

    def __init__(self):
        super().__init__(row=False)


class CoincidentOutwardFaces(
    con.CompoundConstruction, con._QuadrilateralQuadrilateralSignature
):
    """
    Return the coincident error between outward-facing quadrilateral faces

    Parameters
    ----------
    side: Literal['bottom', 'top', 'left', 'right']
        The face to align

        - If `side = 'left'`, the left side of the first quad is coincident with
        the right side of the second quad.
        - If `side = 'bottom'`, the bottom side of the first quad is coincident
        with the top side of the second quad.
        - ...

        Behaviour for the 'top' and 'right' follows the same pattern.

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral, pr.Quadrilateral])
    """

    def __init__(self, side=Literal['bottom', 'top', 'left', 'right']):
        super().__init__(side=side)

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
    def init_signature(cls, side=Literal['bottom', 'top', 'left', 'right']):
        return cls.make_signature(0)

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral, pr.Quadrilateral]):
        return super().assem(prims)

# Argument type: tuple[Quadrilateral, ...]

def idx_1d(multi_idx: tuple[int, ...], shape: tuple[int, ...]):
    """
    Return a 1D array index from a multi-dimensional array index
    """
    strides = shape[1:] + (1,)
    return sum(axis_idx * stride for axis_idx, stride in zip(multi_idx, strides))


class RectilinearGrid(ArrayConstraint, con._QuadrilateralsSignature):
    """
    Return the rectilinear grid error of a set of quadrilaterals

    Parameters
    ----------
    shape: tuple[int]
        The shape (rows, columns) of the grid

    Methods
    -------
    assem(prims: tuple[pr.Quadrilateral, ...])
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

    @classmethod
    def assem(cls, prims: tuple[pr.Quadrilateral, ...]):
        return super().assem(prims)


class Grid(ArrayConstraint, con._QuadrilateralsSignature):
    """
    Return the dimensioned rectilinear grid error for a set of quadrilaterals

    Parameters
    ----------
    shape: tuple[int]
        The shape (rows, columns) of the grid

    Methods
    -------
    assem(
        prims: tuple[pr.Quadrilateral, ...],
        col_widths: NDArray,
        row_heights: NDArray,
        col_margins: NDArray,
        row_margins: NDArray
    )

        The parameters control the grid margins and dimensions as follows:
        - `col_widths`:
            Column widths (from left to right) relative to the left-most column
        - `row_heights`:
            Row height (from top to bottom) relative to the top-most row
        - `col_margins`:
            Absolute column margins (from left to right)
        - `row_margins`:
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

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Quadrilateral, ...],
        col_widths: NDArray,
        row_heights: NDArray,
        col_margins: NDArray,
        row_margins: NDArray
    ):
        return super().assem(
            prims, col_widths, row_heights, col_margins, row_margins
        )


## Axes constraints

# Argument type: tuple[Axes]

class PositionAxis(con.CompoundConstruction, con._AxesSignature):
    """
    Return the x or y axis side position eror

    Parameters
    ----------
    axis: Literal['x', 'y']
        The axis to position
    side: Literal['bottom', 'top', 'left', 'right']
        The side of the axes to place the axis

        If `axis='x'` this can be 'bottom' or 'top'.
        If `axis='y'` this can be 'left' or 'right'.
    twin: bool
        Whether the axis has a twin

        If there is a twin axis, then the twin axis is place opposite to the
        primary axis.

    Methods
    -------
    assem(prims: tuple[pr.Axes])
        Return the coincident error between axis and desired axes frame side
    """

    def __init__(
        self,
        axis: Literal['x', 'y'] = 'x',
        side: Literal['bottom', 'top', 'left', 'right'] = 'bottom',
        twin: bool=False
    ):
        return super().__init__(axis=axis, side=side, twin=twin)

    @classmethod
    def init_children(
        cls,
        axis: Literal['x', 'y'] = 'x',
        side: Literal['bottom', 'top', 'left', 'right'] = 'bottom',
        twin: bool=False
    ):
        keys = ('AlignOutside',)
        constraints = (CoincidentOutwardFaces(side=side),)
        prim_keys = (('arg0/Frame', f'arg0/{axis.upper()}Axis'),)

        if twin:
            keys = keys + ('TwinAlignOutside',)
            constraints = constraints + (CoincidentOutwardFaces(side=con.opposite_side(side)),)
            prim_keys = prim_keys + (('arg0/Frame', f'arg0/Twin{axis.upper()}Axis'),)

        def child_params(params: Params) -> tuple[Params, ...]:
            if twin:
                child_params = 2*((),)
            else:
                child_params = ((),)
            return child_params

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(
        cls,
        axis: Literal['x', 'y'] = 'x',
        side: Literal['bottom', 'top', 'left', 'right'] = 'bottom',
        twin: bool=False
    ):
        return cls.make_signature(0)

    @classmethod
    def assem(cls, prims: tuple[pr.Axes]):
        return super().assem(prims)


class PositionXAxis(PositionAxis):
    """
    Return the x axis side position eror

    See `PositionAxis` with fixed `axis='x'` for more details.
    """

    def __init__(
        self,
        side: Literal['bottom', 'top']='bottom',
        twin: bool=False
    ):
        return super().__init__(axis='x', side=side, twin=twin)


class PositionYAxis(PositionAxis):
    """
    Return the y axis side position eror

    See `PositionAxis` with fixed `axis='y'` for more details.
    """

    def __init__(
        self,
        side: Literal['left', 'right']='left',
        twin: bool=False
    ):
        return super().__init__(axis='y', side=side, twin=twin)


class PositionAxisLabel(con.CompoundConstruction, con._AxesSignature):
    """
    Return the x or y axis label position error along the axes width

    The behaviour depends on the axis.
    - For the x axis, this is the error along the x-axis and distance is
    measured from left to right.
    - For the y axis, this is the error along the y-axis and distance is
    measured from bottom to top.

    Parameters
    ----------
    axis: Literal['x', 'y']
        The axis label to position
    twin: bool
        Whether to position the twin axis or primary axis

    Methods
    -------
    assem(prims: tuple[pr.Axes], value: float)
        Return the difference between the label distance and desired distance
    """

    def __init__(
        self,
        axis: Literal['x', 'y'] = 'x',
        twin: bool=False
    ):
        super().__init__(axis=axis, twin=twin)

    @classmethod
    def init_children(
        cls,
        axis: Literal['x', 'y'] = 'x',
        twin: bool=False
    ):

        keys = ('RelativePointOnLineDistance',)
        constraints = (RelativePointOnLineDistance(),)

        if twin:
            twin_prefix = 'Twin'
        else:
            twin_prefix = ''

        if axis == 'x':
            prim_keys = (
                (f'arg0/{twin_prefix}XAxisLabel', f'arg0/Frame/Line0'),
            )
        elif axis == 'y':
            prim_keys = (
                (f'arg0/{twin_prefix}YAxisLabel', f'arg0/Frame/Line1'),
            )
        else:
            raise ValueError("`axis` must be 'x' or 'y'")

        def child_params(params: Params) -> tuple[Params, ...]:
            distance, = params
            return ((False, distance),)

        return keys, constraints, prim_keys, child_params

    @classmethod
    def init_signature(
        cls,
        axis: Literal['x', 'y'] = 'x',
        twin: bool=False
    ):
        return cls.make_signature(0, (float,))

    @classmethod
    def assem(cls, prims: tuple[pr.Axes], value: float):
        return super().assem(prims, value)


class PositionXAxisLabel(PositionAxisLabel):
    """
    Return the x axis label position error along the axes width

    See `PositionAxis` with fixed `axis='x'` for more details.
    """
    def __init__(self, twin: bool=False):
        super().__init__(axis='x', twin=twin)


class PositionYAxisLabel(PositionAxisLabel):
    """
    Return the y axis label position error along the axes width

    See `PositionAxis` with fixed `axis='y'` for more details.
    """
    def __init__(self, twin: bool=False):
        super().__init__(axis='y', twin=twin)
