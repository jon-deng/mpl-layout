"""
Geometric constraints
"""

import typing as typ

import itertools

from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp

from . import primitives as primitives

PrimList = typ.Tuple[primitives.Primitive, ...]
ConstraintList = typ.List['Constraint']
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
PrimTuple = typ.Tuple[primitives.Primitive, ...]

class Constraint:
    """
    A representation of a constraint on primitives

    A constraint represents a condition on the parameters of geometric
    primitive(s). The condition is specified through a residual function
    `assem_res`, specified with the `jax` library. Usage of `jax` in specifying
    the constraints allows automatic differentiation of constraint conditions,
    which is used for solving constraints.

    Parameters
    ----------
    *args, **kwargs :
        Specific parameters that control a constraint, for example, an angle or
        distance

        See `Constraint` subclasses for specific examples.

    Attributes
    ----------
    _PRIMITIVE_TYPES: typ.Tuple[typ.Type['Primitive'], ...]
        The types of primitives accepted by `assem_res`

        This is used for type checking.
    """

    _PRIMITIVE_TYPES: typ.Tuple[typ.Type['Primitive'], ...]

    def __init__(self, *args, **kwargs):
        self._res_args = args
        self._res_kwargs = kwargs

    def __call__(self, prims: typ.Tuple['Primitive', ...]):
        # Check the input primitives are valid
        # assert len(prims) == len(self._PRIMITIVE_TYPES)
        # for prim, prim_type in zip(prims, self._PRIMITIVE_TYPES):
        #     assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))

    def assem_res(self, prims: typ.Tuple['Primitive', ...]) -> NDArray:
        """
        Return a residual vector representing the constraint satisfaction

        Parameters
        ----------
        prims: Tuple[Primitive, ...]
            A tuple of primitives the constraint applies to

        Returns
        -------
        NDArray
            The residual representing whether the constraint is satisfied.
            The constraint is satisfied when the residual is 0.
        """
        raise NotImplementedError()



class PointToPointDirectedDistance(Constraint):
    """
    A constraint on distance between two points along a direction
    """


    def __init__(
            self, distance: float, direction: typ.Optional[NDArray]=None
        ):

        self._PRIMITIVE_TYPES = (primitives.Point, primitives.Point)

        if direction is None:
            direction = np.array([1, 0])
        else:
            direction = np.array(direction)
        super().__init__(distance=distance, direction=direction)

    def assem_res(self, prims):
        """
        Return the distance error between two points along a given direction

        The distance is measured from the first to the second point along a
        specified direction.
        """
        distance = jnp.dot(
            prims[1].param - prims[0].param, self._res_kwargs['direction']
        )
        return distance - self._res_kwargs['distance']

class PointLocation(Constraint):
    """
    A constraint on the location of a point
    """


    def __init__(
            self,
            location: NDArray
        ):
        self._PRIMITIVE_TYPES = (primitives.Point, )
        super().__init__(location=location)

    def assem_res(self, prims):
        """
        Return the location error for a point
        """
        return prims[0].param - self._res_kwargs['location']

class CoincidentPoints(Constraint):
    """
    A constraint on coincide of two points
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Point, primitives.Point)
        super().__init__()

    def assem_res(self, prims):
        """
        Return the coincident error between two points
        """
        return prims[0].param - prims[1].param


## Line constraints

class LineLength(Constraint):
    """
    A constraint on the length of a line
    """


    def __init__(
            self,
            length: float
        ):
        self._PRIMITIVE_TYPES = (primitives.Line,)
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec = line_direction(prims[0])
        return jnp.sum(vec**2) - self._res_kwargs['length']**2


class RelativeLineLength(Constraint):
    """
    A constraint on relative length between two lines
    """


    def __init__(
            self,
            length: float
        ):
        self._PRIMITIVE_TYPES = (primitives.Line, primitives.Line)
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        vec_a = line_direction(prims[0])
        vec_b = line_direction(prims[1])
        return jnp.sum(vec_a**2) - self._res_kwargs['length']**2 * jnp.sum(vec_b**2)


class OrthogonalLines(Constraint):
    """
    A constraint on orthogonality of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Line, primitives.Line)
        super().__init__()

    def assem_res(self, prims: typ.Tuple['primitives.LineSegment', 'primitives.LineSegment']):
        """
        Return the orthogonal error
        """
        line0, line1 = prims
        dir0 = line0.prims[1].param - line0.prims[0].param
        dir1 = line1.prims[1].param - line1.prims[0].param
        return jnp.dot(dir0, dir1)


class ParallelLines(Constraint):
    """
    A constraint on parallelism of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Line, primitives.Line)
        super().__init__()

    def assem_res(self, prims: typ.Tuple['primitives.LineSegment', 'primitives.LineSegment']):
        """
        Return the parallel error
        """
        line0, line1 = prims
        dir0 = line0.prims[1].param - line0.prims[0].param
        dir1 = line1.prims[1].param - line1.prims[0].param
        return jnp.cross(dir0, dir1)


class VerticalLine(Constraint):
    """
    A constraint that a line must be vertical
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Line,)
        super().__init__()

    def assem_res(self, prims: typ.Tuple['primitives.LineSegment']):
        """
        Return the vertical error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([1, 0]))


class HorizontalLine(Constraint):
    """
    A constraint that a line must be horizontal
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Line,)
        super().__init__()

    def assem_res(self, prims: typ.Tuple['primitives.LineSegment']):
        """
        Return the horizontal error
        """
        line0, = prims
        dir0 = line_direction(line0)
        return jnp.dot(dir0, np.array([0, 1]))


class Angle(Constraint):
    """
    A constraint on the angle between two lines
    """

    def __init__(
            self,
            angle: NDArray
        ):
        self._PRIMITIVE_TYPES = (primitives.Line, primitives.Line)
        super().__init__(angle=angle)

    def assem_res(self, prims):
        """
        Return the angle error
        """
        line0, line1 = prims
        dir0 = line_direction(line0)
        dir1 = line_direction(line1)

        dir0 = dir0/jnp.linalg.norm(dir0)
        dir1 = dir1/jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self._res_kwargs['angle']


class CollinearLines(Constraint):
    """
    A constraint on the collinearity of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (primitives.Line, primitives.Line)
        super().__init__()

    def assem_res(self, prims: typ.Tuple['primitives.LineSegment', 'primitives.LineSegment']):
        """
        Return the collinearity error
        """
        res_parallel = ParallelLines()
        line0, line1 = prims
        line2 = primitives.Line(prims=(line1.prims[1], line0.prims[0]))
        line3 = primitives.Line(prims=(line1.prims[0], line0.prims[1]))

        return jnp.concatenate([
            res_parallel((line0, line1)), res_parallel((line0, line2))
        ])


## Closed polyline constraints

class Box(Constraint):
    """
    Constrain a `Quadrilateral` to have horizontal tops/bottom and vertical sides
    """

    def __init__(
            self
        ):
        self._PRIMITIVE_TYPES = (primitives.Quadrilateral,)
        super().__init__()

    def assem_res(self, prims):
        """
        Return the error in the 'boxiness'
        """
        quad = prims[0]
        horizontal = HorizontalLine()
        vertical = VerticalLine()
        res = jnp.concatenate([
            horizontal((quad[0],)),
            horizontal((quad[2],)),
            vertical((quad[1],)),
            vertical((quad[3],))
        ])
        return res

## Grid constraints

class Grid(Constraint):

    def __init__(
            self,
            shape: typ.Tuple[int, ...],
            horizontal_margins: typ.Union[float, NDArray[float]],
            vertical_margins: typ.Union[float, NDArray[float]],
            widths: typ.Union[float, NDArray[float]],
            heights: typ.Union[float, NDArray[float]]
        ):

        self._PRIMITIVE_TYPES = (primitives.Quadrilateral,) * int(np.prod(shape))

        self._shape = shape
        self._vertical_margins = vertical_margins
        self._horizontal_margins = horizontal_margins
        self._heights = heights
        self._widths = widths

        super().__init__()

    def assem_res(self, prims):
        # boxes = np.array(prims).reshape(self._shape)

        num_row, num_col = self._shape

        # Set the top left (0 box) to have the right width/height
        box_topleft = prims[0]
        # LineLength(self._widths[0])((box_topleft[0],))
        # LineLength(self._heights[0])((box_topleft[1],))

        res_arrays = [np.array([])]

        for ii, jj in itertools.product(range(num_row-1), range(num_col)):

            # Set vertical margins
            margin = self._vertical_margins[ii]

            box_a = prims[(ii)*num_col + jj]
            box_b = prims[(ii+1)*num_col + jj]

            res_arrays.append(
                PointToPointDirectedDistance(
                    margin, direction=np.array([0, -1])
                )((box_a[0][0], box_b[2][1]))
            )

            # Set vertical widths
            length = self._heights[ii]/self._heights[0]
            res_arrays.append(RelativeLineLength(length)((box_topleft[1], box_b[1],)))

            # Set vertical collinearity
            res_arrays.append(CollinearLines()((box_a[1], box_b[1],)))
            res_arrays.append(CollinearLines()((box_a[3], box_b[3],)))

        for ii, jj in itertools.product(range(num_row), range(num_col-1)):

            # Set horizontal margins
            margin = self._horizontal_margins[jj]

            box_a = prims[ii*num_col + (jj)]
            box_b = prims[ii*num_col + (jj+1)]

            res_arrays.append(
                PointToPointDirectedDistance(
                    margin, direction=np.array([1, 0])
                )((box_a[0][1], box_b[0][0]))
            )

            # Set horizontal widths
            length = self._widths[jj]/self._widths[0]
            res_arrays.append(RelativeLineLength(length)((box_topleft[0], box_b[0],)))

            # Set horizontal collinearity
            res_arrays.append(CollinearLines()((box_a[0], box_b[0],)))
            res_arrays.append(CollinearLines()((box_a[2], box_b[2],)))

        return jnp.concatenate(res_arrays)


def line_direction(line: 'primitives.LineSegment'):
    return line.prims[1].param - line.prims[0].param

