"""
Geometric constraints
"""

import typing as tp
from numpy.typing import NDArray

import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr

Primitive = pr.Primitive


class Constraint:
    """
    A geometric constraint on primitives

    A constraint represents a condition on the parameter vectors of geometric
    primitive(s) that they should satisfy.

    The condition is implemented through a residual function `assem_res` which returns
     the error in the constraint satisfaction. The constraint is satisfied when
     `assem_res` returns 0.

    The constraint residual should be implemented using `jax`. This allows automatic
    differentiation of constraint conditions which is important for numerical solution
    of constraints.

    Parameters
    ----------
    *args, **kwargs :
        Parameters for the constraint

        These control aspects of the constraint, for example, an angle or distance.

        See `Constraint` subclasses for specific examples.

    Attributes
    ----------
    _PRIMITIVE_TYPES: tp.Tuple[tp.Type[Primitive], ...]
        The primitive types accepted by `assem_res`

        These are the primitive types the constraint applies on.
    """

    _PRIMITIVE_TYPES: tp.Tuple[tp.Type[Primitive], ...]

    def __init__(self, *args, **kwargs):
        self._res_args = args
        self._res_kwargs = kwargs

    def __call__(self, prims: tp.Tuple[Primitive, ...]):
        # Check the input primitives are valid
        # assert len(prims) == len(self._PRIMITIVE_TYPES)
        # for prim, prim_type in zip(prims, self._PRIMITIVE_TYPES):
        #     assert issubclass(type(prim), prim_type)

        return jnp.atleast_1d(self.assem_res(prims))

    def assem_res(self, prims: tp.Tuple[Primitive, ...]) -> NDArray:
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


## Constraints on points


class DirectedDistance(Constraint):
    """
    A constraint on distance between two points along a direction
    """

    def __init__(self, distance: float, direction: tp.Optional[NDArray] = None):

        self._PRIMITIVE_TYPES = (pr.Point, pr.Point)

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
        point0, point1 = prims
        distance = jnp.dot(
            point1.value - point0.value, self._res_kwargs["direction"]
        )
        return distance - self._res_kwargs["distance"]


class XDistance(DirectedDistance):

    def __init__(self, distance: float):
        super().__init__(distance, direction=np.array([1, 0]))


class YDistance(DirectedDistance):

    def __init__(self, distance: float):
        super().__init__(distance, direction=np.array([0, 1]))


class PointLocation(Constraint):
    """
    A constraint on the location of a point
    """

    def __init__(self, location: NDArray):
        self._PRIMITIVE_TYPES = (pr.Point,)
        super().__init__(location=location)

    def assem_res(self, prims):
        """
        Return the location error for a point
        """
        (point,) = prims
        return point.value - self._res_kwargs["location"]


class CoincidentPoints(Constraint):
    """
    A constraint on coincide of two points
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Point, pr.Point)
        super().__init__()

    def assem_res(self, prims):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return point0.value - point1.value


## Line constraints


class Length(Constraint):
    """
    A constraint on the length of a line
    """

    def __init__(self, length: float):
        self._PRIMITIVE_TYPES = (pr.Line,)
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        (line,) = prims
        vec = line_vector(line)
        return jnp.sum(vec**2) - self._res_kwargs["length"] ** 2


class RelativeLength(Constraint):
    """
    A constraint on relative length between two lines
    """

    def __init__(self, length: float):
        self._PRIMITIVE_TYPES = (pr.Line, pr.Line)
        super().__init__(length=length)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = line_vector(line0)
        vec_b = line_vector(line1)
        return jnp.sum(vec_a**2) - self._res_kwargs["length"] ** 2 * jnp.sum(vec_b**2)


class Orthogonal(Constraint):
    """
    A constraint on orthogonality of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Line, pr.Line)
        super().__init__()

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the orthogonal error
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.dot(dir0, dir1)


class Parallel(Constraint):
    """
    A constraint on parallelism of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Line, pr.Line)
        super().__init__()

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the parallel error
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.cross(dir0, dir1)


class Vertical(Constraint):
    """
    A constraint that a line must be vertical
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Line,)
        super().__init__()

    def assem_res(self, prims: tp.Tuple[pr.Line]):
        """
        Return the vertical error
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([1, 0]))


class Horizontal(Constraint):
    """
    A constraint that a line must be horizontal
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Line,)
        super().__init__()

    def assem_res(self, prims: tp.Tuple[pr.Line]):
        """
        Return the horizontal error
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([0, 1]))


class Angle(Constraint):
    """
    A constraint on the angle between two lines
    """

    def __init__(self, angle: NDArray):
        self._PRIMITIVE_TYPES = (pr.Line, pr.Line)
        super().__init__(angle=angle)

    def assem_res(self, prims):
        """
        Return the angle error
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)

        dir0 = dir0 / jnp.linalg.norm(dir0)
        dir1 = dir1 / jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self._res_kwargs["angle"]


class Collinear(Constraint):
    """
    A constraint on the collinearity of two lines
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Line, pr.Line)
        super().__init__()

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the collinearity error
        """
        res_parallel = Parallel()
        line0, line1 = prims
        line2 = pr.Line(children=(line1[0], line0[0]))
        # line3 = primitives.Line(children=(line1['Point0'], line0['Point1']))

        return jnp.concatenate(
            [res_parallel((line0, line1)), res_parallel((line0, line2))]
        )


## Closed polyline constraints


class Box(Constraint):
    """
    Constrain a `Quadrilateral` to have horizontal tops/bottom and vertical sides
    """

    def __init__(self):
        self._PRIMITIVE_TYPES = (pr.Quadrilateral,)
        super().__init__()

    def assem_res(self, prims):
        """
        Return the error in the 'boxiness'
        """
        (quad,) = prims
        horizontal = Horizontal()
        vertical = Vertical()
        res = jnp.concatenate(
            [
                horizontal((quad[0],)),
                horizontal((quad[2],)),
                vertical((quad[1],)),
                vertical((quad[3],)),
            ]
        )
        return res


## Grid constraints


class Grid(Constraint):

    def __init__(
        self,
        shape: tp.Tuple[int, ...],
        horizontal_margins: tp.Union[float, NDArray[float]],
        vertical_margins: tp.Union[float, NDArray[float]],
        widths: tp.Union[float, NDArray[float]],
        heights: tp.Union[float, NDArray[float]],
    ):

        self._PRIMITIVE_TYPES = (pr.Quadrilateral,) * int(np.prod(shape))

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
        (box_topleft, *_) = prims

        res_arrays = [np.array([])]

        for ii, jj in itertools.product(range(num_row - 1), range(num_col)):

            # Set vertical margins
            margin = self._vertical_margins[ii]

            box_a = prims[(ii) * num_col + jj]
            box_b = prims[(ii + 1) * num_col + jj]

            res_arrays.append(
                DirectedDistance(margin, direction=np.array([0, -1]))(
                    (box_a['Line0/Point0'], box_b['Line2/Point1'])
                )
            )

            # Set vertical widths
            length = self._heights[ii]
            res_arrays.append(RelativeLength(length)((box_b['Line1'], box_topleft['Line1'])))

            # Set vertical collinearity
            res_arrays.append(
                Collinear()(
                    (
                        box_a['Line1'],
                        box_b['Line1'],
                    )
                )
            )
            res_arrays.append(
                Collinear()(
                    (
                        box_a['Line3'],
                        box_b['Line3'],
                    )
                )
            )

        for ii, jj in itertools.product(range(num_row), range(num_col - 1)):

            # Set horizontal margins
            margin = self._horizontal_margins[jj]

            box_a = prims[ii * num_col + (jj)]
            box_b = prims[ii * num_col + (jj + 1)]

            res_arrays.append(
                DirectedDistance(margin, direction=np.array([1, 0]))(
                    (box_a['Line0/Point1'], box_b['Line0/Point0'])
                )
            )

            # Set horizontal widths
            length = self._widths[jj]
            # breakpoint()
            res_arrays.append(RelativeLength(length)((box_b['Line0'], box_topleft['Line0'])))

            # Set horizontal collinearity
            res_arrays.append(
                Collinear()(
                    (
                        box_a['Line0'],
                        box_b['Line0'],
                    )
                )
            )
            res_arrays.append(
                Collinear()(
                    (
                        box_a['Line2'],
                        box_b['Line2'],
                    )
                )
            )

        return jnp.concatenate(res_arrays)

def line_vector(line: pr.Line):
    return line[1].value - line[0].value
