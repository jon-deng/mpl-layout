"""
Geometric constraints
"""

import typing as tp
from numpy.typing import NDArray

import collections
import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat

Primitive = pr.Primitive


Constants = tp.Mapping[str, tp.Any] | tp.Tuple[tp.Any, ...]
PrimKeys = tp.Tuple[str, ...]
ConstraintValue = tp.Tuple[Constants, PrimKeys]

class Constraint(Node[ConstraintValue]):
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
    _CONSTANTS: tp.Type[collections.namedtuple] = collections.namedtuple('Constants', ())

    _CHILD_TYPES: tp.Tuple[tp.Type['Constraint'], ...] = ()
    _CHILD_KEYS: tp.Tuple[str, ...] = ()
    # NOTE: I think the class should be able to specify what the constants are for
    # any child constraints but this isn't possible to do with knowing the parent constraints
    # _CHILD_CONSTANTS: tp.Tuple[Any, ...]
    _CHILD_ARGS: tp.Tuple[tp.Tuple[str, ...], ...] = ()

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
        child_constants: tp.Tuple[Constants, ...] = None
    ):
        if child_constants is None:
            child_constants = ({},) * len(cls._CHILD_TYPES)

        if isinstance(constants, dict):
            constants = cls._CONSTANTS(**constants)
        elif isinstance(constants, tuple):
            constants = cls._CONSTANTS(*constants)
        elif isinstance(constants, cls._CONSTANTS):
            constants = constants
        else:
            raise TypeError()

        if arg_keys is None:
            arg_keys = tuple(f'arg{n}' for n in range(len(cls._PRIMITIVE_TYPES)))
        elif isinstance(arg_keys, tuple):
            pass

        children = {
            key: ChildType.from_std(constant, arg_keys=ChildArgs)
            for key, ChildType, constant, ChildArgs
            in zip(cls._CHILD_KEYS, cls._CHILD_TYPES, child_constants, cls._CHILD_ARGS)
        }

        return cls((constants, arg_keys), children)

    @property
    def constants(self):
        return self.value[0]

    @property
    def prim_keys(self):
        return self.value[1]

    def __call__(self, prims: tp.Tuple[Primitive, ...]):
        root_prim = Node(
            np.array([]),
            {f'arg{n}': prim for n, prim in enumerate(prims)}
        )
        return self.assem_res_from_tree(root_prim)

    def assem_res_from_tree(self, root_prim: Node[NDArray]):
        # flat_constraints = flatten('', self)
        residuals = tuple(
            constraint.assem_res_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in constraint.prim_keys)
            )
            for _, constraint in iter_flat('', self)
        )
        return jnp.concatenate(residuals)

    def assem_res_atleast_1d(self, prims: tp.Tuple[Primitive, ...]) -> NDArray:
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

    _PRIMITIVE_TYPES = (pr.Point, pr.Point)
    _CONSTANTS = collections.namedtuple("Constants", ["distance", "direction"])

    def assem_res(self, prims: tp.Tuple[pr.Point, pr.Point]):
        """
        Return the distance error between two points along a given direction

        The distance is measured from the first to the second point along a
        specified direction.
        """
        point0, point1 = prims
        distance = jnp.dot(point1.value - point0.value, self.constants.direction)
        return distance - self.constants.distance


class XDistance(DirectedDistance):

    @classmethod
    def from_std(cls, constants: dict[str, tp.Any] | tp.Tuple[tp.Any, ...]):
        direction = np.array([1, 0])
        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({'direction': direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        return super().from_std(constants)


class YDistance(DirectedDistance):

    @classmethod
    def from_std(cls, constants: dict[str, tp.Any] | tp.Tuple[tp.Any, ...]):
        direction = np.array([0, 1])
        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({'direction': direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        return super().from_std(constants)


class PointLocation(Constraint):
    """
    A constraint on the location of a point
    """

    _PRIMITIVE_TYPES = (pr.Point,)
    _CONSTANTS = collections.namedtuple("Constants", ["location"])

    def assem_res(self, prims):
        """
        Return the location error for a point
        """
        (point,) = prims
        return point.value - self.constants.location


class CoincidentPoints(Constraint):
    """
    A constraint on coincide of two points
    """

    _PRIMITIVE_TYPES = (pr.Point, pr.Point)
    _CONSTANTS = collections.namedtuple("Constants", [])

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

    _PRIMITIVE_TYPES = (pr.Line,)
    _CONSTANTS = collections.namedtuple("Constants", ["length"])

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        (line,) = prims
        vec = line_vector(line)
        return jnp.sum(vec**2) - self.constants.length ** 2


class RelativeLength(Constraint):
    """
    A constraint on relative length between two lines
    """

    _CONSTANTS = collections.namedtuple("Constants", ["length"])
    _PRIMITIVE_TYPES = (pr.Line, pr.Line)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = line_vector(line0)
        vec_b = line_vector(line1)
        return jnp.sum(vec_a**2) - self.constants.length ** 2 * jnp.sum(vec_b**2)


class Orthogonal(Constraint):
    """
    A constraint on orthogonality of two lines
    """

    _PRIMITIVE_TYPES = (pr.Line, pr.Line)
    _CONSTANTS = collections.namedtuple("Constants", [])

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

    _PRIMITIVE_TYPES = (pr.Line, pr.Line)
    _CONSTANTS = collections.namedtuple("Constants", [])

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

    _PRIMITIVE_TYPES = (pr.Line,)
    _CONSTANTS = collections.namedtuple("Constants", [])

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

    _PRIMITIVE_TYPES = (pr.Line,)
    _CONSTANTS = collections.namedtuple("Constants", [])

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

    _PRIMITIVE_TYPES = (pr.Line, pr.Line)
    _CONSTANTS = collections.namedtuple("Constants", ["angle"])

    def assem_res(self, prims):
        """
        Return the angle error
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)

        dir0 = dir0 / jnp.linalg.norm(dir0)
        dir1 = dir1 / jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - self.constants.angle


class Collinear(Constraint):
    """
    A constraint on the collinearity of two lines
    """

    _PRIMITIVE_TYPES = (pr.Line, pr.Line)
    _CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the collinearity error
        """
        res_parallel = Parallel.from_std({})
        line0, line1 = prims
        line2 = pr.Line.from_std(children=(line1[0], line0[0]))
        # line3 = primitives.Line.from_std(children=(line1['Point0'], line0['Point1']))

        return jnp.concatenate(
            [res_parallel((line0, line1)), res_parallel((line0, line2))]
        )


## Closed polyline constraints


class Box(Constraint):
    """
    Constrain a `Quadrilateral` to have horizontal tops/bottom and vertical sides
    """

    _PRIMITIVE_TYPES = (pr.Quadrilateral,)
    _CONSTANTS = collections.namedtuple("Constants", [])
    _CHILD_KEYS = ('HorizontalBottom', 'HorizontalTop', 'VerticalLeft', 'VerticalRight')
    _CHILD_TYPES = (Horizontal, Horizontal, Vertical, Vertical)
    _CHILD_ARGS = (('arg0/Line0',), ('arg0/Line2',), ('arg0/Line3',), ('arg0/Line1',))

    # def assem_res(self, prims):
    #     """
    #     Return the error in the 'boxiness'
    #     """
    #     (quad,) = prims
    #     horizontal = Horizontal.from_std({})
    #     vertical = Vertical.from_std({})
    #     res = jnp.concatenate(
    #         [
    #             horizontal((quad[0],)),
    #             horizontal((quad[2],)),
    #             vertical((quad[1],)),
    #             vertical((quad[3],)),
    #         ]
    #     )
    #     return res

    def assem_res(self, prims):
        return np.array([])


## Grid constraints


class Grid(Constraint):

    _PRIMITIVE_TYPES = (pr.Quadrilateral,)
    _CONSTANTS = collections.namedtuple(
        "Constants", ("shape", "horizontal_margins", "vertical_margins", "widths", "heights")
    )

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
        child_constants: tp.Tuple[Constants, ...] = None
    ):
        _temp = super().from_std(constants)
        num_args = np.prod(_temp.constants.shape)
        return super().from_std(constants, tuple(f'arg{n}' for n in range(num_args)))

    def assem_res(self, prims):
        # boxes = np.array(prims).reshape(self._shape)

        num_row, num_col = self.constants.shape

        # Set the top left (0 box) to have the right width/height
        (box_topleft, *_) = prims

        res_arrays = [np.array([])]

        for ii, jj in itertools.product(range(num_row - 1), range(num_col)):

            # Set vertical margins
            margin = self.constants.vertical_margins[ii]

            box_a = prims[(ii) * num_col + jj]
            box_b = prims[(ii + 1) * num_col + jj]

            res_arrays.append(
                DirectedDistance.from_std((margin, np.array([0, -1])))(
                    (box_a["Line0/Point0"], box_b["Line2/Point1"])
                )
            )

            # Set vertical widths
            length = self.constants.heights[ii]
            res_arrays.append(
                RelativeLength.from_std((length,))((box_b["Line1"], box_topleft["Line1"]))
            )

            # Set vertical collinearity
            res_arrays.append(
                Collinear.from_std({})(
                    (
                        box_a["Line1"],
                        box_b["Line1"],
                    )
                )
            )
            res_arrays.append(
                Collinear.from_std({})(
                    (
                        box_a["Line3"],
                        box_b["Line3"],
                    )
                )
            )

        for ii, jj in itertools.product(range(num_row), range(num_col - 1)):

            # Set horizontal margins
            margin = self.constants.horizontal_margins[jj]

            box_a = prims[ii * num_col + (jj)]
            box_b = prims[ii * num_col + (jj + 1)]

            res_arrays.append(
                DirectedDistance.from_std((margin, np.array([1, 0])))(
                    (box_a["Line0/Point1"], box_b["Line0/Point0"])
                )
            )

            # Set horizontal widths
            length = self.constants.widths[jj]
            # breakpoint()
            res_arrays.append(
                RelativeLength.from_std((length,))((box_b["Line0"], box_topleft["Line0"]))
            )

            # Set horizontal collinearity
            res_arrays.append(
                Collinear.from_std({})(
                    (
                        box_a["Line0"],
                        box_b["Line0"],
                    )
                )
            )
            res_arrays.append(
                Collinear.from_std({})(
                    (
                        box_a["Line2"],
                        box_b["Line2"],
                    )
                )
            )

        return jnp.concatenate(res_arrays)


def line_vector(line: pr.Line):
    return line[1].value - line[0].value
