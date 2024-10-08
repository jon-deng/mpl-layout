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
    ARG_TYPES: tp.Tuple[tp.Type[Primitive], ...]
        The primitive types accepted by `assem_res`

        These are the primitive types the constraint applies on.
    """

    ARG_TYPES: tp.Tuple[type[Primitive], ...]
    CONSTANTS: type[collections.namedtuple] = collections.namedtuple('Constants', ())

    CHILD_TYPES: tp.Tuple[type['Constraint'], ...] = ()
    CHILD_KEYS: tp.Tuple[str, ...] = ()
    CHILD_CONSTANTS: tp.Callable[
        [type['Constraint'], Constants],
        tp.Tuple[Constants, ...]
    ]
    CHILD_ARGS: tp.Tuple[tp.Tuple[str, ...], ...] = ()

    @classmethod
    def CHILD_CONSTANTS(cls, constants: Constants):
        # Use default constants if not specified
        return len(cls.CHILD_TYPES) * ({},)

    @classmethod
    def load_constants(cls, constants: Constants):
        if isinstance(constants, dict):
            constants = cls.CONSTANTS(**constants)
        elif isinstance(constants, tuple):
            constants = cls.CONSTANTS(*constants)
        elif isinstance(constants, cls.CONSTANTS):
            pass
        else:
            raise TypeError()
        return constants

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        constants = cls.load_constants(constants)

        # `arg_keys` specifies the keys from a root primitive that gives the primitives
        # for `assem_res(prims)`.
        # If `arg_keys` is not supplied (the constraint is top-level constraint) then these
        # are simply f'arg{n}' for integer argument numbers `n`
        # Child keys `arg_keys` are assumed to index from `arg_keys`
        if arg_keys is None:
            arg_keys = tuple(f'arg{n}' for n in range(len(cls.ARG_TYPES)))

        # Replace the first 'arg{n}/...' key with the appropriate parent argument keys
        def get_parent_arg_number(arg_key: str):
            arg_number_str = arg_key.split('/', 1)[0]
            if arg_number_str[:3] == 'arg':
                arg_number = int(arg_number_str[3:])
            else:
                raise ValueError(f"Argument key, {arg_key}, must contain 'arg' prefix")
            return arg_number

        parent_args_numbers = [
            tuple(get_parent_arg_number(arg_key) for arg_key in carg_keys)
            for carg_keys in cls.CHILD_ARGS
        ]
        parent_args = [
            tuple(arg_keys[parent_arg_num] for parent_arg_num in parent_arg_nums)
            for parent_arg_nums in parent_args_numbers
        ]
        child_args = tuple(
            tuple(
                "/".join([parent_arg_key]+arg_key.split('/', 1)[1:])
                for parent_arg_key, arg_key in zip(parent_arg_keys, carg_keys)
            )
            for parent_arg_keys, carg_keys in zip(parent_args, cls.CHILD_ARGS)
        )

        child_constants = cls.CHILD_CONSTANTS(constants)
        children = {
            key: ChildType.from_std(constant, arg_keys=arg_keys)
            for key, ChildType, constant, arg_keys
            in zip(cls.CHILD_KEYS, cls.CHILD_TYPES, child_constants, child_args)
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

    ARG_TYPES = (pr.Point, pr.Point)
    CONSTANTS = collections.namedtuple("Constants", ["distance", "direction"])

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
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        direction = np.array([1, 0])

        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({'direction': direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        return super().from_std(constants, arg_keys)


class YDistance(DirectedDistance):

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        direction = np.array([0, 1])
        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({'direction': direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        return super().from_std(constants, arg_keys)


class PointLocation(Constraint):
    """
    A constraint on the location of a point
    """

    ARG_TYPES = (pr.Point,)
    CONSTANTS = collections.namedtuple("Constants", ["location"])

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

    ARG_TYPES = (pr.Point, pr.Point)
    CONSTANTS = collections.namedtuple("Constants", [])

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

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", ["length"])

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

    CONSTANTS = collections.namedtuple("Constants", ["length"])
    ARG_TYPES = (pr.Line, pr.Line)

    def assem_res(self, prims):
        """
        Return the error in the length of the line
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = line_vector(line0)
        vec_b = line_vector(line1)
        return jnp.sum(vec_a**2) - self.constants.length ** 2 * jnp.sum(vec_b**2)


class RelativeLengths(Constraint):
    """
    A constraint on relative lengths of a set of lines
    """

    CONSTANTS = collections.namedtuple("Constants", ("lengths",))

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = cls.load_constants(constants)
        num_args = len(_constants.lengths) + 1
        cls.ARG_TYPES = num_args*(pr.Line,)

        cls.CHILD_TYPES = (num_args-1)*(RelativeLength,)
        cls.CHILD_ARGS = tuple(
            (f'arg{n}', f'arg{num_args-1}') for n in range(num_args-1)
        )
        cls.CHILD_CONSTANTS = lambda constants: tuple(
            (length,) for length in constants.lengths
        )
        cls.CHILD_KEYS = tuple(
            f'RelativeLength{n}' for n in range(num_args-1)
        )

        return super().from_std(constants, arg_keys)

    def assem_res(self, prims):
        return np.array([])


class Orthogonal(Constraint):
    """
    A constraint on orthogonality of two lines
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

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

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

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

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", [])

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

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", [])

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

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", ["angle"])

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

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

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


class CollinearLines(Constraint):
    """
    A constraint on the collinearity of 2 or more lines
    """

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple("Constants", ("size",))

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = cls.load_constants(constants)
        size = _constants.size
        if size < 1:
            raise ValueError()

        cls.ARG_TYPES = size*(pr.Line,)

        cls.CHILD_TYPES = (size-1)*(Collinear,)
        cls.CHILD_ARGS = tuple(('arg0', f'arg{n}') for n in range(1, size))
        cls.CHILD_KEYS = tuple(f'Collinear[0][{n}]' for n in range(1, size))
        cls.CHILD_CONSTANTS = lambda constants: (constants.size-1)*((),)
        return super().from_std(constants, arg_keys)

    def assem_res(self, prims):
        return np.array([])


## Closed polyline constraints


class Box(Constraint):
    """
    Constrain a `Quadrilateral` to have horizontal tops/bottom and vertical sides
    """

    ARG_TYPES = (pr.Quadrilateral,)
    CONSTANTS = collections.namedtuple("Constants", [])

    CHILD_KEYS = ('HorizontalBottom', 'HorizontalTop', 'VerticalLeft', 'VerticalRight')
    CHILD_TYPES = (Horizontal, Horizontal, Vertical, Vertical)
    CHILD_ARGS = (('arg0/Line0',), ('arg0/Line2',), ('arg0/Line3',), ('arg0/Line1',))

    def assem_res(self, prims):
        return np.array([])


## Grid constraints

class RectilinearGrid(Constraint):
    """
    Constrain quads to a rectilinear grid
    """

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple(
        "Constants", ("shape",)
    )

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = cls.load_constants(constants)
        shape = _constants.shape

        num_row, num_col = shape
        num_args = num_row*num_col

        cls.ARG_TYPES = num_args*(pr.Quadrilateral,)

        # Specify child constraints given the grid shape

        # Line up bot/top/left/right
        def flat_idx(row, col):
            return row*num_col + col
        CHILD_TYPES = 2*num_row*(CollinearLines,) + 2*num_col*(CollinearLines,)
        CHILD_ARGS = [
            tuple(f'arg{flat_idx(nrow, ncol)}/Line0' for ncol in range(num_col))
            for nrow in range(num_row)
        ] + [
            tuple(f'arg{flat_idx(nrow, ncol)}/Line2' for ncol in range(num_col))
            for nrow in range(num_row)
        ] + [
            tuple(f'arg{flat_idx(nrow, ncol)}/Line3' for nrow in range(num_row))
            for ncol in range(num_col)
        ] + [
            tuple(f'arg{flat_idx(nrow, ncol)}/Line1' for nrow in range(num_row))
            for ncol in range(num_col)
        ]
        CHILD_KEYS =(
            [f'CollinearRowBottom{nrow}' for nrow in range(num_row)]
            + [f'CollinearRowTop{nrow}' for nrow in range(num_row)]
            + [f'CollinearColumnLeft{ncol}' for ncol in range(num_col)]
            + [f'CollinearColumnRight{ncol}' for ncol in range(num_col)]
        )

        cls.CHILD_CONSTANTS = lambda constants: (
            [(constants.shape[1],) for nrow in range(constants.shape[0])]
            +[(constants.shape[1],) for nrow in range(constants.shape[0])]
            +[(constants.shape[0],) for ncol in range(constants.shape[1])]
            +[(constants.shape[0],) for ncol in range(constants.shape[1])]
        )

        cls.CHILD_TYPES = CHILD_TYPES
        cls.CHILD_ARGS = CHILD_ARGS
        cls.CHILD_KEYS = CHILD_KEYS

        return super().from_std(constants, arg_keys)

    def assem_res(self, prims):
        return np.array([])


class Grid(Constraint):

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple(
        "Constants", ("shape", "horizontal_margins", "vertical_margins", "widths", "heights")
    )

    @classmethod
    def from_std(
        cls,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = cls.load_constants(constants)
        num_args = np.prod(_constants.shape)
        cls.ARG_TYPES = num_args*(pr.Quadrilateral,)
        return super().from_std(constants, arg_keys)

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
