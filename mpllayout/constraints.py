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
Parameters = tp.Tuple[tp.Any, ...]
PrimKeys = tp.Tuple[str, ...]
ConstraintValue = tp.Tuple[Constants, PrimKeys]

ChildConstraint = tp.TypeVar("ChildConstraint", bound="Constraint")


class Constraint(Node[ConstraintValue, ChildConstraint]):
    """
    Geometric constraint on primitives

    A geometric constraint is a condition on parameters of geometric primitives.
    Constraints have a tree structure where each constraint can contain
    child constraints.

    The condition is implemented through a residual function `assem_res` which returns
    the error in the constraint satisfaction; when `assem_res(prims, params)` returns 0,
    the primitives, `prims`, satisfy the constraint for given parameters, `params`.

    The constraint residual should also be implemented using `jax`. This allows
    automatic differentiation of constraint conditions which is needed for the numerical
    solution of constraints.

    Parameters
    ----------
    constants : tp.Mapping[str, tp.Any] | tp.Tuple[tp.Any, ...]
        Constants for the constraint

        These constants control aspects of the constraint, for example, the shape of
        a grid.
    arg_keys : tp.Tuple[str, ...] | None
        Strings indicating which primitives `assem_res` applies to

        This parameter controls what primitives `assem_res` takes from the
        root constraint primitive arguments.
        If the root constraint is `root`, these arguments are given by `root_prims` in
        `root.assem_res(root_prims, params)`.
        Each string in `arg_keys` has the format 'arg{n}/{ChildPrimKey}'.
        The first part 'arg{n}' indicates which primitive to take from `root_prims`.
        The second part '/{ChildPrimKey}' indicates which child primitive to take from
        `root_prims[n]`.

        For example, consider `root.assem_res(root_prims, params)` where
        `root_prims = (line0, line1)` contains two `Line` primitives.
            - The root constraint would have `arg_keys = ('arg0', ..., 'arg{n}')` where
            `n = len(root_prims)-1`.
            - A `child` constraint that acts on the `line1`'s first point would have
            `arg_keys = ('arg1/Point0',)`.

    Attributes
    ----------
    ARG_TYPES: tp.Tuple[tp.Type[Primitive], ...]
        Primitives accepted by `assem_res`

        (see `arg_keys` above)
    ARG_PARAMETERS: tp.Tuple[tp.Type[Primitive], ...]
        Parameters accept by `assem_res`

        (see `arg_keys` above)
    CONSTANTS: collections.namedtuple('Constants', ...)
        Constants for the constraint

        These are things like lengths, angles, etc.
    CHILD_TYPES:
        Constraint types child constraints
    CHILD_KEYS:
        Keys for child constraints
    CHILD_CONSTANTS:
        Constants for child constraints
    CHILD_ARGS:
        Primitives for child constraints' `assem_res`

        (see `arg_keys` above)
    """

    ARG_TYPES: tp.Tuple[type[Primitive], ...]
    ARG_PARAMETERS: type[collections.namedtuple] = collections.namedtuple("Parameters", ())
    CONSTANTS: type[collections.namedtuple] = collections.namedtuple("Constants", ())

    CHILD_TYPES: tp.Tuple[type["Constraint"], ...] = ()
    CHILD_KEYS: tp.Tuple[str, ...] = ()
    CHILD_PARAMETERS: tp.Callable[
        [type["Constraint"], Constants], tp.Tuple[Constants, ...]
    ]
    CHILD_ARGS: tp.Tuple[tp.Tuple[str, ...], ...] = ()

    @classmethod
    def CHILD_PARAMETERS(cls, parameters: Parameters):
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

    def __init__(
        self,
        constants: Constants = None,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        if constants is None:
            constants = ()
        constants = self.load_constants(constants)

        # `arg_keys` specifies the keys from a root primitive that gives the primitives
        # for `assem_res(prims)`.
        # If `arg_keys` is not supplied (the constraint is top-level constraint) then these
        # are simply f'arg{n}' for integer argument numbers `n`
        # Child keys `arg_keys` are assumed to index from `arg_keys`
        if arg_keys is None:
            arg_keys = tuple(f"arg{n}" for n in range(len(self.ARG_TYPES)))

        # Replace the first 'arg{n}/...' key with the appropriate parent argument keys
        def get_parent_arg_number(arg_key: str):
            arg_number_str = arg_key.split("/", 1)[0]
            if arg_number_str[:3] == "arg":
                arg_number = int(arg_number_str[3:])
            else:
                raise ValueError(f"Argument key, {arg_key}, must contain 'arg' prefix")
            return arg_number

        parent_args_numbers = [
            tuple(get_parent_arg_number(arg_key) for arg_key in carg_keys)
            for carg_keys in self.CHILD_ARGS
        ]
        parent_args = [
            tuple(arg_keys[parent_arg_num] for parent_arg_num in parent_arg_nums)
            for parent_arg_nums in parent_args_numbers
        ]
        child_args = tuple(
            tuple(
                "/".join([parent_arg_key] + arg_key.split("/", 1)[1:])
                for parent_arg_key, arg_key in zip(parent_arg_keys, carg_keys)
            )
            for parent_arg_keys, carg_keys in zip(parent_args, self.CHILD_ARGS)
        )

        child_constants = self.CHILD_PARAMETERS(constants)
        children = {
            key: ChildType(constant, arg_keys=arg_keys)
            for key, ChildType, constant, arg_keys in zip(
                self.CHILD_KEYS, self.CHILD_TYPES, child_constants, child_args
            )
        }

        super().__init__((constants, arg_keys), children)

    # @property
    # def constants(self):
    #     return self.value[0]

    @property
    def arg_keys(self):
        return self.value[1]

    def __call__(self, prims: tp.Tuple[Primitive, ...], params: Parameters):
        root_prim = Node(
            np.array([]), {f"arg{n}": prim for n, prim in enumerate(prims)}
        )
        return self.assem_res_from_tree(root_prim, params)

    def assem_res_from_tree(self, root_prim: Node[NDArray, pr.Primitive], params: Parameters):
        # flat_constraints = flatten('', self)
        residuals = tuple(
            constraint.assem_res_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in constraint.arg_keys),
                params
            )
            for _, constraint in iter_flat("", self)
        )
        return jnp.concatenate(residuals)

    def assem_res_atleast_1d(self, prims: tp.Tuple[Primitive, ...], params: Parameters) -> NDArray:
        return jnp.atleast_1d(self.assem_res(prims, params))

    def assem_res(self, prims: tp.Tuple[Primitive, ...], params: Parameters) -> NDArray:
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
    Constrain the distance between two points along a direction
    """

    ARG_TYPES = (pr.Point, pr.Point)
    CONSTANTS = collections.namedtuple("Constants", ["distance", "direction"])

    def assem_res(self, prims, params):
        """
        Return the distance error between two points along a given direction

        The distance is measured from `prims[0]` to `prims[1]` along the direction.
        """
        point0, point1 = prims
        distance = jnp.dot(point1.value - point0.value, params.direction)
        return distance - params.distance


class XDistance(DirectedDistance):
    """
    Constrain the x-distance between two points
    """

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        direction = np.array([1, 0])

        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({"direction": direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        super().__init__(constants, arg_keys)


class YDistance(DirectedDistance):
    """
    Constrain the x-distance between two points
    """

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        direction = np.array([0, 1])
        if isinstance(constants, dict):
            constants = constants.copy()
            constants.update({"direction": direction})
        elif isinstance(constants, tuple):
            constants = constants[:1] + (direction,)

        super().__init__(constants, arg_keys)


class Fix(Constraint):
    """
    Constrain all coordinates of a point
    """

    ARG_TYPES = (pr.Point,)
    CONSTANTS = collections.namedtuple("Constants", ["location"])

    def assem_res(self, prims, params):
        """
        Return the location error for a point
        """
        (point,) = prims
        return point.value - params.location


class Coincident(Constraint):
    """
    Constrain two points to be coincident
    """

    ARG_TYPES = (pr.Point, pr.Point)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return point0.value - point1.value


## Line constraints


class Length(Constraint):
    """
    Constrain the length of a line
    """

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", ["length"])

    def assem_res(self, prims, params):
        """
        Return the length error of a line
        """
        # This sets the length of a line
        (line,) = prims
        vec = line_vector(line)
        return jnp.sum(vec**2) - params.length**2


class RelativeLength(Constraint):
    """
    Constrain the length of a line relative to another line
    """

    CONSTANTS = collections.namedtuple("Constants", ["length"])
    ARG_TYPES = (pr.Line, pr.Line)

    def assem_res(self, prims, params):
        """
        Return the length error of line `prims[0]` relative to line `prims[1]`
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = line_vector(line0)
        vec_b = line_vector(line1)
        return jnp.sum(vec_a**2) - params.length**2 * jnp.sum(vec_b**2)


class RelativeLengthArray(Constraint[RelativeLength]):
    """
    Constrain the relative lengths of a set of lines
    """

    CONSTANTS = collections.namedtuple("Constants", ("lengths",))

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = self.load_constants(constants)
        num_args = len(_constants.lengths) + 1
        self.ARG_TYPES = num_args * (pr.Line,)

        self.CHILD_TYPES = (num_args - 1) * (RelativeLength,)
        self.CHILD_ARGS = tuple(
            (f"arg{n}", f"arg{num_args-1}") for n in range(num_args - 1)
        )
        self.CHILD_PARAMETERS = lambda constants: tuple(
            (length,) for length in constants.lengths
        )
        self.CHILD_KEYS = tuple(f"RelativeLength{n}" for n in range(num_args - 1))

        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


class XDistanceMidpoints(Constraint):
    """
    Constrain the x-distance between two line midpoints
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", ("distance",))

    def assem_res(self, prims, params):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims
        start_points = (line0["Point0"], line1["Point0"])
        end_points = (line0["Point1"], line1["Point1"])
        distance_start = jnp.dot(
            start_points[1].value - start_points[0].value, np.array([1, 0])
        )
        distance_end = jnp.dot(
            end_points[1].value - end_points[0].value, np.array([1, 0])
        )
        # distance_end = 0
        return 1 / 2 * (distance_start + distance_end) - params.distance


class XDistanceMidpointsArray(Constraint[XDistanceMidpoints]):
    """
    Constrain the x-distances between a set of line midpoints
    """

    CONSTANTS = collections.namedtuple("Constants", ("distances",))

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = self.load_constants(constants)
        num_child = len(_constants.distances)

        self.ARG_TYPES = num_child * (pr.Line, pr.Line)
        self.CHILD_TYPES = num_child * (XDistanceMidpoints,)
        self.CHILD_ARGS = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        self.CHILD_KEYS = tuple(f"LineMidpointXDistance{n}" for n in range(num_child))
        self.CHILD_PARAMETERS = lambda constants: tuple(
            (distance,) for distance in constants.distances
        )

        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


class YDistanceMidpoints(Constraint):
    """
    Constrain the y-distance between two line midpoints
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", ("distance",))

    def assem_res(self, prims, params):
        """
        Return the y-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims
        start_points = (line0["Point0"], line1["Point0"])
        end_points = (line0["Point1"], line1["Point1"])
        distance_start = jnp.dot(
            start_points[1].value - start_points[0].value, np.array([0, 1])
        )
        distance_end = jnp.dot(
            end_points[1].value - end_points[0].value, np.array([0, 1])
        )
        # distance_end = 0
        return 1 / 2 * (distance_start + distance_end) - params.distance


class YDistanceMidpointsArray(Constraint[YDistanceMidpoints]):
    """
    Constrain the y-distances between a set of line midpoints
    """

    CONSTANTS = collections.namedtuple("Constants", ("distances",))

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = self.load_constants(constants)
        num_child = len(_constants.distances)

        self.ARG_TYPES = num_child * (pr.Line, pr.Line)
        self.CHILD_TYPES = num_child * (YDistanceMidpoints,)
        self.CHILD_ARGS = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        self.CHILD_KEYS = tuple(f"LineMidpointYDistance{n}" for n in range(num_child))
        self.CHILD_PARAMETERS = lambda constants: tuple(
            (distance,) for distance in constants.distances
        )

        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


class Orthogonal(Constraint):
    """
    Constrain two lines to be orthogonal
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the orthogonal error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.dot(dir0, dir1)


class Parallel(Constraint):
    """
    Constrain two lines to be parallel
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the parallel error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.cross(dir0, dir1)


class Vertical(Constraint):
    """
    Constrain a line to be vertical
    """

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the vertical error for a line
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([1, 0]))


class Horizontal(Constraint):
    """
    Constrain a line to be horizontal
    """

    ARG_TYPES = (pr.Line,)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the horizontal error for a line
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([0, 1]))


class Angle(Constraint):
    """
    Constrain the angle between two lines
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", ["angle"])

    def assem_res(self, prims, params):
        """
        Return the angle error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)

        dir0 = dir0 / jnp.linalg.norm(dir0)
        dir1 = dir1 / jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - params.angle


class Collinear(Constraint):
    """
    Constrain two lines to be collinear
    """

    ARG_TYPES = (pr.Line, pr.Line)
    CONSTANTS = collections.namedtuple("Constants", [])

    def assem_res(self, prims, params):
        """
        Return the collinearity error between two lines
        """
        res_parallel = Parallel({})
        line0, line1 = prims
        line2 = pr.Line(children=(line1[0], line0[0]))
        # line3 = primitives.Line(children=(line1['Point0'], line0['Point1']))

        return jnp.concatenate(
            [res_parallel((line0, line1)), res_parallel((line0, line2))]
        )


class CollinearArray(Constraint[Collinear]):
    """
    Constrain a set of lines to be collinear
    """

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple("Constants", ("size",))

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = self.load_constants(constants)
        size = _constants.size
        if size < 1:
            raise ValueError()

        self.ARG_TYPES = size * (pr.Line,)

        self.CHILD_TYPES = (size - 1) * (Collinear,)
        self.CHILD_ARGS = tuple(("arg0", f"arg{n}") for n in range(1, size))
        self.CHILD_KEYS = tuple(f"Collinear[0][{n}]" for n in range(1, size))
        self.CHILD_PARAMETERS = lambda constants: (constants.size - 1) * ((),)
        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


class CoincidentLines(Constraint):
    """
    Constrain two lines to be coincident
    """

    ARG_TYPES = (pr.Point, pr.Point)
    CONSTANTS = collections.namedtuple("Constants", ["reverse"])

    def assem_res(self, prims, params):
        """
        Return the coincident error between two lines
        """
        line0, line1 = prims
        if not params.reverse:
            point0_err = line1["Point0"].value - line0["Point0"].value
            point1_err = line1["Point1"].value - line0["Point1"].value
        else:
            point0_err = line1["Point0"].value - line0["Point1"].value
            point1_err = line1["Point1"].value - line0["Point0"].value
        return jnp.concatenate([point0_err, point1_err])


## Polygon constraints


class Box(Constraint[Horizontal | Vertical]):
    """
    Constrain a `Quadrilateral` to be rectangular
    """

    ARG_TYPES = (pr.Quadrilateral,)
    CONSTANTS = collections.namedtuple("Constants", [])

    CHILD_KEYS = ("HorizontalBottom", "HorizontalTop", "VerticalLeft", "VerticalRight")
    CHILD_TYPES = (Horizontal, Horizontal, Vertical, Vertical)
    CHILD_ARGS = (("arg0/Line0",), ("arg0/Line2",), ("arg0/Line3",), ("arg0/Line1",))

    def assem_res(self, prims, params):
        return np.array([])


## Grid constraints


def idx_1d(multi_idx: tp.Tuple[int, ...], shape: tp.Tuple[int, ...]):
    """
    Return a 1D array index from a multi-dimensional array index
    """
    strides = shape[1:] + (1,)
    return sum(axis_idx * stride for axis_idx, stride in zip(multi_idx, strides))


class RectilinearGrid(Constraint[CollinearArray]):
    """
    Constrain a set of `Quadrilateral`s to lie on a rectilinear grid
    """

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple("Constants", ("shape",))

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):
        _constants = self.load_constants(constants)
        shape = _constants.shape

        num_row, num_col = shape
        num_args = num_row * num_col

        self.ARG_TYPES = num_args * (pr.Quadrilateral,)

        # Specify child constraints given the grid shape

        # Line up bot/top/left/right
        CHILD_TYPES = 2 * num_row * (CollinearArray,) + 2 * num_col * (CollinearArray,)
        CHILD_ARGS = (
            [
                tuple(
                    f"arg{idx_1d((nrow, ncol), shape)}/Line0" for ncol in range(num_col)
                )
                for nrow in range(num_row)
            ]
            + [
                tuple(
                    f"arg{idx_1d((nrow, ncol), shape)}/Line2" for ncol in range(num_col)
                )
                for nrow in range(num_row)
            ]
            + [
                tuple(
                    f"arg{idx_1d((nrow, ncol), shape)}/Line3" for nrow in range(num_row)
                )
                for ncol in range(num_col)
            ]
            + [
                tuple(
                    f"arg{idx_1d((nrow, ncol), shape)}/Line1" for nrow in range(num_row)
                )
                for ncol in range(num_col)
            ]
        )
        CHILD_KEYS = (
            [f"CollinearRowBottom{nrow}" for nrow in range(num_row)]
            + [f"CollinearRowTop{nrow}" for nrow in range(num_row)]
            + [f"CollinearColumnLeft{ncol}" for ncol in range(num_col)]
            + [f"CollinearColumnRight{ncol}" for ncol in range(num_col)]
        )

        self.CHILD_PARAMETERS = lambda constants: (
            [(constants.shape[1],) for nrow in range(constants.shape[0])]
            + [(constants.shape[1],) for nrow in range(constants.shape[0])]
            + [(constants.shape[0],) for ncol in range(constants.shape[1])]
            + [(constants.shape[0],) for ncol in range(constants.shape[1])]
        )

        self.CHILD_TYPES = CHILD_TYPES
        self.CHILD_ARGS = CHILD_ARGS
        self.CHILD_KEYS = CHILD_KEYS

        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


class Grid(
    Constraint[
        RectilinearGrid
        | RelativeLengthArray
        | XDistanceMidpointsArray
        | YDistanceMidpointsArray
    ]
):
    """
    Constrain a set of `Quadrilateral`s to lie on a dimensioned rectilinear grid
    """

    ARG_TYPES = None
    CONSTANTS = collections.namedtuple(
        "Constants",
        ("shape", "col_margins", "row_margins", "col_widths", "row_heights"),
    )

    def __init__(
        self,
        constants: Constants,
        arg_keys: tp.Tuple[str, ...] = None,
    ):

        _constants = self.load_constants(constants)
        num_args = np.prod(_constants.shape)
        self.ARG_TYPES = num_args * (pr.Quadrilateral,)

        # Children constraints do:
        # 1. Align all quads in a grid
        # 2. Set relative column widths relative to column 0
        # 3. Set relative row heights relative to row 0
        self.CHILD_TYPES = (
            RectilinearGrid,
            RelativeLengthArray,
            RelativeLengthArray,
            XDistanceMidpointsArray,
            YDistanceMidpointsArray,
        )
        self.CHILD_KEYS = (
            "RectilinearGrid",
            "ColumnWidths",
            "RowHeights",
            "ColumnMargins",
            "RowMargins",
        )

        shape = _constants.shape
        rows, cols = list(range(shape[0])), list(range(shape[1]))

        col_margin_line_labels = itertools.chain.from_iterable(
            (
                f"arg{idx_1d((0, col), shape)}/Line1",
                f"arg{idx_1d((0, col+1), shape)}/Line3",
            )
            for col in cols[:-1]
        )
        row_margin_line_labels = itertools.chain.from_iterable(
            (
                f"arg{idx_1d((row+1, 0), shape)}/Line2",
                f"arg{idx_1d((row, 0), shape)}/Line0",
            )
            for row in rows[:-1]
        )

        self.CHILD_ARGS = (
            tuple(f"arg{n}" for n in range(num_args)),
            tuple(
                f"arg{idx_1d((row, col), shape)}/Line0"
                for row, col in itertools.product([0], cols[1:] + cols[:1])
            ),
            tuple(
                f"arg{idx_1d((row, col), shape)}/Line1"
                for row, col in itertools.product(rows[1:] + rows[:1], [0])
            ),
            tuple(col_margin_line_labels),
            tuple(row_margin_line_labels),
        )
        self.CHILD_PARAMETERS = lambda constants: (
            {"shape": constants.shape},
            {"lengths": constants.col_widths},
            {"lengths": constants.row_heights},
            {"distances": constants.col_margins},
            {"distances": constants.row_margins},
        )

        super().__init__(constants, arg_keys)

    def assem_res(self, prims, params):
        return np.array([])


def line_vector(line: pr.Line):
    return line[1].value - line[0].value
