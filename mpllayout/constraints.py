"""
Geometric constraints
"""

import typing as tp
from numpy.typing import NDArray

from collections import namedtuple
import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat

Primitive = pr.Primitive


ResConstants = namedtuple("Constants", ())

ResParams = namedtuple("Parameters", ())
ResParamsType = type[ResParams]

ResPrims = tp.Tuple[Primitive, ...]
ResPrimTypes = tp.Tuple[type[Primitive], ...]

PrimKeys = tp.Tuple[str, ...]
ChildrenPrimKeys = tp.Tuple[PrimKeys, ...]

ConstraintValue = tp.Tuple[ResConstants, ResPrimTypes, ResParamsType, ChildrenPrimKeys]

def load_named_tuple(
        NamedTuple: namedtuple,
        args: tp.Mapping[str, tp.Any] | tp.Tuple[tp.Any, ...]
    ):
    if isinstance(args, dict):
        args = NamedTuple(**args)
    elif isinstance(args, tuple):
        args = NamedTuple(*args)
    elif isinstance(args, NamedTuple):
        pass
    else:
        raise TypeError()
    return args


class PrimKeysNode(Node[PrimKeys, "PrimKeysNode"]):
    """
    Tree of primitive keys corresponding to a constraint tree

    Each node's value is a tuple of strings indicating primitives for the
    corresponding constraint's `assem_res`.
    """
    pass


class ParamsNode(Node[ResParams, "ParamsNode"]):
    """
    Tree of parameters corresponding to a constraint tree

    Each node's value are parameters indicating for the corresponding constraint's
    `assem_res`.
    """
    pass


class Constraint(Node[ConstraintValue, "Constraint"]):
    """
    Geometric constraint on primitives

    A geometric constraint is a condition on parameters of geometric primitives.
    The condition is implemented through a residual function
        `Constraint.assem_res(prims, **kwargs)`
    where `prims` are geometric primitives and `params` are parameters for the residual.
    For constraint is satisified when `Constraint.assem_res(prims, **kwargs) == 0` for a
    given `prims` and `params`.

    For more details on how to implement `Constaint.assem_res`, see the docstring below.

    Constraints also have a tree structure where constraints can contain child
    constraints. The residual of a constraint is the result of joining all child
    constraint residuals together.

    To create a constraint, you have to subclass `Constraint` then:
        1. Define the residual for the constraint (`assem_res`)
        2. Specify the parameters for `Constraint.__init__` (see below)
    Note that some of the `Constraint.__init__` parameters are for type checking inputs
    to `assem_res` while the others are for specifying child constraints.

    Parameters
    ----------
    # TODO: Remove the constants attribute?
    res_constants:
        Constants for the constraint

        Currently this is either empty or stores a shape
    res_prim_types:
        Primitive types for `assem_res`
    res_params_type:
        Primitive parameter vector named tuple
    children_prim_keys:
        Primitive key tuples for each child constraint

        For a given child constraint, a tuple of primitive keys indicates a subset of
        parent primitives to form child constraint primitive arguments.

        Consider a parent constraint with residual
            ```parent.assem_res(prims, **kwargs)```
        and the nth child constraint with primitive key tuple
            ```children_primkeys[n] = ('arg0', 'arg3/Line2')```.
        This indicates the nth child constraint should be evaluated with
            ```parent.children[n].assem_res((prims[0], prims[3]['Line0']), **child_kwargs)```
    children:
        A dictionary of child constraints
    """

    def __init__(
        self,
        res_constants: ResConstants,
        res_prim_types: ResPrimTypes,
        res_params_type: ResParamsType,
        children_prim_keys: ChildrenPrimKeys,
        children: tp.Mapping[str, "Constraint"]
    ):
        super().__init__((res_constants, res_prim_types, res_params_type, children_prim_keys), children)

    # TODO: Turn this into something passed through __init__ instead!
    # This separate function is confusing because you have to specify a constraint
    # through __init__ and this
    def split_children_params(cls, parameters: ResParams):
        raise NotImplementedError()

    def root_params(self, parameters: ResParams):
        parameters = load_named_tuple(self.RES_PARAMS_TYPE, parameters)

        keys, child_constraints = self.keys(), self.children
        child_parameters = self.split_children_params(parameters)
        children = {
            key: child_constraint.root_params(child_params)
            for key, child_constraint, child_params in zip(keys, child_constraints, child_parameters)
        }
        root_params = ParamsNode(parameters, children)
        return root_params

    def root_prim_keys(self, prim_keys: PrimKeys):
        # Replace the first 'arg{n}/...' key with the appropriate parent argument keys

        def parent_argnum_from_key(arg_key: str):
            arg_number_str = arg_key.split("/", 1)[0]
            if arg_number_str[:3] == "arg":
                arg_number = int(arg_number_str[3:])
            else:
                raise ValueError(f"Argument key, {arg_key}, must contain 'arg' prefix")
            return arg_number

        def replace_prim_key_prefix(arg_key: str, parent_prim_keys):
            split_key = arg_key.split("/", 1)
            prefix, postfix = split_key[0], split_key[1:]
            new_prefix = parent_prim_keys[parent_argnum_from_key(arg_key)]
            return "/".join([new_prefix] + postfix)

        # For each child, find the parent primitive part of its argument tuple
        children_prim_keys = tuple(
            tuple(
                replace_prim_key_prefix(prim_key, prim_keys)
                for prim_key in child_prim_keys
            )
            for child_prim_keys in self.CHILDREN_PRIM_KEYS
        )

        children = {
            key: child.root_prim_keys(child_prim_keys)
            for (key, child), child_prim_keys
            in zip(self.children_map.items(), children_prim_keys)
        }
        return PrimKeysNode(prim_keys, children)

    @property
    def RES_CONSTANTS(self):
        return self.value[0]

    @property
    def RES_PRIM_TYPES(self):
        return self.value[1]

    @property
    def RES_PARAMS_TYPE(self):
        return self.value[2]

    @property
    def CHILDREN_PRIM_KEYS(self):
        return self.value[3]

    def __call__(
            self,
            prims: ResPrims,
            params: tp.Tuple[tp.Any, ...] | tp.Mapping[str, tp.Any]
        ):
        root_prim = pr.PrimitiveNode(
            np.array([]), {f"arg{n}": prim for n, prim in enumerate(prims)}
        )
        root_prim_keys = self.root_prim_keys(root_prim.keys())
        root_params = self.root_params(params)
        return self.assem_res_from_tree(root_prim, root_prim_keys, root_params)

    def assem_res_from_tree(
            self,
            root_prim: pr.PrimitiveNode,
            root_prim_keys: PrimKeysNode,
            root_params: ParamsNode,
        ):
        flat_constraints = (x for _, x in iter_flat("", self))
        flat_prim_keys = (x.value for _, x in iter_flat("", root_prim_keys))
        flat_params = (x.value for _, x in iter_flat("", root_params))

        residuals = tuple(
            constraint.assem_res_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in argkeys), params
            )
            for constraint, argkeys, params in zip(flat_constraints, flat_prim_keys, flat_params)
        )
        return jnp.concatenate(residuals)

    def assem_res_atleast_1d(
            self, prims: ResPrims, params: ResParams
        ) -> NDArray:
        return jnp.atleast_1d(self.assem_res(prims, **params._asdict()))

    # TODO: Replace params with actual keywords arguments? would be more readable
    def assem_res(
            self, prims: ResPrims, **kwargs
        ) -> NDArray:
        """
        Return a residual vector representing the constraint satisfaction

        Parameters
        ----------
        prims: ResPrims
            A tuple of primitives the constraint applies to
        params: ResParams
            A set of parameters for the residual

            These are things like length, distance, angle, etc.

        Returns
        -------
        NDArray
            The residual representing whether the constraint is satisfied. The
            constraint is satisfied when the residual is 0.
        """
        raise NotImplementedError()


class ConstraintNode(Node[ConstraintValue, Constraint]):
    """
    Container tree for constraints
    """
    pass


ChildKeys = tp.Tuple[str, ...]
ChildConstraints = tp.Tuple[Constraint, ...]

class StaticConstraint(Constraint):
    """
    Constraint with static number of arguments and/or children

    To specify a `StaticConstraint` you have to define `init_tree` to return
    parameters for `Constraint.__init__`.
    """

    @classmethod
    def init_tree(
        cls
    ) -> tp.Tuple[ConstraintValue, tp.Tuple[ChildKeys, ChildConstraints]]:
        raise NotImplementedError()

    def split_children_params(self, parameters):
        return tuple(child.RES_PARAMS_TYPE() for child in self.children)

    def __init__(self):
        (ARG_TYPES, ARG_PARAMETERS, CHILDREN_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS) = self.init_tree()

        constants = namedtuple("Constants", ())

        children = {
            key: constraint
            for key, constraint in zip(CHILD_KEYS, CHILD_CONSTRAINTS)
        }

        super().__init__(constants, ARG_TYPES, ARG_PARAMETERS, CHILDREN_ARGKEYS, children)


class ParameterizedConstraint(Constraint):
    """
    Constraint parameterized by generic parameters

    To specify a `ParameterizedConstraint` you have to define `init_tree` to
    return parameters for `Constraint.__init__`.
    """

    @classmethod
    def init_tree(
        cls,
        **kwargs
    ) -> tp.Tuple[ConstraintValue, tp.Tuple[ChildKeys, ChildConstraints]]:
        raise NotImplementedError()

    def split_children_params(self, parameters):
        raise NotImplementedError()

    def __init__(self, **kwargs):

        constants = namedtuple("Constants", tuple(kwargs.keys()))(**kwargs)

        (ARG_TYPES, ARG_PARAMETERS, CHILDREN_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS) = self.init_tree(**kwargs)

        children = {key: constraint for key, constraint in zip(CHILD_KEYS, CHILD_CONSTRAINTS)}

        super().__init__(constants, ARG_TYPES, ARG_PARAMETERS, CHILDREN_ARGKEYS, children)


class DynamicConstraint(ParameterizedConstraint):
    """
    Constraint with dynamic number of arguments and/or children depending on a shape

    To specify a `DynamicConstraint` you have to define `init_tree` to return
    parameters for `Constraint.__init__`.
    """

    @classmethod
    def init_tree(
        cls,
        shape: tp.Tuple[int, ...]
    ) -> tp.Tuple[ConstraintValue, tp.Tuple[ChildKeys, ChildConstraints]]:
        raise NotImplementedError()

    def split_children_params(self, parameters):
        raise NotImplementedError()

    def __init__(self, shape: tp.Tuple[int, ...]=(0,)):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(shape=shape)


## Point constraints
# NOTE: These are actual constraint classes that can be called so class docstrings
# document there `assem_res` function.

# Argument type: Tuple[Point,]

class Fix(StaticConstraint):
    """
    Constrain coordinates of a point

    Parameters
    ----------
    prims: tp.Tuple[pr.Point]
        The point
    location: NDArray
        The coordinates
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point,)
        ARG_PARAMETERS = namedtuple("Parameters", ("location",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Point], location: NDArray):
        """
        Return the location error for a point
        """
        (point,) = prims
        return point.value - location

# Argument type: Tuple[Point, Point]

class DirectedDistance(StaticConstraint):
    """
    Constrain the distance between two points along a direction

    Parameters
    ----------
    prims: tp.Tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    distance: float
        The distance
    direction: NDArray
        The direction
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point, pr.Point)
        ARG_PARAMETERS = namedtuple("Parameters", ("distance", "direction"))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(
        self,
        prims: tp.Tuple[pr.Point, pr.Point],
        distance: float=0.0,
        direction: NDArray=np.zeros(2)
    ):
        """
        Return the distance error between two points along a given direction

        The distance is measured from `prims[0]` to `prims[1]` along the direction.
        """
        point0, point1 = prims
        proj_distance = jnp.dot(point1.value - point0.value, direction)
        return proj_distance - distance


class XDistance(StaticConstraint):
    """
    Constrain the x-distance between two points

    Parameters
    ----------
    prims: tp.Tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    distance: float
        The distance
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point, pr.Point)
        ARG_PARAMETERS = namedtuple("Parameters", ("distance",))

        CHILD_KEYS = ("DirectedDistance",)
        CHILD_CONSTRAINTS = (DirectedDistance(),)
        CHILD_ARGKEYS = (("arg0", "arg1"),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return ({"distance": params.distance, "direction": np.array([1, 0])},)

    def assem_res(
        self, prims: tp.Tuple[pr.Point, pr.Point], distance: float=0.0
    ):
        return np.array([])


class YDistance(StaticConstraint):
    """
    Constrain the y-distance between two points

    Parameters
    ----------
    prims: tp.Tuple[pr.Point, pr.Point]
        The two points

        Distance is measured from the first to the second point
    distance: float
        The distance
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point, pr.Point)
        ARG_PARAMETERS = namedtuple("Parameters", ("distance",))

        CHILD_KEYS = ("DirectedDistance",)
        CHILD_CONSTRAINTS = (DirectedDistance(),)
        CHILD_ARGKEYS = (("arg0", "arg1"),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return ({"distance": params.distance, "direction": np.array([0, 1])},)

    def assem_res(
        self, prims: tp.Tuple[pr.Point, pr.Point], distance: float=0.0
    ):
        return np.array([])


class Coincident(StaticConstraint):
    """
    Constrain two points to be coincident

    Parameters
    ----------
    prims: tp.Tuple[pr.Point, pr.Point]
        The two points
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point, pr.Point)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Point, pr.Point]):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return point0.value - point1.value


## Line constraints

# Argument type: Tuple[Line,]

class Length(StaticConstraint):
    """
    Constrain the length of a line

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The line
    length: float
        The length
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ("length",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line], length: float=0):
        """
        Return the length error of a line
        """
        # This sets the length of a line
        (line,) = prims
        vec = line_vector(line)
        return jnp.sum(vec**2) - length**2


class Vertical(StaticConstraint):
    """
    Constrain a line to be vertical

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line]):
        """
        Return the vertical error for a line
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([1, 0]))


class Horizontal(StaticConstraint):
    """
    Constrain a line to be horizontal

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line]):
        """
        Return the horizontal error for a line
        """
        (line0,) = prims
        dir0 = line_vector(line0)
        return jnp.dot(dir0, np.array([0, 1]))


class DirectedLength(StaticConstraint):
    """
    Constrain the length of a line projected along a vector

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The line
    length: float
        The length
    direction: NDArray
        The direction
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ("length", "direction"))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(
        self,
        prims: tp.Tuple[pr.Line],
        length: float=0,
        direction: NDArray=np.array([1, 0])
    ):
        """
        Return the length error of a line
        """
        # This sets the length of a line
        (line,) = prims
        vec = line_vector(line)
        return jnp.dot(vec, direction) - length


class XLength(StaticConstraint):
    """
    Constrain the length of a line projected along the x direction

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The line
    length: float
        The length
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ("length",))

        CHILD_KEYS = ("DirectedLength",)
        CHILD_CONSTRAINTS = (DirectedLength(),)
        CHILD_ARGKEYS = (("arg0",),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return ({"length": params.length, "direction": np.array([1, 0])},)

    def assem_res(self, prims: tp.Tuple[pr.Line], length: float=0):
        """
        Return the length error of a line
        """
        return np.array([])


class YLength(StaticConstraint):
    """
    Constrain the length of a line projected along the y direction

    Parameters
    ----------
    prims: tp.Tuple[pr.Line]
        The line
    length: float
        The length
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ("length",))

        CHILD_KEYS = ("DirectedLength",)
        CHILD_CONSTRAINTS = (DirectedLength(),)
        CHILD_ARGKEYS = (("arg0",),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return ({"length": params.length, "direction": np.array([0, 1])},)

    def assem_res(self, prims: tp.Tuple[pr.Line], length: float=0):
        """
        Return the length error of a line
        """
        return np.array([])


# Argument type: Tuple[Line, Line]

class RelativeLength(StaticConstraint):
    """
    Constrain the length of a line relative to another line

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines

        The length of the first line is measured relative to the second line
    length: float
        The relative length
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("length",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line], length: float=1):
        """
        Return the length error of line `prims[0]` relative to line `prims[1]`
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = line_vector(line0)
        vec_b = line_vector(line1)
        return jnp.sum(vec_a**2) - length**2 * jnp.sum(vec_b**2)


class MidpointXDistance(StaticConstraint):
    """
    Constrain the x-distance between two line midpoints

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines

        The distance is measured from the first to the second line
    distance: float
        The distance
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("distance",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line], distance: float=0):
        """
        Return the x-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims

        midpoint0 = 1/2*(line0["Point0"].value + line0["Point1"].value)
        midpoint1 = 1/2*(line1["Point0"].value + line1["Point1"].value)

        mid_distance = jnp.dot(midpoint1 - midpoint0, np.array([1, 0]))
        return mid_distance - distance


class MidpointYDistance(StaticConstraint):
    """
    Constrain the y-distance between two line midpoints

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines

        The distance is measured from the first to the second line
    distance: float
        The distance
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("distance",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line], distance: float=0):
        """
        Return the y-distance error from the midpoint of line `prims[0]` to `prims[1]`
        """
        line0, line1 = prims

        midpoint0 = 1/2*(line0["Point0"].value + line0["Point1"].value)
        midpoint1 = 1/2*(line1["Point0"].value + line1["Point1"].value)

        mid_distance = jnp.dot(midpoint1 - midpoint0, np.array([0, 1]))
        return mid_distance - distance


class Orthogonal(StaticConstraint):
    """
    Constrain two lines to be orthogonal

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the orthogonal error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.dot(dir0, dir1)


class Parallel(StaticConstraint):
    """
    Constrain two lines to be parallel

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the parallel error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.cross(dir0, dir1)


class Angle(StaticConstraint):
    """
    Constrain the angle between two lines

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines
    angle: float
        The angle
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("angle",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line], angle: float=0):
        """
        Return the angle error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)

        dir0 = dir0 / jnp.linalg.norm(dir0)
        dir1 = dir1 / jnp.linalg.norm(dir1)
        return jnp.arccos(jnp.dot(dir0, dir1)) - angle


class Collinear(StaticConstraint):
    """
    Constrain two lines to be collinear

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line]):
        """
        Return the collinearity error between two lines
        """
        res_parallel = Parallel()
        line0, line1 = prims
        line2 = pr.Line(children=(line1[0], line0[0]))
        # line3 = primitives.Line(children=(line1['Point0'], line0['Point1']))

        return jnp.concatenate(
            [res_parallel((line0, line1), ()), res_parallel((line0, line2), ())]
        )


class CoincidentLines(StaticConstraint):
    """
    Constrain two lines to be coincident

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, pr.Line]
        The lines
    reverse: bool
        A boolean indicating whether to coincide lines in the same or reverse directions
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Point, pr.Point)
        ARG_PARAMETERS = namedtuple("Parameters", ("reverse",))

        CHILD_ARGKEYS = ()
        CHILD_KEYS, CHILD_CONSTRAINTS = (), ()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Line, pr.Line], reverse=False):
        """
        Return the coincident error between two lines
        """
        line0, line1 = prims
        if not reverse:
            point0_err = line1["Point0"].value - line0["Point0"].value
            point1_err = line1["Point1"].value - line0["Point1"].value
        else:
            point0_err = line1["Point0"].value - line0["Point1"].value
            point1_err = line1["Point1"].value - line0["Point0"].value
        return jnp.concatenate([point0_err, point1_err])

# Argument type: Tuple[Line, ...]

class RelativeLengthArray(DynamicConstraint):
    """
    Constrain the lengths of a set of lines relative to the last

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, ...]
        The lines

        The length of the lines are measured relative to the last line
    lengths: NDArray
        The relative lengths
    """

    @classmethod
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        size = np.prod(shape)

        ARG_TYPES = size * (pr.Line,) + (pr.Line,)
        ARG_PARAMETERS = namedtuple("Parameters", ("lengths",))

        CHILD_KEYS = tuple(f"RelativeLength{n}" for n in range(size))
        CHILD_CONSTRAINTS = (size) * (RelativeLength(),)
        CHILD_ARGKEYS = tuple(
            (f"arg{n}", f"arg{size}")
            for n in range(size)
        )

        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        num_args = np.prod(self.RES_CONSTANTS.shape)

        return tuple(
            child.RES_PARAMS_TYPE(length)
            for child, length in zip(self.children, parameters.lengths)
        )

    def assem_res(self, prims: tp.Tuple[pr.Line, ...], lengths: NDArray):
        return np.array([])


class MidpointXDistanceArray(DynamicConstraint):
    """
    Constrain the x-distances between a set of line midpoints

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, ...]
        The lines

        The distances are measured from the first to the second line in pairs
    distances: NDArray
        The distances
    """

    CONSTANTS = namedtuple("Constants", ("distances",))

    @classmethod
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        num_child = np.prod(shape)

        ARG_TYPES = num_child * (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("distances",))
        CHILD_ARGKEYS = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        CHILD_KEYS = tuple(f"LineMidpointXDistance{n}" for n in range(num_child))
        CHILD_CONSTRAINTS = num_child * (MidpointXDistance(),)

        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return tuple(
            child.RES_PARAMS_TYPE(distance)
            for child, distance in zip(self.children, params.distances)
        )

    def assem_res(self, prims: tp.Tuple[pr.Line, ...], distances: NDArray):
        return np.array(())


class MidpointYDistanceArray(DynamicConstraint):
    """
    Constrain the y-distances between a set of line midpoints

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, ...]
        The lines

        The distances are measured from the first to the second line in pairs
    distances: NDArray
        The distances
    """

    @classmethod
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        num_child = np.prod(shape)

        ARG_TYPES = num_child * (pr.Line, pr.Line)
        ARG_PARAMETERS = namedtuple("Parameters", ("distances",))
        CHILD_ARGKEYS = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        CHILD_KEYS = tuple(f"LineMidpointYDistance{n}" for n in range(num_child))
        CHILD_CONSTRAINTS = num_child * (MidpointYDistance(),)

        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        return tuple(
            child.RES_PARAMS_TYPE(distance)
            for child, distance in zip(self.children, params.distances)
        )

    def assem_res(self, prims: tp.Tuple[pr.Line, ...], distances: NDArray):
        return np.array(())


class CollinearArray(DynamicConstraint):
    """
    Constrain a set of lines to be collinear

    Parameters
    ----------
    prims: tp.Tuple[pr.Line, ...]
        The lines
    """

    @classmethod
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        size = np.prod(shape)

        ARG_TYPES = size * (pr.Line, )
        ARG_PARAMETERS = namedtuple("Parameters", ())
        CHILD_ARGKEYS = tuple(("arg0", f"arg{n}") for n in range(1, size))
        CHILD_KEYS = tuple(f"Collinear[0][{n}]" for n in range(1, size))
        CHILD_CONSTRAINTS = size * (Collinear(),)

        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        return tuple(child.RES_PARAMS_TYPE() for child in self.children)

    def assem_res(self, prims: tp.Tuple[pr.Line, ...]):
        return np.array([])

## Point and Line constraints

# TODO:
# class BoundPointsByLine(DynamicConstraint)
# A class that ensures all points have orthogonal distance to a line > offset
# The orthogonal direction should be rotated 90 degrees from the line direction
# (or some other convention)
# You should also have some convention to specify whether you bound for positive
# distance or negative distance
# This would be like saying the points all lie to the left or right of the line
# + and offset
# This would be useful for aligning axis labels

# TODO:
# class DistanceOnLine(StaticConstraint)
# This would constraint the projected absolute distance of a point on a line

# TODO:
# class RelativeDistanceOnLine(StaticConstraint)
# This would constraint the projected distance of a point on a line relative to the line
# length (i.e. 0 and 1 would correspond to the start and end points of the line)
# This would be useful to put axis label halfway along an axis for example!

# TODO:
# class DistanceToLine(StaticConstraint)
# This would constraint the projected distance of a point to a line
# You would need a convention of which "distance" is positive by picking an orthogonal
# to the line
# This would be useful to constrain a point to line on a line or off set a line from a point


## Quad constraints

# Argument type: Tuple[Quadrilateral]

class Box(StaticConstraint):
    """
    Constrain a quadrilateral to be rectangular

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral]
        The quad
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ())
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ("HorizontalBottom", "HorizontalTop", "VerticalLeft", "VerticalRight")
        CHILD_CONSTRAINTS = (Horizontal(), Horizontal(), Vertical(), Vertical())
        CHILD_ARGKEYS = (("arg0/Line0",), ("arg0/Line2",), ("arg0/Line3",), ("arg0/Line1",))
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral]):
        return np.array(())


class AspectRatio(StaticConstraint):
    """
    Constrain the aspect ratio of a quadrilateral

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral]
        The quad
    ar: float
        The aspect ratio
    """

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ("ar",))
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ("RelativeLength",)
        CHILD_CONSTRAINTS = (RelativeLength(), )
        CHILD_ARGKEYS = (("arg0/Line0", "arg0/Line1"),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        return ({'length': parameters.ar},)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral], ar:float=1):
        return np.array(())


# Argument type: Tuple[Quadrilateral, Quadrilateral]

class OuterMargin(ParameterizedConstraint):
    """
    Constrain the outer margin between two quadrilaterals

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    margin: float
    """

    @classmethod
    def init_tree(cls, side: str="left"):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ("margin",))
        CONSTANTS = namedtuple("Constants", ())

        if side == "left":
            CHILD_KEYS = ("MidpointXDistance",)
            CHILD_CONSTRAINTS = (MidpointXDistance(),)
            CHILD_ARGKEYS = (("arg1/Line1", "arg0/Line3"),)
        elif side == "right":
            CHILD_KEYS = ("MidpointXDistance",)
            CHILD_CONSTRAINTS = (MidpointXDistance(),)
            CHILD_ARGKEYS = (("arg0/Line1", "arg1/Line3"),)
        elif side == "bottom":
            CHILD_KEYS = ("MidpointYDistance",)
            CHILD_CONSTRAINTS = (MidpointYDistance(),)
            CHILD_ARGKEYS = (("arg1/Line2", "arg0/Line0"),)
        elif side == "top":
            CHILD_KEYS = ("MidpointYDistance",)
            CHILD_CONSTRAINTS = (MidpointYDistance(),)
            CHILD_ARGKEYS = (("arg0/Line2", "arg1/Line0"),)
        else:
            raise ValueError()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    def split_children_params(self, params):
        return ({"distance": params.margin},)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral, pr.Quadrilateral], margin: float=0):
        return np.array(())


class InnerMargin(ParameterizedConstraint):
    """
    Constrain the inner margin between two quadrilaterals

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    margin: float
    """

    @classmethod
    def init_tree(cls, side: str="left"):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ("margin",))
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ("MidpointXDistance",)
        if side == "left":
            CHILD_CONSTRAINTS = (MidpointXDistance(),)
            CHILD_ARGKEYS = (("arg1/Line3", "arg0/Line3"),)
        elif side == "right":
            CHILD_CONSTRAINTS = (MidpointXDistance(),)
            CHILD_ARGKEYS = (("arg0/Line1", "arg1/Line1"),)
        elif side == "bottom":
            CHILD_CONSTRAINTS = (MidpointYDistance(),)
            CHILD_ARGKEYS = (("arg1/Line0", "arg0/Line0"),)
        elif side == "top":
            CHILD_CONSTRAINTS = (MidpointYDistance(),)
            CHILD_ARGKEYS = (("arg0/Line2", "arg1/Line2"),)
        else:
            raise ValueError()
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    def split_children_params(self, params):
        return ({"distance": params.margin},)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral, pr.Quadrilateral], margin: float=0):
        return np.array(())


# Argument type: Tuple[Quadrilateral, ...]

def idx_1d(multi_idx: tp.Tuple[int, ...], shape: tp.Tuple[int, ...]):
    """
    Return a 1D array index from a multi-dimensional array index
    """
    strides = shape[1:] + (1,)
    return sum(axis_idx * stride for axis_idx, stride in zip(multi_idx, strides))

class RectilinearGrid(DynamicConstraint):
    """
    Constrain a set of quads to lie on a rectilinear grid

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral, ...]
        The quadrilaterals
    """

    @classmethod
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        size = np.prod(shape)
        num_row, num_col = shape

        ARG_TYPES = size * (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ())

        def idx(i, j):
            return idx_1d((i, j), shape)

        # Specify child constraints given the grid shape
        # Line up bottom/top and left/right
        CHILD_CONSTRAINTS = (
            2 * num_row * (CollinearArray(num_col),)
            + 2 * num_col * (CollinearArray(num_row),)
        )
        align_bottom = [
            tuple(f"arg{idx(nrow, ncol)}/Line0" for ncol in range(num_col))
            for nrow in range(num_row)
        ]
        align_top = [
            tuple(f"arg{idx(nrow, ncol)}/Line2" for ncol in range(num_col))
            for nrow in range(num_row)
        ]
        align_left = [
            tuple(f"arg{idx(nrow, ncol)}/Line3" for nrow in range(num_row))
            for ncol in range(num_col)
        ]
        align_right = [
            tuple(f"arg{idx(nrow, ncol)}/Line1" for nrow in range(num_row))
            for ncol in range(num_col)
        ]
        CHILD_ARGKEYS = align_bottom + align_top + align_left + align_right
        CHILD_KEYS = (
            [f"CollinearRowBottom{nrow}" for nrow in range(num_row)]
            + [f"CollinearRowTop{nrow}" for nrow in range(num_row)]
            + [f"CollinearColumnLeft{ncol}" for ncol in range(num_col)]
            + [f"CollinearColumnRight{ncol}" for ncol in range(num_col)]
        )
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        return tuple(child.RES_PARAMS_TYPE() for child in self.children)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral, ...]):
        return np.array(())


class Grid(DynamicConstraint):
    """
    Constrain a set of quads to lie on a dimensioned rectilinear grid

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral, ...]
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
    def init_tree(cls, shape: tp.Tuple[int, ...]):
        num_args = np.prod(shape)
        num_row, num_col = shape

        ARG_TYPES = num_args * (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple(
            "Parameters",
            ("col_widths", "row_heights", "col_margins", "row_margins"),
        )

        # Children constraints do:
        # 1. Align all quads in a grid
        # 2. Set relative column widths relative to column 0
        # 3. Set relative row heights relative to row 0
        CHILD_KEYS = (
            "RectilinearGrid",
            "ColumnWidths",
            "RowHeights",
            "ColumnMargins",
            "RowMargins",
        )
        CHILD_CONSTRAINTS = (
            RectilinearGrid(shape),
            RelativeLengthArray(num_col-1),
            RelativeLengthArray(num_row-1),
            MidpointXDistanceArray(num_col-1),
            MidpointYDistanceArray(num_row-1),
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
        col_margin_line_labels = itertools.chain.from_iterable(
            (f"arg{idx(0, col)}/Line1", f"arg{idx(0, col+1)}/Line3")
            for col in cols[:-1]
        )
        row_margin_line_labels = itertools.chain.from_iterable(
            (f"arg{idx(row+1, 0)}/Line2", f"arg{idx(row, 0)}/Line0")
            for row in rows[:-1]
        )

        CHILD_ARGKEYS = (
            rectilineargrid_args,
            colwidth_args,
            rowheight_args,
            tuple(col_margin_line_labels),
            tuple(row_margin_line_labels),
        )

        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, params):
        values = (
            (),
            (params.col_widths,),
            (params.row_heights,),
            (params.col_margins,),
            (params.row_margins,)
        )
        return tuple(
            load_named_tuple(child.RES_PARAMS_TYPE, value)
            for child, value in zip(self.children, values)
        )

    def assem_res(
        self,
        prims: tp.Tuple[pr.Quadrilateral, ...],
        col_widths: NDArray,
        row_heights: NDArray,
        col_margins: NDArray,
        row_margins: NDArray
    ):
        return np.array([])

# TODO: Incorporate this into primitives?
def line_vector(line: pr.Line):
    return line[1].value - line[0].value


## Axes constraints

from matplotlib.axis import XAxis, YAxis

# Argument type: Tuple[Quadrilateral]

def get_axis_dim(axis: XAxis | YAxis, side: str):

    # Ignore the axis label in the height by temporarily making it invisible
    label_visibility = axis.label.get_visible()
    axis.label.set_visible(False)

    axis_bbox = axis.get_tightbbox()

    if axis_bbox is None:
        height = 0
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

class XAxisHeight(StaticConstraint):
    """
    Constrain the x-axis height for an axes

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral]
        The axes
    axis: XAxis
        The XAxis
    """

    @staticmethod
    def get_xaxis_height(axis: XAxis):
        return get_axis_dim(axis, axis.get_ticks_position())

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ('axis',))
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ("Height",)
        CHILD_CONSTRAINTS = (YDistance(),)
        CHILD_ARGKEYS = (("arg0/Line1/Point0", "arg0/Line1/Point1"),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        xaxis: XAxis | None = parameters.axis
        if xaxis is None:
            return ({"distance": 0},)
        else:
            return ({"distance": self.get_xaxis_height(xaxis)},)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral], axis: XAxis):
        return np.array([])


class YAxisWidth(StaticConstraint):
    """
    Constrain the y-axis width for an axes

    Parameters
    ----------
    prims: tp.Tuple[pr.Quadrilateral]
        The axes
    axis: YAxis
        The YAxis
    """

    @staticmethod
    def get_yaxis_width(axis: YAxis):
        return get_axis_dim(axis, axis.get_ticks_position())

    @classmethod
    def init_tree(cls):
        ARG_TYPES = (pr.Quadrilateral,)
        ARG_PARAMETERS = namedtuple("Parameters", ('axis',))
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ("Width",)
        CHILD_CONSTRAINTS = (XDistance(),)
        CHILD_ARGKEYS = (("arg0/Line0/Point0", "arg0/Line0/Point1"),)
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    def split_children_params(self, parameters):
        yaxis: YAxis | None = parameters.axis
        if yaxis is None:
            return ({"distance": 0},)
        else:
            return ({"distance": self.get_yaxis_width(yaxis)},)

    def assem_res(self, prims: tp.Tuple[pr.Quadrilateral], axis: YAxis):
        return np.array([])

# Argument type: Tuple[Axes]

class PositionXAxis(ParameterizedConstraint):
    """
    Constrain the x-axis to the top or bottom of an axes

    Parameters
    ----------
    prims: tp.Tuple[pr.AxesY | pr.AxesXY]
        The axes
    """

    @classmethod
    def init_tree(cls, bottom: bool, top: bool):
        # TODO: Handle more specialized x/y axes combos?
        ARG_TYPES = (pr.AxesXY,)
        ARG_PARAMETERS = namedtuple("Parameters", ())
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ('CoincidentLines',)
        CHILD_CONSTRAINTS = (CoincidentLines(),)
        if bottom:
            CHILD_ARGKEYS = (('arg0/Frame/Line0', 'arg0/XAxis/Line2'),)
        elif top:
            CHILD_ARGKEYS = (('arg0/Frame/Line2', 'arg0/XAxis/Line0'),)
        else:
            raise ValueError(
                "Currently, 'bottom' and 'top' can't both be true"
            )
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    @classmethod
    def split_children_params(cls, params):
        return ({"reverse": True},)

    def __init__(self, bottom: bool=True, top: bool=False):
        return super().__init__(bottom=bottom, top=top)

    def assem_res(self, prims: tp.Tuple[pr.AxesXY]):
        return np.array([])


class PositionYAxis(ParameterizedConstraint):
    """
    Constrain the y-axis to the left or right of an axes

    Parameters
    ----------
    prims: tp.Tuple[pr.AxesY | pr.AxesXY]
        The axes
    """

    @classmethod
    def init_tree(cls, left: bool=True, right: bool=False):
        # TODO: Handle more specialized x/y axes combos?
        ARG_TYPES = (pr.AxesXY,)
        ARG_PARAMETERS = namedtuple("Parameters", ())
        CONSTANTS = namedtuple("Constants", ())

        CHILD_KEYS = ('CoincidentLines',)
        CHILD_CONSTRAINTS = (CoincidentLines(),)
        if left:
            CHILD_ARGKEYS = (('arg0/Frame/Line3', 'arg0/YAxis/Line1'),)
        elif right:
            CHILD_ARGKEYS = (('arg0/Frame/Line1', 'arg0/YAxis/Line3'),)
        else:
            raise ValueError(
                "Currently, 'left' and 'right' can't both be true"
            )
        return (ARG_TYPES, ARG_PARAMETERS, CHILD_ARGKEYS), (CHILD_KEYS, CHILD_CONSTRAINTS)

    @classmethod
    def split_children_params(cls, params):
        return ({"reverse": True},)

    def __init__(self, left: bool=True, right: bool=False):
        return super().__init__(left=left, right=right)

    def assem_res(self, prims: tp.Tuple[pr.AxesXY]):
        return np.array([])

