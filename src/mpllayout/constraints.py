"""
Geometric constraints
"""

from typing import Optional, Any
from numpy.typing import NDArray

from collections import namedtuple
import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat
from . import constructions as co

Primitive = pr.Primitive


ResParams = dict[str, Any]

ResPrims = tuple[Primitive, ...]
ResPrimTypes = tuple[type[Primitive], ...]

PrimKeys = tuple[str, ...]
ChildPrimKeys = tuple[PrimKeys, ...]

def load_named_tuple(
        NamedTuple: namedtuple,
        args: dict[str, Any] | tuple[Any, ...]
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
    A tree of primitive keys indicating primitives for a constraint

    The tree structure should match the tree structure of a constraint such that
    for each constraint node, there is a corresponding primitive keys node.
    """
    pass


class ParamsNode(Node[ResParams, "ParamsNode"]):
    """
    A tree of residual parameters (kwargs) for a constraint

    The tree structure should match the tree structure of a constraint such that
    for each constraint node, there is a corresponding parameters node.
    """
    pass

# TODO: Add constraint class that accepts a unit
# This would handle the case of setting a length relative to another one

class Constraint(Node[ChildPrimKeys, "Constraint"]):
    """
    The base geometric constraint class

    A geometric constraint represents a condition on the parameter vectors of
    geometric primitives.
    The condition is implemented through a residual function
        `assem_res(self, prims, **kwargs)`
    where
    - `prims` are the geometric primitives to constraint
    - and `**kwargs` are additional arguments for the residual.
    The constraint is satisified when `assem_res(self, prims, **kwargs) == 0`.
    To implement `assem_res`, `jax` functions should be used to return a
    residual vector from the parameter vectors of input primitives.

    Constraints have a tree-like structure.
    A constraint can contain child constraints by passing subsets of its input
    primitives (`prims` in `assem_res`) on to child constraints.
    The residual of a constraint is the result of concatenating all child
    constraint residuals.

    To create a constraint, subclass `Constraint` then:
        1. Define the residual for the constraint (`assem_res`)
        2. Specify the parameters for `Constraint.__init__` (see `Parameters`)\
        3. Define `Constraint.split_children_params` (see below)
    Note that some of the `Constraint.__init__` parameters are for type checking
    inputs to `assem_res` while the others are for specifying child constraints.
    `StaticConstraint` and `ParameterizedConstraint` are two `Constraint`
    subclasses that can be subclassed to create constraints.

    Parameters
    ----------
    child_prim_keys: tuple[PrimKeys, ...]
        Primitive key tuples for each child constraint

        This is stored as the "value" of the tree structure and encodes how to
        create primitives for each child constraint from the parent constraint's
        `prims` (in `assem_res(self, prims, **kwargs)`).
        For each child constraint, a tuple of primitive keys indicates a
        subset of parent primitives to form child constraint primitive
        arguments.

        To illustrate this, consider a parent constraint with residual

        ```python
        Parent.assem_res(self, prims, **kwargs)
        ```

        and the n'th child constraint with primitive key tuple

        ```python
        child_prim_keys[n] == ('arg0', 'arg3/Line2')
        ```.

        This indicates the n'th child constraint should be evaluated with

        ```
        child_prims = (prims[0], prims[3]['Line2'])
        ```
    child_keys: list[str]
        Keys for any child constraints
    child_contraints: list[Constraint]
        Child constraints
    aux_data: Mapping[str, Any]
        Any auxiliary data

        This is usually for type checking/validation of inputs
    """

    # TODO: Implement `assem_res` type checking using `aux_data`
    def __init__(
        self,
        child_prim_keys: ChildPrimKeys,
        child_keys: list[str],
        child_constraints: list["Constraint"],
        aux_data: Optional[dict[str, Any]] = None
    ):
        children = {
            key: child for key, child in zip(child_keys, child_constraints)
        }
        super().__init__((child_prim_keys, aux_data), children)

    # TODO: Make this something that's passed through __init__?
    # That would make it harder to forget defining this?
    def propogate_child_params(self, params: ResParams) -> tuple[ResParams, ...]:
        """
        Return children constraint parameters from parent constraint parameters
        """
        raise NotImplementedError()

    def root_params(self, params: ResParams) -> ParamsNode:
        """
        Return a tree of residual kwargs for the constraint and all children

        The tree structure should match the tree structure of the constraint.

        Parameters
        ----------
        params: ResParams
            Residual keyword arguments for the constraint

        Returns
        -------
        root_params: ParamsNode
            A tree of keyword arguments for the constraint and all children
        """
        params = load_named_tuple(self.RES_PARAMS_TYPE, params)

        child_parameters = self.propogate_child_params(params)
        children = {
            key: child.root_params(child_params)
            for (key, child), child_params in zip(self.items(), child_parameters)
        }
        root_params = ParamsNode(params, children)
        return root_params

    def root_prim_keys(self, prim_keys: PrimKeys) -> PrimKeysNode:
        """
        Return a tree of primitive keys for the constraint and all children

        The tree structure should match the tree structure of the constraint.

        For a given constraint, `c`, every key tuple in
        `c.root_prim_keys(prim_keys)` specifies a tuple of primitives for the
        corresponding constraint by indexing from
        `c.root_prim(prim_keys, prims)`.

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the constraint

        Returns
        -------
        root_prim_keys: PrimKeysNode
            A tree of primitive keys for the constraint and all children
        """
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
            for child_prim_keys in self.child_prim_keys
        )

        children = {
            key: child.root_prim_keys(child_prim_keys)
            for (key, child), child_prim_keys
            in zip(self.items(), children_prim_keys)
        }
        return PrimKeysNode(prim_keys, children)

    def root_prim(self, prim_keys: PrimKeys, prims: ResPrims) -> pr.PrimitiveNode:
        """
        Return a root primitive containing primitives for the constraint

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the constraint
        prims: ResPrims
            Primitives for the constraint

        Returns
        -------
        PrimitiveNode
            A root primitive containing primitives for the constraint
        """
        return pr.PrimitiveNode(
            np.array([]), {key: prim for key, prim in zip(prim_keys, prims)}
        )

    @property
    def child_prim_keys(self):
        return self.value[0]

    @property
    def RES_PARAMS_TYPE(self):
        return self.value[1]['RES_PARAMS_TYPE']

    def __call__(
            self,
            prims: ResPrims,
            *params: list[Any]
        ):
        prim_keys = tuple(f'arg{n}' for n, _ in enumerate(prims))
        root_prim = self.root_prim(prim_keys, prims)
        root_prim_keys = self.root_prim_keys(prim_keys)
        root_params = self.root_params(params)
        return self.assem_from_tree(root_prim, root_prim_keys, root_params)

    def assem_from_tree(
            self,
            root_prim: pr.PrimitiveNode,
            root_prim_keys: PrimKeysNode,
            root_params: ParamsNode,
        ):
        flat_constraints = (x for _, x in iter_flat("", self))
        flat_prim_keys = (x.value for _, x in iter_flat("", root_prim_keys))
        flat_params = (x.value for _, x in iter_flat("", root_params))

        residuals = tuple(
            constraint.assem_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in argkeys), *params
            )
            for constraint, argkeys, params in zip(flat_constraints, flat_prim_keys, flat_params)
        )
        return jnp.concatenate(residuals)

    def assem_atleast_1d(
            self, prims: ResPrims, *params: list[Any]
        ) -> NDArray:
        return jnp.atleast_1d(self.assem(prims, *params))

    def assem(
            self, prims: ResPrims, *params: list[Any]
        ) -> NDArray:
        """
        Return a residual vector representing the constraint satisfaction

        Parameters
        ----------
        prims: ResPrims
            A tuple of primitives the constraint applies to
        **kwargs:
            A set of parameters for the residual

            These are things like length, distance, angle, etc.

        Returns
        -------
        NDArray
            The residual representing whether the constraint is satisfied. The
            constraint is satisfied when the residual is 0.
        """
        raise NotImplementedError()


class ConstraintNode(Node[ChildPrimKeys, Constraint]):
    """
    Container tree for constraints
    """
    pass


ChildKeys = tuple[str, ...]
ChildConstraints = tuple[Constraint, ...]

class StaticConstraint(Constraint):
    """
    Constraint with static primitive argument types and child constraints

    To specify a `StaticConstraint`:
    - define `init_aux_data`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the constraint will have no child
    constraints by default.

    If `split_children_params` is undefined, all child constraints will be passed
    empty parameters, and therefore use default values.
    """

    @classmethod
    def init_children(
        cls
    ) -> tuple[ChildPrimKeys, tuple[ChildKeys, ChildConstraints]]:
        return (), ((), ())

    def propogate_child_params(self, params: ResParams) -> ResParams:
        return tuple(() for _ in self)

    @classmethod
    def init_aux_data(
        cls
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def __init__(self):
        child_prim_keys, (child_keys, child_constraints) = self.init_children()
        aux_data = self.init_aux_data()
        super().__init__(child_prim_keys, child_keys, child_constraints, aux_data)


class ParameterizedConstraint(Constraint):
    """
    Constraint with parameterized primitive argument types and child constraints

    To specify a `ParameterizedConstraint`:
    - define `init_aux_data`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the constraint will have no child
    constraints by default.

    If `split_children_params` is undefined, all child constraints will be passed
    empty parameters, and therefore use default values.

    Parameters
    ----------
    **kwargs
        Parameter controlling the constraint definition

        Subclasses should define what these keyword arguments are.
    """

    @classmethod
    def init_children(
        cls, **kwargs
    ) -> tuple[ChildPrimKeys, tuple[ChildKeys, ChildConstraints]]:
        return (), ((), ())

    def propogate_child_params(self, params: ResParams) -> ResParams:
        return tuple(() for _ in self)

    @classmethod
    def init_aux_data(
        cls, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def __init__(self, **kwargs):
        child_prim_keys, (child_keys, child_constraints) = self.init_children(**kwargs)
        aux_data = self.init_aux_data(**kwargs)
        super().__init__(child_prim_keys, child_keys, child_constraints, aux_data)


class ArrayConstraint(ParameterizedConstraint):
    """
    Constraint representing an array of child constraints
    """

    def __init__(self, shape: tuple[int, ...]=(0,)):
        if isinstance(shape, int):
            shape = (shape,)
        super().__init__(shape=shape)


## Point constraints
# NOTE: These are actual constraint classes that can be called so class docstrings
# document there `assem_res` function.

def generate_constraint(
    ConstructionType: type[co.Construction],
    constraint_name: str
):

    class DerivedConstraint(ConstructionType):

        def assem(self, prims, *args):
            *params, value = args
            return ConstructionType.assem(prims, *params) - value

    DerivedConstraint.__name__ = constraint_name

    return DerivedConstraint


# Argument type: tuple[Point,]

Fix = generate_constraint(co.Coordinate, 'Fix')

# Argument type: tuple[Point, Point]

DirectedDistance = generate_constraint(co.DirectedDistance, 'DirectedDistance')

XDistance = generate_constraint(co.XDistance, 'XDistance')

YDistance = generate_constraint(co.YDistance, 'YDistance')

class Coincident(co.StaticConstruction):
    """
    Constrain two points to be coincident

    Parameters
    ----------
    prims: tuple[pr.Point, pr.Point]
        The two points
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Point, pr.Point),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Point, pr.Point]):
        """
        Return the coincident error between two points
        """
        point0, point1 = prims
        return co.Coordinate.assem((point1,)) - co.Coordinate.assem((point0,))


## Line constraints

# Argument type: tuple[Line,]

Length = generate_constraint(co.Length, 'Length')

DirectedLength = generate_constraint(co.DirectedLength, 'DirectedLength')

XLength = generate_constraint(co.XLength, 'XLength')

YLength = generate_constraint(co.YLength, 'YLength')


class Vertical(co.StaticConstruction):
    """
    Constrain a line to be vertical

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(co.LineVector.assem(prims), np.array([1, 0]))


class Horizontal(co.StaticConstruction):
    """
    Constrain a line to be horizontal

    Parameters
    ----------
    prims: tuple[pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        return jnp.dot(co.LineVector.assem(prims), np.array([0, 1]))


# Argument type: tuple[Line, Line]

class RelativeLength(co.StaticConstruction):
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
        return {
            'RES_ARG_TYPES': (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("length",))
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Line, pr.Line], length: float):
        """
        Return the length error of line `prims[0]` relative to line `prims[1]`
        """
        # This sets the length of a line
        line0, line1 = prims
        vec_a = co.LineVector.assem((line0,))
        vec_b = co.LineVector.assem((line1,))
        return jnp.sum(vec_a**2) - length**2 * jnp.sum(vec_b**2)

MidpointXDistance = generate_constraint(co.MidpointXDistance, 'MidpointXDistance')

MidpointYDistance = generate_constraint(co.MidpointYDistance, 'MidpointYDistance')

class Orthogonal(StaticConstraint):
    """
    Constrain two lines to be orthogonal

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Line, pr.Line]):
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
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Line, pr.Line]):
        """
        Return the parallel error between two lines
        """
        line0, line1 = prims
        dir0 = line_vector(line0)
        dir1 = line_vector(line1)
        return jnp.cross(dir0, dir1)


Angle = generate_constraint(co.Angle, 'Angle')


class Collinear(StaticConstraint):
    """
    Constrain two lines to be collinear

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Line, pr.Line]):
        """
        Return the collinearity error between two lines
        """
        res_parallel = Parallel()
        line0, line1 = prims
        line2 = pr.Line(prims=(line1[0], line0[0]))
        # line3 = primitives.Line(children=(line1['Point0'], line0['Point1']))

        return jnp.concatenate(
            [res_parallel((line0, line1)), res_parallel((line0, line2))]
        )


class CoincidentLines(StaticConstraint):
    """
    Constrain two lines to be coincident

    Parameters
    ----------
    prims: tuple[pr.Line, pr.Line]
        The lines
    reverse: bool
        A boolean indicating whether to coincide lines in the same or reverse directions
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("reverse",))
        }

    def assem(self, prims: tuple[pr.Line, pr.Line], reverse: bool):
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

# Argument type: tuple[Line, ...]

class RelativeLengthArray(ArrayConstraint):
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
        child_constraints = (size) * (RelativeLength(),)
        child_prim_keys = tuple((f"arg{n}", f"arg{size}") for n in range(size))
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, parameters):
        return tuple(
            (length,) for length in parameters.lengths
        )

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Line,) + (pr.Line,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("lengths",))
        }

    def assem(self, prims: tuple[pr.Line, ...], lengths: NDArray):
        return np.array([])


class MidpointXDistanceArray(ArrayConstraint):
    """
    Constrain the x-distances between a set of line midpoints

    Parameters
    ----------
    prims: tuple[pr.Line, ...]
        The lines

        The distances are measured from the first to the second line in pairs
    distances: NDArray
        The distances
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        num_child = np.prod(shape)

        child_prim_keys = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        child_keys = tuple(f"LineMidpointXDistance{n}" for n in range(num_child))
        child_constraints = num_child * (MidpointXDistance(),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        return tuple((distance,) for distance in params.distances)

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        num_child = np.prod(shape)
        return {
            'RES_ARG_TYPES': num_child * (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("distances",))
        }

    def assem(self, prims: tuple[pr.Line, ...], distances: NDArray):
        return np.array(())


class MidpointYDistanceArray(ArrayConstraint):
    """
    Constrain the y-distances between a set of line midpoints

    Parameters
    ----------
    prims: tuple[pr.Line, ...]
        The lines

        The distances are measured from the first to the second line in pairs
    distances: NDArray
        The distances
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        num_child = np.prod(shape)

        child_prim_keys = tuple((f"arg{2*n}", f"arg{2*n+1}") for n in range(num_child))
        child_keys = tuple(f"LineMidpointYDistance{n}" for n in range(num_child))
        child_constraints = num_child * (MidpointYDistance(),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        return tuple((distance,) for distance in params.distances)

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        num_child = np.prod(shape)
        return {
            'RES_ARG_TYPES': num_child * (pr.Line, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("distances",))
        }

    def assem(self, prims: tuple[pr.Line, ...], distances: NDArray):
        return np.array(())


class CollinearArray(ArrayConstraint):
    """
    Constrain a set of lines to be collinear

    Parameters
    ----------
    prims: tuple[pr.Line, ...]
        The lines
    """

    @classmethod
    def init_children(cls, shape: tuple[int, ...]):
        size = np.prod(shape)

        child_prim_keys = tuple(("arg0", f"arg{n}") for n in range(1, size))
        child_keys = tuple(f"Collinear[0][{n}]" for n in range(1, size))
        child_constraints = size * (Collinear(),)
        return child_prim_keys, (child_keys, child_constraints)

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Line, ),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Line, ...]):
        return np.array([])

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

PointOnLineDistance = generate_constraint(co.PointOnLineDistance, 'PointOnLineDistance')

PointToLineDistance = generate_constraint(co.PointToLineDistance, 'PointToLineDistance')


class RelativePointOnLineDistance(StaticConstraint):
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
        return {
            'RES_ARG_TYPES': (pr.Point, pr.Line),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("distance", "reverse"))
        }

    def assem(
        self,
        prims: tuple[pr.Point, pr.Line],
        reverse: bool,
        distance: float
    ):
        """
        Return the projected distance error of a point along a line
        """
        point, line = prims
        if reverse:
            origin = line['Point1'].value
            line_vec = -line_vector(line)
        else:
            origin = line['Point0'].value
            line_vec = line_vector(line)
        line_length = jnp.linalg.norm(line_vec)
        unit_vec = line_vec / line_length

        proj_dist = jnp.dot(point.value-origin, unit_vec)
        return jnp.array([proj_dist - distance*line_length])


## Quad constraints

# Argument type: tuple[Quadrilateral]

class Box(StaticConstraint):
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
        child_constraints = (Horizontal(), Horizontal(), Vertical(), Vertical())
        child_prim_keys = (("arg0/Line0",), ("arg0/Line2",), ("arg0/Line3",), ("arg0/Line1",))
        return child_prim_keys, (child_keys, child_constraints)

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Quadrilateral]):
        return np.array(())


class AspectRatio(StaticConstraint):
    """
    Constrain the aspect ratio of a quadrilateral

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral]
        The quad
    ar: float
        The aspect ratio
    """

    @classmethod
    def init_children(cls):
        child_keys = ("RelativeLength",)
        child_constraints = (RelativeLength(), )
        child_prim_keys = (("arg0/Line0", "arg0/Line1"),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, parameters):
        return [(parameters.ar,)]

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("ar",))
        }

    def assem(self, prims: tuple[pr.Quadrilateral], ar: float):
        return np.array(())


# Argument type: tuple[Quadrilateral, Quadrilateral]

class OuterMargin(ParameterizedConstraint):
    """
    Constrain the outer margin between two quadrilaterals

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    margin: float
    """

    @classmethod
    def init_children(cls, side: str="left"):
        child_keys = ("Margin",)
        if side == "left":
            child_constraints = (MidpointXDistance(),)
            child_prim_keys = (("arg1/Line1", "arg0/Line3"),)
        elif side == "right":
            child_constraints = (MidpointXDistance(),)
            child_prim_keys = (("arg0/Line1", "arg1/Line3"),)
        elif side == "bottom":
            child_constraints = (MidpointYDistance(),)
            child_prim_keys = (("arg1/Line2", "arg0/Line0"),)
        elif side == "top":
            child_constraints = (MidpointYDistance(),)
            child_prim_keys = (("arg0/Line2", "arg1/Line0"),)
        else:
            raise ValueError()
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        margin, = params
        return [(margin,)]

    @classmethod
    def init_aux_data(cls, side: str="left"):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("margin",))
        }

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    def assem(self, prims: tuple[pr.Quadrilateral, pr.Quadrilateral], margin: float):
        return np.array(())


class InnerMargin(ParameterizedConstraint):
    """
    Constrain the inner margin between two quadrilaterals

    Parameters
    ----------
    prims: tuple[pr.Quadrilateral, pr.Quadrilateral]
        The quad
    margin: float
    """

    @classmethod
    def init_children(cls, side: str="left"):
        child_keys = ("Margin",)
        if side == "left":
            child_constraints = (MidpointXDistance(),)
            child_prim_keys = (("arg1/Line3", "arg0/Line3"),)
        elif side == "right":
            child_constraints = (MidpointXDistance(),)
            child_prim_keys = (("arg0/Line1", "arg1/Line1"),)
        elif side == "bottom":
            child_constraints = (MidpointYDistance(),)
            child_prim_keys = (("arg1/Line0", "arg0/Line0"),)
        elif side == "top":
            child_constraints = (MidpointYDistance(),)
            child_prim_keys = (("arg0/Line2", "arg1/Line2"),)
        else:
            raise ValueError()
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        margin, = params
        return [(margin,)]

    @classmethod
    def init_aux_data(cls, side: str="left"):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("margin",))
        }

    def __init__(self, side: str="left"):
        super().__init__(side=side)

    def assem(self, prims: tuple[pr.Quadrilateral, pr.Quadrilateral], margin: float):
        return np.array(())


# Argument type: tuple[Quadrilateral, ...]

def idx_1d(multi_idx: tuple[int, ...], shape: tuple[int, ...]):
    """
    Return a 1D array index from a multi-dimensional array index
    """
    strides = shape[1:] + (1,)
    return sum(axis_idx * stride for axis_idx, stride in zip(multi_idx, strides))

class RectilinearGrid(ArrayConstraint):
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
        child_constraints = (
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
        child_prim_keys = align_bottom + align_top + align_left + align_right
        child_keys = (
            [f"CollinearRowBottom{nrow}" for nrow in range(num_row)]
            + [f"CollinearRowTop{nrow}" for nrow in range(num_row)]
            + [f"CollinearColumnLeft{ncol}" for ncol in range(num_col)]
            + [f"CollinearColumnRight{ncol}" for ncol in range(num_col)]
        )
        return child_prim_keys, (child_keys, child_constraints)

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def assem(self, prims: tuple[pr.Quadrilateral, ...]):
        return np.array(())


class Grid(ArrayConstraint):
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
        child_constraints = (
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

        child_prim_keys = (
            rectilineargrid_args,
            colwidth_args,
            rowheight_args,
            tuple(col_margin_line_labels),
            tuple(row_margin_line_labels),
        )

        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        # col_widths, row_heights, col_margins, row_margins = param3s
        return [()] + [(value,) for value in params]

    @classmethod
    def init_aux_data(cls, shape: tuple[int, ...]):
        size = np.prod(shape)
        return {
            'RES_ARG_TYPES': size * (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple(
                "Parameters",
                ("col_widths", "row_heights", "col_margins", "row_margins")
            )
        }

    def assem(
        self,
        prims: tuple[pr.Quadrilateral, ...],
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

# Argument type: tuple[Quadrilateral]

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

class XAxisHeight(StaticConstraint):
    """
    Constrain the x-axis height for an axes

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
        child_constraints = (YDistance(),)
        child_prim_keys = (("arg0/Line1/Point0", "arg0/Line1/Point1"),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, parameters):
        xaxis: XAxis | None = parameters.axis
        if xaxis is None:
            return [(0,)]
        else:
            return [(self.get_xaxis_height(xaxis),)]

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ('axis',))
        }

    def assem(self, prims: tuple[pr.Quadrilateral], axis: XAxis):
        return np.array([])


class YAxisWidth(StaticConstraint):
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
        child_constraints = (XDistance(),)
        child_prim_keys = (("arg0/Line0/Point0", "arg0/Line0/Point1"),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, parameters):
        yaxis: YAxis | None = parameters.axis
        if yaxis is None:
            return [(0,)]
        else:
            return [(self.get_yaxis_width(yaxis),)]

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Quadrilateral,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ('axis',))
        }

    def assem(self, prims: tuple[pr.Quadrilateral], axis: YAxis):
        return np.array([])

# Argument type: tuple[Axes]

# TODO: Handle more specialized x/y axes combos? (i.e. twin x/y axes)
# The below axis constraints are made for single x and y axises

class PositionXAxis(ParameterizedConstraint):
    """
    Constrain the x-axis to the top or bottom of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    @classmethod
    def init_children(cls, bottom: bool, top: bool):

        child_keys = ('CoincidentLines',)
        child_constraints = (CoincidentLines(),)
        if bottom:
            child_prim_keys = (('arg0/Frame/Line0', 'arg0/XAxis/Line2'),)
        elif top:
            child_prim_keys = (('arg0/Frame/Line2', 'arg0/XAxis/Line0'),)
        else:
            raise ValueError(
                "Currently, 'bottom' and 'top' can't both be true"
            )
        return child_prim_keys, (child_keys, child_constraints)

    @classmethod
    def propogate_child_params(cls, params):
        return [(True,)]

    @classmethod
    def init_aux_data(cls, bottom: bool, top: bool):
        return {
            'RES_ARG_TYPES': (pr.Axes,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def __init__(self, bottom: bool=True, top: bool=False):
        return super().__init__(bottom=bottom, top=top)

    def assem(self, prims: tuple[pr.Axes]):
        return np.array([])


class PositionYAxis(ParameterizedConstraint):
    """
    Constrain the y-axis to the left or right of an axes

    Parameters
    ----------
    prims: tuple[pr.Axes]
        The axes
    """

    @classmethod
    def init_children(cls, left: bool=True, right: bool=False):

        child_keys = ('CoincidentLines',)
        child_constraints = (CoincidentLines(),)
        if left:
            child_prim_keys = (('arg0/Frame/Line3', 'arg0/YAxis/Line1'),)
        elif right:
            child_prim_keys = (('arg0/Frame/Line1', 'arg0/YAxis/Line3'),)
        else:
            raise ValueError(
                "Currently, 'left' and 'right' can't both be true"
            )
        return child_prim_keys, (child_keys, child_constraints)

    @classmethod
    def propogate_child_params(cls, params):
        return [(True,)]

    @classmethod
    def init_aux_data(cls, left: bool=True, right: bool=False):
        return {
            'RES_ARG_TYPES': (pr.Axes,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    def __init__(self, left: bool=True, right: bool=False):
        return super().__init__(left=left, right=right)

    def assem(self, prims: tuple[pr.Axes]):
        return np.array([])


class PositionXAxisLabel(StaticConstraint):
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
        child_constraints = (RelativePointOnLineDistance(),)
        child_prim_keys = (('arg0/XAxisLabel', 'arg0/XAxis/Line0'),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        distance, = params
        return [(False, params.distance)]

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Axes,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("distance",))
        }

    def assem(self, prims: tuple[pr.Axes], distance: float):
        return np.array([])


class PositionYAxisLabel(StaticConstraint):
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
        child_constraints = (RelativePointOnLineDistance(),)
        child_prim_keys = (('arg0/YAxisLabel', 'arg0/YAxis/Line1'),)
        return child_prim_keys, (child_keys, child_constraints)

    def propogate_child_params(self, params):
        distance, = params
        return [(False, params.distance)]

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Axes,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("distance",))
        }

    def assem(self, prims: tuple[pr.Axes], distance: float):
        return np.array([])
