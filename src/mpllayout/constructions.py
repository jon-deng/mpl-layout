"""
Geometric constructions

Constructions are functions that accept primitives and return a vector.
For example, this could be the coordinates of a point, the angle between two
lines, or the length of a single line.
"""

from typing import Optional, Any
from numpy.typing import NDArray

from collections import namedtuple
import itertools

import numpy as np
import jax.numpy as jnp

from . import primitives as pr
from .containers import Node, iter_flat

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
    A tree of primitive keys indicating primitives for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding primitive keys node.
    """
    pass


class ParamsNode(Node[ResParams, "ParamsNode"]):
    """
    A tree of residual parameters (kwargs) for a construction

    The tree structure should match the tree structure of a construction such that
    for each construction node, there is a corresponding parameters node.
    """
    pass

# TODO: Add construction class that accepts a unit
# This would handle the case of setting a length relative to another one

class Construction(Node[ChildPrimKeys, "Construction"]):
    """
    The base geometric construction class

    A geometric construction is a function returning a vector from geometric
    primitives.
    The condition is implemented through a function
        `assem(self, prims, **kwargs)`
    where
    - `prims` are the geometric primitives to construction
    - and `**kwargs` are additional arguments for the residual.
    The construction is satisified when `assem(self, prims, **kwargs) == 0`.
    To implement `assem`, `jax` functions should be used to return a
    residual vector from the parameter vectors of input primitives.

    Constraints have a tree-like structure.
    A construction can contain child constructions by passing subsets of its input
    primitives (`prims` in `assem`) on to child constructions.
    The residual of a construction is the result of concatenating all child
    construction residuals.

    To create a construction, subclass `Construction` then:
        1. Define the residual for the construction (`assem`)
        2. Specify the parameters for `Construction.__init__` (see `Parameters`)\
        3. Define `Construction.split_children_params` (see below)
    Note that some of the `Construction.__init__` parameters are for type checking
    inputs to `assem` while the others are for specifying child constructions.
    `StaticConstraint` and `ParameterizedConstraint` are two `Construction`
    subclasses that can be subclassed to create constructions.

    Parameters
    ----------
    child_prim_keys: tuple[PrimKeys, ...]
        Primitive key tuples for each child construction

        This is stored as the "value" of the tree structure and encodes how to
        create primitives for each child construction from the parent construction's
        `prims` (in `assem(self, prims, **kwargs)`).
        For each child construction, a tuple of primitive keys indicates a
        subset of parent primitives to form child construction primitive
        arguments.

        To illustrate this, consider a parent construction with residual

        ```python
        Parent.assem(self, prims, **kwargs)
        ```

        and the n'th child construction with primitive key tuple

        ```python
        child_prim_keys[n] == ('arg0', 'arg3/Line2')
        ```.

        This indicates the n'th child construction should be evaluated with

        ```
        child_prims = (prims[0], prims[3]['Line2'])
        ```
    child_keys: list[str]
        Keys for any child constructions
    child_constructions: list[Construction]
        Child constructions
    aux_data: Mapping[str, Any]
        Any auxiliary data

        This is usually for type checking/validation of inputs
    """

    # TODO: Implement `assem` type checking using `aux_data`
    def __init__(
        self,
        child_prim_keys: ChildPrimKeys,
        child_keys: list[str],
        child_constructions: list["Construction"],
        aux_data: Optional[dict[str, Any]] = None
    ):
        children = {
            key: child for key, child in zip(child_keys, child_constructions)
        }
        super().__init__((child_prim_keys, aux_data), children)

    # TODO: Make this something that's passed through __init__?
    # That would make it harder to forget defining this?
    def split_children_params(self, params: ResParams) -> tuple[ResParams, ...]:
        """
        Return children construction parameters from parent construction parameters
        """
        raise NotImplementedError()

    def root_params(self, params: ResParams) -> ParamsNode:
        """
        Return a tree of residual kwargs for the construction and all children

        The tree structure should match the tree structure of the construction.

        Parameters
        ----------
        params: ResParams
            Residual keyword arguments for the construction

        Returns
        -------
        root_params: ParamsNode
            A tree of keyword arguments for the construction and all children
        """
        params = load_named_tuple(self.RES_PARAMS_TYPE, params)

        child_parameters = self.split_children_params(params)
        children = {
            key: child.root_params(child_params)
            for (key, child), child_params in zip(self.items(), child_parameters)
        }
        root_params = ParamsNode(params, children)
        return root_params

    def root_prim_keys(self, prim_keys: PrimKeys) -> PrimKeysNode:
        """
        Return a tree of primitive keys for the construction and all children

        The tree structure should match the tree structure of the construction.

        For a given construction, `c`, every key tuple in
        `c.root_prim_keys(prim_keys)` specifies a tuple of primitives for the
        corresponding construction by indexing from
        `c.root_prim(prim_keys, prims)`.

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the construction

        Returns
        -------
        root_prim_keys: PrimKeysNode
            A tree of primitive keys for the construction and all children
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
        Return a root primitive containing primitives for the construction

        Parameters
        ----------
        prim_keys: PrimKeys
            Primitive keys for the construction
        prims: ResPrims
            Primitives for the construction

        Returns
        -------
        PrimitiveNode
            A root primitive containing primitives for the construction
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
            params: tuple[Any, ...] | dict[str, Any]
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
        flat_constructions = (x for _, x in iter_flat("", self))
        flat_prim_keys = (x.value for _, x in iter_flat("", root_prim_keys))
        flat_params = (x.value for _, x in iter_flat("", root_params))

        residuals = tuple(
            construction.assem_atleast_1d(
                tuple(root_prim[arg_key] for arg_key in argkeys), params
            )
            for construction, argkeys, params in zip(flat_constructions, flat_prim_keys, flat_params)
        )
        return jnp.concatenate(residuals)

    def assem_atleast_1d(
            self, prims: ResPrims, params: ResParams
        ) -> NDArray:
        return jnp.atleast_1d(self.assem(prims, **params._asdict()))

    def assem(
            self, prims: ResPrims, **kwargs
        ) -> NDArray:
        """
        Return a residual vector representing the construction satisfaction

        Parameters
        ----------
        prims: ResPrims
            A tuple of primitives the construction applies to
        **kwargs:
            A set of parameters for the residual

            These are things like length, distance, angle, etc.

        Returns
        -------
        NDArray
            The residual representing whether the construction is satisfied. The
            construction is satisfied when the residual is 0.
        """
        raise NotImplementedError()


class ConstructionNode(Node[ChildPrimKeys, Construction]):
    """
    Container tree for constructions
    """
    pass


ChildKeys = tuple[str, ...]
ChildConstraints = tuple[Construction, ...]

class StaticConstruction(Construction):
    """
    Construction with static primitive argument types and child constructions

    To specify a `StaticConstraint`:
    - define `init_aux_data`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the construction will have no child
    constructions by default.

    If `split_children_params` is undefined, all child constructions will be passed
    empty parameters, and therefore use default values.
    """

    @classmethod
    def init_children(
        cls
    ) -> tuple[ChildPrimKeys, tuple[ChildKeys, ChildConstraints]]:
        return (), ((), ())

    def split_children_params(self, params: ResParams) -> ResParams:
        return tuple({} for _ in self)

    @classmethod
    def init_aux_data(
        cls
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def __init__(self):
        child_prim_keys, (child_keys, child_constructions) = self.init_children()
        aux_data = self.init_aux_data()
        super().__init__(child_prim_keys, child_keys, child_constructions, aux_data)


class ParameterizedConstruction(Construction):
    """
    Construction with parameterized primitive argument types and child constructions

    To specify a `ParameterizedConstraint`:
    - define `init_aux_data`,
    - and optionally define, `init_children` and `split_children_params`.

    If `init_children` is undefined the construction will have no child
    constructions by default.

    If `split_children_params` is undefined, all child constructions will be passed
    empty parameters, and therefore use default values.

    Parameters
    ----------
    **kwargs
        Parameter controlling the construction definition

        Subclasses should define what these keyword arguments are.
    """

    @classmethod
    def init_children(
        cls, **kwargs
    ) -> tuple[ChildPrimKeys, tuple[ChildKeys, ChildConstraints]]:
        return (), ((), ())

    def split_children_params(self, params: ResParams) -> ResParams:
        return tuple({} for _ in self)

    @classmethod
    def init_aux_data(
        cls, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def __init__(self, **kwargs):
        child_prim_keys, (child_keys, child_constructions) = self.init_children(**kwargs)
        aux_data = self.init_aux_data(**kwargs)
        super().__init__(child_prim_keys, child_keys, child_constructions, aux_data)

## Line constructions

# Argument type: tuple[Line,]

class LineVector(StaticConstruction):
    """
    Return the line vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        """
        Return the length error of a line
        """
        line, = prims
        pointa, pointb = line.values()
        return pointb.value - pointa.value


class Length(StaticConstruction):
    """
    Return the length of a line

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_PARAMS_TYPE': (),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ())
        }

    @classmethod
    def assem(cls, prims: tuple[pr.Line]):
        (line,) = prims
        return jnp.sum(LineVector.assem((line,))**2)**(1/2)


class DirectedLength(StaticConstruction):
    """
    Return the length of a line along a vector

    Parameters
    ----------
    prims: tuple[pr.Line]
        The line
    direction: NDArray
        The direction
    """

    @classmethod
    def init_aux_data(cls):
        return {
            'RES_ARG_TYPES': (pr.Line,),
            'RES_PARAMS_TYPE': namedtuple("Parameters", ("direction"))
        }

    @classmethod
    def assem(
        cls,
        prims: tuple[pr.Line],
        direction: NDArray=np.array([1, 0])
    ):
        (line,) = prims
        return jnp.dot(LineVector.assem((line,)), direction)
