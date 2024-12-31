"""
Geometric primitive definitions
"""

from typing import Optional, TypeVar, Any
from numpy.typing import NDArray

import numpy as np
import jax

# from .containers import Node, _make_flatten_unflatten, iter_flat, unflatten, FlatNodeStructure
import mpllayout.containers as cn


## Generic primitive class/interface
# You can create specific primitive definitions by inheriting from these and
# defining appropriate class attributes

TPrim = TypeVar("TPrim", bound="PrimitiveNode")
PrimValue = NDArray[np.float64]
PrimTypes = tuple[type[TPrim] | None, ...]
PrimTypesSignature = PrimTypes | set[PrimTypes]
PrimNodeSignature = tuple[int, PrimTypesSignature]

def validate_prims(
    prims: list[TPrim], prim_types: PrimTypes
) -> tuple[bool, str]:
    """
    Return whether input primitives are valid

    Parameters
    ----------
    prims: list[TPrim]
        Primitives to validate
    prim_types: PrimTypes
        A tuple of primitive types that `prims` must match

    Raises
    -------
    ValueError
    """
    ## Pre-process `prim_types` into a simple tuple of primitive types
    # Expand any `ellipsis` in `prim_types` by treating types before the ...
    # as repeating units
    if len(prim_types) > 0:
       if prim_types[-1] == Ellipsis:
            repeat_prim_types = prim_types[:-1]
            n_repeat = len(prims) // len(repeat_prim_types)
            prim_types = n_repeat * repeat_prim_types

    # Replace any `None` primitive types with `PrimitiveNode`
    prim_types = [
        PrimitiveNode if prim_type is None else prim_type
        for prim_type in prim_types
    ]

    ## Check `prims` has the right length
    if len(prims) != len(prim_types):
        raise ValueError(
            f"Invalid `prims` must be {len(prim_types)} not {len(prims)}"
        )

    ## Check `prims` are the right types
    _prim_types = [type(prim) for prim in prims]
    if not all(
        issubclass(_prim_type, prim_type)
        for _prim_type, prim_type in zip(_prim_types, prim_types)
    ):
        raise ValueError(
            f"Invalid `prims` types must be {prim_types} not {_prim_types}"
        )

class PrimitiveNode(cn.Node[NDArray[np.float64]]):
    """
    Node representation of a geometric primitive

    A geometric primitive (prim for short) is represented by a parameter vector
    and child primitives. For example, in 2D, a point has a 2 element parameter
    vector representing (x, y) coordinates and no child prims. A straight
    line segment has an empty parameter vector with two point child prims
    representing the start and end points.

    Parameters
    ----------
    value: PrimValue
        Parameter vector for the prim
    children: dict[str, TPrim]
        Child prims

    Attributes
    ----------
    signature: PrimNodeSignature
        A tuple specifying valid parameter vector and child types

        A signature has two components
            ``(param_size, prim_types) = signature``,
        where `param_size` is the size of the parameter vector and `prim_types`
        indicates valid child primitives.
    """

    # NOTE: `None` indicates `PrimitiveNode` but the name isn't available within
    # the class itself
    signature: PrimNodeSignature = (0, (None, ...))

    def __init__(self, value: PrimValue, children: dict[str, TPrim]):

        # Type checks that value is an array and children are prims
        assert isinstance(value, np.ndarray)
        assert all(
            isinstance(cprim, PrimitiveNode) for _, cprim in children.items()
        )
        param_size, prim_type_sig = self.signature

        assert len(value) == param_size

        prims = [cprim for _, cprim in children.items()]
        if isinstance(prim_type_sig, tuple):
            prim_types = prim_type_sig
            try:
                validate_prims(prims, prim_types)
            except ValueError as err:
                raise err
        elif isinstance(prim_type_sig, set):
            def match(prims, prim_types):
                try:
                    validate_prims(prims, prim_types)
                except ValueError as err:
                    return False
                else:
                    return True

            if not any(
                match(prims, prim_types) for prim_types in prim_type_sig
            ):
                raise ValueError(
                    f"No matching signatures for `prims` in {prim_type_sig}"
                )

        super().__init__(value, children)


class Primitive(PrimitiveNode):
    """
    Primitive parameterized by a value, primitives and keyword arguments

    To define a primitive, subclass `Primitive` and define the class methods
    `init_children`, `default_value` and `default_prims`.

    Parameters
    ----------
    value: Optional[PrimValue]
        Parameter vector for the primitive
    prims: Optional[list[TPrim]]
        Parameterizing primitives

        Note that these are often the same as child primitives but this isn't
        always
    **kwargs: dict[str, Any]
        Additional keyword arguments

    Attributes
    ----------
    see `PrimitiveNode`
    """

    def __init__(
        self,
        value: Optional[NDArray] = None,
        prims: Optional[list[TPrim]] = None,
        **kwargs: dict[str, Any]
    ):
        if value is None:
            value = self.default_value(**kwargs)
        elif isinstance(value, (list, tuple)):
            value = np.array(value)
        elif isinstance(value, (np.ndarray, jax.numpy.ndarray)):
            value = value
        else:
            raise TypeError()

        if prims is None:
            prims = self.default_prims(**kwargs)

        child_keys, child_prims = self.init_children(prims, **kwargs)

        super().__init__(
            value, {key: prim for key, prim in zip(child_keys, child_prims)}
        )

    @classmethod
    def init_children(
        cls, prims: list[TPrim], **kwargs: dict[str, Any]
    ) -> tuple[list[str], list[TPrim]]:
        """
        Return child primitives from parameterizing primitives

        Parameters
        ----------
        prims: list[TPrim]
            Parameterizing primitives

        Returns
        -------
        List[str]
            Child primitive keys
        List[TPrim]
            Child primitives
        """
        raise NotImplementedError()

    @classmethod
    def default_value(cls, **kwargs: dict[str, Any]) -> NDArray:
        """
        Return a default parameter vector
        """
        raise NotImplementedError()

    @classmethod
    def default_prims(cls, **kwargs: dict[str, Any]) -> list[TPrim]:
        """
        Return default parameterizing primitives
        """
        raise NotImplementedError()


class StaticPrimitive(Primitive):
    """
    A `Primitive` with no additional keyword arguments

    Static primitives have a static parameter vector shape and child primitive
    types.

    Parameters
    ----------
    See `Primitive`
    """

    def __init__(
        self,
        value: Optional[NDArray] = None,
        prims: Optional[list[TPrim]] = None,
    ):
        super().__init__(value=value, prims=prims)

    @classmethod
    def init_children(
        cls, prims: list[TPrim]
    ) -> tuple[list[str], list[TPrim]]:
        """
        See `Primitive`
        """
        raise NotImplementedError()

    @classmethod
    def default_value(cls) -> NDArray:
        """
        See `Primitive`
        """
        raise NotImplementedError()

    @classmethod
    def default_prims(cls) -> list[TPrim]:
        """
        See `Primitive`
        """
        raise NotImplementedError()

## Primitive definitions

class Point(StaticPrimitive):
    """
    A point

    Child primitives are:
    - no child primitives

    Parameters
    ----------
    value: NDArray (2,), optional
        The point coordinates
    prims: tuple[]
        An empty set of primitives
    """

    signature = (2, ())

    @classmethod
    def init_children(cls, prims):
        return (), ()

    @classmethod
    def default_value(cls):
        return np.array([0, 0])

    @classmethod
    def default_prims(cls):
        return ()


class Line(StaticPrimitive):
    """
    A straight line segment between two points

    Child primitives are:
    - `line['Point0']` : start point
    - `line['Point1']` : end point

    Parameters
    ----------
    value: NDArray (), optional
        An empty array
    prims: Tuple[Point, Point]
        The start and end point
    """

    signature = (0, (Point, Point))

    @classmethod
    def init_children(cls, prims: tuple[Point, Point]):
        return ("Point0", "Point1"), prims

    @classmethod
    def default_value(cls):
        return np.array(())

    @classmethod
    def default_prims(cls):
        return (Point([0, 0]), Point([0, 1]))


class Polygon(Primitive):
    """
    A polygon through a given set of points

    Child primitives are:
    - `Polygon[f'Line{n}']` : the n'th line in the polygon

    The end point of a line joins the start point of the next line to form a
    closed loop.

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: List[Point]
        A list of vertices the polygon passes through

        The final point in `prims` will automatically be connected to the first
        point in `prims`.
        The length of `prims` should match `size`.
    size: int
        The number of points in the polygon
    """

    signature = (0, (Line, ...))

    def __init__(
        self,
        value: Optional[NDArray] = None,
        prims: Optional[list[TPrim]] = None,
        size: int = 3
    ):
        return super().__init__(value=value, prims=prims, size=size)

    @classmethod
    def init_children(
        cls, prims: list[Point], size: int=3
    ):
        points = prims
        child_prims = [
            Line(np.array([]), (pointa, pointb))
            for pointa, pointb in zip(points[:], points[1:] + points[:1])
        ]
        child_keys = [f"Line{n}" for n, _ in enumerate(child_prims)]
        return child_keys, child_prims

    @classmethod
    def default_value(cls, size: int=3):
        return np.array(())

    @classmethod
    def default_prims(cls, size: int=3):
        # Generate points around circle
        ii = np.arange(size)
        xs = np.cos(2*np.pi/size * ii)
        ys = np.sin(2*np.pi/size * ii)
        return [Point((x, y)) for x, y in zip(xs, ys)]


class Quadrilateral(Polygon):
    """
    A quadrilateral (4 sided polygon)

    Child primitives are:
    - `quad['Line0']` : the first line in the quadrilateral
    - ...
    - `quad['Line3']` : the last line in the quadrilateral

    For modelling rectangles in matplotlib (axes, bbox, etc.) the lines are
    treated as the bottom, right, top, and left of a box in a clockwise fasion.
    Specifically, the lines correspond to:
    - 'Line0' : bottom
    - 'Line1' : right
    - 'Line2' : top
    - 'Line3' : left

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: List[Point]
        A list of 4 vertices the quadrilateral passes through
    """

    signature = (0, (Line, Line, Line, Line))

    def __init__(
        self,
        value: Optional[NDArray] = None,
        children: Optional[list[Point]] = None,
    ):
        super().__init__(value, children, size=4)

    @classmethod
    def default_value(cls, size: int=4):
        return np.array(())

    @classmethod
    def default_prims(cls, size: int=4):
        # Generate a unit square
        xs = [0, 1, 1, 0]
        ys = [0, 0, 1, 1]
        return [Point((x, y)) for x, y in zip(xs, ys)]

AxesChildPrims = (
    tuple[Quadrilateral]
    | tuple[Quadrilateral, Quadrilateral, Point]
    | tuple[Quadrilateral, Quadrilateral, Point, Quadrilateral, Point]
)

class Axes(Primitive):
    """
    A collection of quadrilaterals and points representing an axes

    Child primitives are:
    - `quad['Frame']` : A `Quadrilateral` representing the plotting area
    - `quad['XAxis']` : A `Quadrilateral` representing the x-axis
    - `quad['XAxisLabel']` : A `Point` representing the x-axis label anchor
    - `quad['YAxis']` : A `Quadrilateral` representing the y-axis
    - `quad['YAxisLabel']` : A `Point` representing the y-axis label anchor

    Parameters
    ----------
    value: NDArray ()
        An empty array
    prims: tuple[Quadrilateral, Quadrilateral, Point, Quadrilateral, Point]
        A tuple of quadrilateral and points

        The the number of quadrilaterals and points to supply depends on
        where an x/y axis is included.
    xaxis, yaxis: bool
        Whether to include an x/y axis and corresponding label

        If false for a given axis, the corresponding child primitives will not
        be present.
    """

    signature = (
        0,
        {
            (Quadrilateral,),
            (Quadrilateral, Quadrilateral, Point),
            (Quadrilateral, Quadrilateral, Point, Quadrilateral, Point)
        }
    )

    def __init__(
            self,
            value: Optional[NDArray]=None,
            prims: Optional[AxesChildPrims]=None,
            xaxis: bool=False,
            yaxis: bool=False
        ):
        super().__init__(value, prims, xaxis=xaxis, yaxis=yaxis)

    @classmethod
    def init_children(
        cls,
        prims: AxesChildPrims,
        xaxis: bool=False,
        yaxis: bool=False
    ) -> tuple[list[str], AxesChildPrims]:

        if xaxis:
            xaxis_keys = ("XAxis", "XAxisLabel")
        else:
            xaxis_keys = ()

        if yaxis:
            yaxis_keys = ("YAxis", "YAxisLabel")
        else:
            yaxis_keys = ()
        return ("Frame",) + xaxis_keys + yaxis_keys, prims

    @classmethod
    def default_value(cls, xaxis: bool=False, yaxis: bool=False):
        return np.array(())

    @classmethod
    def default_prims(cls, xaxis: bool=False, yaxis: bool=False):
        if xaxis:
            xaxis_prims = (Quadrilateral(), Point())
        else:
            xaxis_prims = ()

        if yaxis:
            yaxis_prims = (Quadrilateral(), Point())
        else:
            yaxis_prims = ()
        return (Quadrilateral(),) + xaxis_prims + yaxis_prims


## Register `Primitive` classes as `jax.pytree`
_PrimitiveClasses = [
    Primitive,
    PrimitiveNode,
    Point,
    Line,
    Polygon,
    Quadrilateral,
    Axes,
]
for _PrimitiveClass in _PrimitiveClasses:
    _flatten_primitive, _unflatten_primitive = cn._make_flatten_unflatten(_PrimitiveClass)
    jax.tree_util.register_pytree_node(
        _PrimitiveClass, _flatten_primitive, _unflatten_primitive
    )


## Primitive value vector methods
# These are used to get the primitive parameter vector from a primitive
# and to update primitives with new parameters

def filter_unique_values_from_prim(
    root_prim: Primitive,
) -> tuple[dict[str, int], list[Primitive]]:
    """
    Return unique primitives from a root primitive and indicate their indices

    Note that primitives in a primitive node are not necessarily unique
    ; for example `Point`s are shared between lines in a polygon.

    When solving a set of geometric constraints, the geometric constraint
    residual should be linked to a function of unique primitives only.

    Returns
    -------
    prim_to_idx: dict[str, int]
        A mapping from each primitive key to its unique primitive index
    prims: list[Primitive]
        A list of unique primitives
    """
    value_id_to_idx = {}
    values = []
    prim_to_idx = {}

    for key, prim in cn.iter_flat("", root_prim):
        value_id = id(prim.value)

        if value_id not in value_id_to_idx:
            values.append(prim.value)
            value_idx = len(values) - 1
            value_id_to_idx[value_id] = value_idx
        else:
            value_idx = value_id_to_idx[value_id]

        prim_to_idx[key] = value_idx

    return prim_to_idx, values

def build_prim_from_unique_values(
    flat_prim: list[cn.FlatNodeStructure], prim_to_idx: dict[str, int], values: list[NDArray]
) -> Primitive:
    """
    Return a new primitive with values updated from unique values

    Parameters
    ----------
    flat_prim: list[FlatNodeStructure]
        The flat primitive tree (see `flatten`)
    prim_to_idx: dict[str, int]
        A mapping from each primitive key to a unique primitive value in `values`
    values: list[NDArray]
        A list of primitive values for unique primitives in `root_prim`

    Returns
    -------
    Primitive
        The new primitive with updated values
    """
    prim_keys = (flat_struct[0] for flat_struct in flat_prim)
    new_prim_values = (values[prim_to_idx[key]] for key in prim_keys)

    new_prim_structs = [
        (prim_key, PrimType, new_value, child_keys)
        for (prim_key, PrimType, _old_value, child_keys), new_value
        in zip(flat_prim, new_prim_values)
    ]
    return cn.unflatten(new_prim_structs)[0]
