"""
Geometric primitive and constraints
"""

import typing as typ
from numpy.typing import NDArray

import numpy as np
import jax.numpy as jnp

from .array import LabelledTuple

PrimList = typ.Tuple['Primitive', ...]
ConstraintList = typ.List['Constraint']
Idxs = typ.Tuple[int]

ArrayShape = typ.Tuple[int, ...]
PrimTuple = typ.Tuple['Primitive', ...]

from .primitives import *
from .constraints import *
