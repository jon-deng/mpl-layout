"""
Routines for handling collections of primitives and constraints
"""

import typing as typ
import warnings

import jax
from jax import numpy as jnp
import numpy as np

from . import geometry as geo
from .array import LabelledList

PrimIdx = geo.PrimitiveIndex
PrimIdxs = typ.Tuple[PrimIdx, ...]
PrimList = typ.List[typ.Union[geo.Primitive, geo.PrimitiveArray]]
ConstraintList = typ.List[geo.Constraint]
Graph = typ.List[PrimIdxs]
SolverInfo = typ.Mapping[str, typ.Any]


class Layout:
    """
    Class used to handle a collection of primitives and associated constraints

    The class contains functions to add primitives to the collection, add
    constraints on those primitives, and create the associated graph between
    constraints and primitives.

    Parameters
    ----------
    prims: typ.Optional[PrimList]
        A list of primitives
    constraints: typ.Optional[ConstraintList]
        A list of constraints
    constraint_graph: typ.Optional[Graph]
        A constraint graph
    """

    def __init__(
            self,
            prims: typ.Optional[PrimList]=None,
            constraints: typ.Optional[ConstraintList]=None,
            constraint_graph: typ.Optional[Graph]=None
        ):

        if prims is None:
            prims = []
        if constraints is None:
            constraints = []
        if constraint_graph is None:
            constraint_graph = []

        self._prims = LabelledList(prims)
        self._constraints = LabelledList(constraints)
        self._constraint_graph = constraint_graph

        self._prim_type_count = {}
        self._label_to_primidx = {}

        self._constraint_type_count = {}
        self._label_to_constraintidx = {}

    @property
    def prims(self):
        return self._prims

    @property
    def constraints(self):
        return self._constraints

    @property
    def constraint_graph(self):
        return self._constraint_graph

    def add_prim(
            self,
            prim: geo.Primitive,
            prim_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `geo.Primitive` to the `Layout`

        The primitive will be added with the label 'prim_label'.
        In addition, all child primitives will be recursively added with
        label 'prim_label.child_label'.

        Parameters
        ----------
        prim: geo.Primitive
            The primitive to add
        prim_label: typ.Optional[str]
            An optional label for the primitive

            If not provided, an automatic name based on the primitive class will
            be used.

        Returns
        -------
        prim_label: str
            The label for the added primitive
        """

        # Append the root primitive
        prim_label = self.prims.append(prim, label=prim_label)

        # Append all child primitives
        subprims, subprim_labels, subconstrs, subconstr_graph = \
            expand_prim(prim, label=prim_label)

        for sub_label, sub_prim in zip(subprim_labels, subprims):
            self.prims.append(sub_prim, label=sub_label)

        for constr, prim_idxs in zip(subconstrs, subconstr_graph):
            self.add_constraint(constr, prim_idxs)

        return prim_label

    def add_constraint(
            self,
            constraint: geo.Constraint,
            prim_idxs: PrimIdxs,
            constraint_label: typ.Optional[str]=None
        ) -> str:
        """
        Add a `Constraint` between `Primitive`s

        Parameters
        ----------
        constraint: geo.Constraint
            The constraint to apply
        prim_idxs: PrimIdxs
            Indices of the primitives the constraint applies to
        constraint_label: typ.Optional[str]
            An optional label for the constraint.
            If not provided, an automatic name based on the constraint class
            will be used.

        Returns
        -------
        constraint_label: str
            The label for the added constraint
        """
        # TODO: Refactor this function so that the `constraint_graph`
        # links constraints by `PrimIdx` objects rather than through integers
        # Solving the `Layout` should then do a conversion to integer indices,
        # rather than the conversion happening here
        # These are prims/prim integer indices the constraint applies to
        prims = tuple(self.prims[prim_idx.label] for prim_idx in prim_idxs)
        prim_int_idxs = tuple(
            self.prims.key_to_idx(prim_idx.label)
            for prim_idx in prim_idxs
        )

        # Depack all the sub-index specs for indexed `PrimitiveArray`s
        # Each element of the `prim_sub_idx_specs` is a tuple consisting of:
        # a function that takes a primitive tuple and returns the primitive being indexed
        # and the indices of child primitives that are input to that function to get the primitive
        prim_sub_idx_specs = tuple(
            (None, None) if prim_idx.sub_idx is None
            else prim.index_spec(prim_idx.sub_idx)
            for prim, prim_idx in zip(prims, prim_idxs)
        )
        make_prims = tuple(spec[0] for spec in prim_sub_idx_specs)
        child_idxs = tuple(spec[1] for spec in prim_sub_idx_specs)

        if all(_ is None for _ in make_prims):
            constraint_label = self.constraints.append(constraint, label=constraint_label)
            self.constraint_graph.append(prim_int_idxs)
        else:
            # Modify the primitive indices to account for sub-indexes
            # This is done by the offsetting the root primitive index by any child
            # index offsets from `child_idxs`
            new_prim_int_idxs = []
            for make_prim, child_idx, root_idx in zip(make_prims, child_idxs, prim_int_idxs):
                if make_prim is None:
                    new_prim_int_idxs = new_prim_int_idxs + [root_idx]
                else:
                    new_prim_int_idxs = new_prim_int_idxs + [(root_idx+1)+ii for ii in child_idx]
            new_prim_int_idxs = tuple(new_prim_int_idxs)

            make_prim_arg_lengths = tuple(
                1 if sub_idx is None else len(sub_idx)
                for sub_idx in child_idxs
            )

            # Modify the constraint function to account for sub-indexing
            def new_constraint(new_prims):
                arg_bounds = [0] + np.cumsum(make_prim_arg_lengths).tolist()
                prims = tuple(
                    new_prims[start] if make_prim is None
                    else make_prim(new_prims[start:end])
                    for make_prim, start, end in zip(make_prims, arg_bounds[:-1], arg_bounds[1:])
                )
                return constraint(prims)

            constraint_label = self.constraints.append(new_constraint, label=constraint_label)
            self.constraint_graph.append(new_prim_int_idxs)
        return constraint_label

def expand_prim(
        prim: geo.Primitive,
        label: str
    ) -> typ.Tuple[PrimList, typ.List[str], ConstraintList, Graph]:
    """
    Expand all child primitives of `prim` into a flat list

    This also recursively flattens any implicit constraints and constraint
    graphs.
    The flattening is done so that if a parent primitive has `n` child
    primitives, these are placed immediately after the parent.

    Parameters
    ----------
    prim: geo.Primitive
        The primitive to be expanded
    label: str
        The label for the primitive

    Returns
    -------
    child_prims
        The list of child primitives
    child_labels
        The list of child primitive labels
    child_constraints
        A list of implicit constraints
    child_constraint_graph
        A list of the implicit constraint graph
    """
    # Expand child primitives, constraints, and constraint graph
    child_prims = list(prim.prims)
    child_constraints = list(prim.constraints)

    # To unpack the child constraint graph, we need to prepend all
    # child primitives with the root primitive label, `label`
    child_labels = [
        f'{label}.{child_label}' for child_label in prim.prims.keys()
    ]
    child_constraint_graph = [
        tuple(
            PrimIdx(
                '.'.join([label] + prim_idx.label.split('.')[1:]),
                prim_idx.sub_idx
            )
            for prim_idx in prim_idxs
        )
        for prim_idxs in prim.constraint_graph
    ]

    # Recursively expand any child primitives/constraints
    if len(prim.prims) == 0:
        return child_prims, child_labels, child_constraints, child_constraint_graph
    else:
        for child_prim, child_label in zip(child_prims, child_labels):
            (_re_child_prims,
                _re_child_labels,
                _re_child_constraints,
                _re_child_constraint_graph) = expand_prim(child_prim, child_label)
            child_prims += _re_child_prims
            child_labels += _re_child_labels
            child_constraints += _re_child_constraints
            child_constraint_graph += _re_child_constraint_graph
        return child_prims, child_labels, child_constraints, child_constraint_graph

def contract_prim(
        prim: geo.Primitive,
        prims: PrimList
    ) -> typ.Tuple[geo.Primitive, int]:
    """
    Collapse a flat collection of child primitives into a parent

    This function builds a parent primitive from child primitives.
    This is needed because primitives are parameterized by child primitives and
    are immutable.
    This function should undo the result of `expand_prim`.

    Parameters
    ----------
    prim: geo.Primitive
        The parent primitive
    prims: PrimList
        The child primitives to collapse into the parent

    Returns
    -------
    geo.Primitive
        The parent primitive with updated child primitives
    int
        The total number of child primitives used in the list
    """
    num_child = len(prim.prims)

    child_prims = []
    m = num_child
    for child_prim in prims[:num_child]:
        cprim, n = contract_prim(child_prim, prims[m:])

        child_prims.append(cprim)
        m += n

    return type(prim)(param=prim.param, prims=tuple(child_prims)), m

def build_prims(
        prims: PrimList,
        params: typ.List[np.typing.NDArray]
    ) -> PrimList:
    """
    Create an updated list of `Primitive`s from new parameters

    This function rebuilds a list of primitives with new parameters in
    a corresponding list of parameters.

    Parameters
    ----------
    prims: PrimList
        The old list of primitives
    params: typ.List[np.typing.NDArray]
        The new list of parameters for the primitives

    Returns
    -------
    PrimList
        The new list of primitives
    """

    # First create primitives where the new parameters have been applied
    _new_prims = [
        type(prim)(param=param, prims=prim.prims)
        for prim, param in zip(prims, params)
    ]

    # Contract all child primitives into parents.
    # This is needed because primitive are parameterized both by parameter and
    # other primitives.
    m = 0
    new_prims = []
    while m < len(_new_prims):
        prim, dm = contract_prim(_new_prims[m], _new_prims[m+1:])
        cprims, *_ = expand_prim(prim, '')
        new_prims = new_prims + [prim] + cprims
        m += 1+dm

    return new_prims

def solve(
        prims: typ.List[geo.Primitive],
        constraints: typ.List[geo.Constraint],
        constraint_graph: Graph,
        abs_tol = 1e-10,
        rel_tol = 1e-7,
        max_iter = 10
    ) -> typ.Tuple[PrimList, SolverInfo]:
    """
    Return a set of primitives that satisfy the constraints

    This solves a set of, potentially non-linear constraints, with an
    iterative Newton method.

    Parameters
    ----------
    prims: typ.List[geo.Primitive]
        The list of primitives
    constraints: typ.List[geo.Constraint]
        The list of constraints
    constraint_graph: Graph
        A mapping from each constraint to the primitives it applies to.
        For example, `constraint_graph[0] == (0, 5, 8)` means the first
        constraint applies to primitives `(prims[0], prims[5], prims[8])`.

    Returns
    -------
    PrimList
        The list of primitives satisfying the constraints
    SolverInfo
        Information about the solve.
        Keys are:
            'abs_errs':
                A list of absolute errors for each solver iteration.
                This is the 2-norm of the constraint residual vector.
            'rel_errs':
                A list of relative errors for each solver iteration.
                This is the absolute error at each iteration, relative to the
                initial absolute error.
    """

    nonlinear_solve_info = {}
    abs_errs = []
    rel_errs = []

    n = 0
    abs_err = np.inf
    rel_err = np.inf
    prims_n = prims
    while (abs_err > abs_tol) and (rel_err > rel_tol) and (n < max_iter):
        prims_n, linear_solve_info = solve_linear(prims_n, constraints, constraint_graph)

        n += 1
        abs_err = np.linalg.norm(linear_solve_info['res'])
        abs_errs.append(abs_err)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rel_err = abs_errs[-1]/abs_errs[0]
        rel_errs.append(rel_err)

    nonlinear_solve_info['abs_errs'] = abs_errs
    nonlinear_solve_info['rel_errs'] = rel_errs

    return prims_n, nonlinear_solve_info

def solve_linear(
        prims: typ.List[geo.Primitive],
        constraints: typ.List[geo.Constraint],
        constraint_graph: Graph
    ) -> typ.Tuple[PrimList, SolverInfo]:
    """
    Return a set of primitives that satisfy the (linearized) constraints

    Parameters
    ----------
    prims: typ.List[geo.Primitive]
        The list of primitives
    constraints: typ.List[geo.Constraint]
        The list of constraints
    constraint_graph: Graph
        A mapping from each constraint to the primitives it applies to.
        For example, `constraint_graph[0] == (0, 5, 8)` means the first
        constraint applies to primitives `(prims[0], prims[5], prims[8])`.

    Returns
    -------
    PrimList
        The list of primitives satisfying the (linearized) constraints
    SolverInfo
        Information about the solve.
        Keys are:
            'err': the least squares solver error
            'rank': the rank of the linearized constraint problem
            's': a matrix of singular values for the problem
            'num_dof': the number of degrees of freedom in the problem
            'res': The global constraint residual vector
    """

    # `prim_idx_bounds` stores the right/left indices for each primitive's
    # parameter vector in the global parameter vector array
    # For primitive with index `n`, for example,
    # `prim_idx_bounds[n], prim_idx_bounds[n+1]` are the indices between which
    # the parameter vectors are stored.
    prim_sizes = [prim.param.size for prim in prims]
    prim_idx_bounds = np.cumsum([0] + prim_sizes)

    global_param_n = np.concatenate([prim.param for prim in prims])

    def assem_global_res(global_param):
        new_prim_params = [
            global_param[idx_start:idx_end]
            for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
        ]
        new_prims = build_prims(prims, new_prim_params)
        constraint_vals = []
        for constraint_idx, prim_idxs in enumerate(constraint_graph):
            constraint = constraints[constraint_idx]
            local_prims = tuple(new_prims[idx] for idx in prim_idxs)

            constraint_vals.append(constraint(local_prims))
        return jnp.concatenate(constraint_vals)

    global_res = assem_global_res(global_param_n)
    global_jac = jax.jacfwd(assem_global_res)(global_param_n)

    dglobal_param, err, rank, s = np.linalg.lstsq(global_jac, -global_res, rcond=None)
    global_param_n = global_param_n + dglobal_param
    solver_info = {
        'err': err, 'rank': rank, 's': s, 'num_dof': len(global_param_n),
        'res': global_res
    }

    ## Build a list of primitives from a global parameter vector
    new_prim_params = [
        np.array(global_param_n[idx_start:idx_end])
        for idx_start, idx_end in zip(prim_idx_bounds[:-1], prim_idx_bounds[1:])
    ]
    new_prims = build_prims(prims, new_prim_params)
    new_prims = LabelledList(new_prims, list(prims.keys()))

    return new_prims, solver_info
