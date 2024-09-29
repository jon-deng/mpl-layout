"""
Container/array objects with both integer and string indices

This modules contains container objects with both string and integer indices.
These act basically as `dictionary` objects but can also be indexed with
integers.

The dual string/integer indices are used to facilitate creation of constraints
and primitives. String labels for primitives are easy to keep track of and apply
constraints to while integer indices are needed to solve systems of equations
and build matrices.
"""

import typing as tp

import itertools

import jax

T = tp.TypeVar("T")


class Node(tp.Generic[T]):
    """
    Tree structure with labelled child nodes

    Parameters
    ----------
    value: T
        A value associated with the node
    children: tp.Tuple[Node, ...]
        Child nodes
    labels: tp.Tuple[str, ...]
        Child node labels
    """

    def __init__(
        self, value: tp.Union[None, T], children: tp.List["Node"], keys: tp.List[str]
    ):
        self._value = value
        self._children = children
        self._keys = keys

        if len(children) == len(keys):
            self._key_to_child = {label: cnode for label, cnode in zip(keys, children)}
        else:
            raise ValueError(
                f"Number of child nodes {len(children)} must equal number of keys {len(keys)}"
            )

    @property
    def children(self):
        """
        Return any children
        """
        return self._children

    @property
    def children_map(self):
        """
        Return any children
        """
        return self._key_to_child

    @property
    def value(self) -> T:
        """
        Return the value
        """
        return self._value

    ## Flattened interface

    ## String

    def __repr__(self) -> str:
        keys_repr = ", ".join(self.keys())
        children_repr = ", ".join([node.__repr__() for node in self.children])
        return f"{type(self).__name__}({self.value}, ({children_repr}), ({keys_repr}))"

    def __str__(self) -> str:
        return self.__repr__()

    ## Dict-like interface

    def __contains__(self, key: str):
        split_keys = key.split("/")
        parent_key = split_keys[0]
        child_key = "/".join(split_keys[1:])

        if child_key == "":
            return parent_key in self.children_map
        else:
            return child_key in self[parent_key]

    def __len__(self) -> int:
        return len(self.children)

    def keys(self) -> tp.List[str]:
        """
        Return child keys

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys

            Child keys are separated using '/'
        """
        return list(self.children_map.keys())

    def values(self, flat: bool = False) -> tp.List[T]:
        """
        Return child primitives

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten child primitives
        """
        return list(self.children_map.values())

    def items(self, flat: bool = False) -> tp.List[tp.Tuple[str, "Node[T]"]]:
        """
        Return paired child keys and associated trees

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys and trees
        """
        return self.children_map.items()

    def __getitem__(self, key: tp.Union[str, int]) -> "Node[T]":
        """
        Return the value indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        return self.get_child(key)

    def get_child(self, key: tp.Union[str, int]) -> "Node[T]":
        if isinstance(key, int):
            return self.get_child_from_int(key)
        elif isinstance(key, str):
            return self.get_child_from_str(key)
        else:
            raise TypeError("")

    def get_child_from_int(self, key: int) -> "Node[T]":
        return self.children[key]

    def get_child_from_str(self, key: str) -> "Node[T]":
        split_key = key.split("/")
        parent_key = split_key[0]
        child_key = "/".join(split_key[1:])

        try:
            if len(split_key) == 1:
                return self.children_map[parent_key]
            else:
                return self.children_map[parent_key].get_child_from_str(child_key)
        except KeyError as err:
            raise KeyError(f"{key}") from err

    def add_child(self, key: str, child: "Node[T]"):
        """
        Add a primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split("/")
        parent_key = split_key[0]
        child_keys = split_key[1:]
        child_key = "/".join(child_keys)

        try:
            if len(child_keys) > 0:
                self.children_map[parent_key][child_key] = child
            elif len(child_keys) == 0:
                self._children.append(child)
                self.children_map[key] = child
            else:
                assert False

        except KeyError as err:
            raise KeyError(f"{key}") from err


## Manual flattening/unflattening implementation
NodeType = tp.Type[Node]
FlatNodeStructure = tp.Tuple[NodeType, str, T, int]


def iter_flat(key: str, node: Node) -> tp.Iterable[tp.Tuple[str, Node]]:
    """
    Return a flat iterator over all nodes
    """
    num_child = len(node)

    if num_child == 0:
        nodes = [(key, node)]
    else:
        cnodes = [
            iter_flat("/".join((key, ckey)), cnode)
            for ckey, cnode in zip(node.keys(), node.values())
        ]
        cnodes = itertools.chain(cnodes)

        nodes = itertools.chain([(key, node)], *cnodes)
    return nodes


def flatten(key: str, node: Node) -> tp.List[FlatNodeStructure]:
    node_structs = [
        (type(_node), _key, _node.value, len(_node))
        for _key, _node in iter_flat(key, node)
    ]
    return node_structs


def unflatten(
    node_structs: tp.List[FlatNodeStructure],
) -> tp.Tuple[Node, tp.List[FlatNodeStructure]]:
    NodeType, pkey, value, num_child = node_structs[0]
    node_structs = node_structs[1:]

    if num_child == 0:
        node = NodeType(value, (), ())
    else:
        ckeys = []
        children = []
        for _ in range(num_child):
            child_struct = node_structs[0]

            ckey = child_struct[1][len(pkey) + 1 :]
            child, node_structs = unflatten(node_structs)

            ckeys.append(ckey)
            children.append(child)

        node = NodeType(value, children, ckeys)

    return node, node_structs


## pytree flattening/unflattening implementation
# These functions register `Node` classes as a `jax.pytree` so jax can flatten/unflatten
# them

Children = tp.List[Node[T]]
FlatNode = tp.Tuple[T, Children]
Keys = tp.List[str]
AuxData = tp.Tuple[Keys]


def _make_flatten_unflatten(NodeClass: tp.Type[Node[T]]):

    def _flatten_node(node: NodeClass) -> tp.Tuple[FlatNode, AuxData]:
        flat_node = (node.value, node.children)
        aux_data = (node.keys(),)
        return (flat_node, aux_data)

    def _unflatten_node(aux_data: AuxData, flat_node: FlatNode) -> NodeClass:
        value, children = flat_node
        (keys,) = aux_data
        return NodeClass(value, children, keys)

    return _flatten_node, _unflatten_node


## Register `Node` as `jax.pytree`
_NodeClasses = [Node]
for _NodeClass in _NodeClasses:
    _flatten, _unflatten = _make_flatten_unflatten(_NodeClass)
    jax.tree_util.register_pytree_node(_NodeClass, _flatten, _unflatten)
