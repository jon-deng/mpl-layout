"""
Tree class (`Node`) definition and related utilities for trees

This module defines a tree class `Node` and related utilities.
This class is used by itself as well as to define geometric primitives and constraints.
"""

from typing import TypeVar, Generic, Any
from collections.abc import Callable

import itertools

import jax

TValue = TypeVar("TValue")
TChild = TypeVar("TChild", bound="Node")
TNode = TypeVar("TNode", bound="Node")

class Node(Generic[TValue, TChild]):
    """
    Tree structure with labelled child nodes

    Parameters
    ----------
    value: TValue
        A value associated with the node
    children: dict[str, TChild]
        A dictionary of child nodes

    Attributes
    ----------
    value: TValue
        The value stored in the node
    children: dict[str, TChild]
        A dictionary of child nodes
    """

    def __init__(self, value: TValue, children: dict[str, TChild]):
        assert isinstance(children, dict)
        self._value = value
        self._children = children

    @classmethod
    def from_tree(cls, value: TValue, children: dict[str, TChild]):
        node = super().__new__(cls)
        Node.__init__(node, value, children)
        return node

    @property
    def value(self):
        """
        Return the node value
        """
        return self._value

    @property
    def children(self):
        """
        Return all child nodes
        """
        return self._children

    ## Tree methods

    def node_height(self) -> int:
        if len(self) == 0:
            return 0
        else:
            return 1 + max(child.node_height() for _, child in self.items())

    ## Flattened interface

    ## String

    def __repr__(self) -> str:
        keys_repr = ", ".join(self.keys())
        children_repr = ", ".join([node.__repr__() for _, node in self.children.items()])
        return f"{type(self).__name__}({self.value}, ({children_repr}), ({keys_repr}))"

    def __str__(self) -> str:
        return self.__repr__()

    ## Dict-like interface

    def __contains__(self, key: str) -> bool:
        split_keys = key.split("/")
        parent_key = split_keys[0]
        child_key = "/".join(split_keys[1:])

        if child_key == "":
            return parent_key in self.children
        else:
            return child_key in self[parent_key]

    def __len__(self) -> int:
        return len(self.children)

    def keys(self):
        """
        Return child keys

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys

            Child keys are separated using '/'
        """
        return list(self.children.keys())

    def values(self, flat: bool = False):
        """
        Return child primitives

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten child primitives
        """
        return list(self.children.values())

    def items(self, flat: bool = False):
        """
        Return paired child keys and associated trees

        Parameters
        ----------
        flat:
            Toggle whether to recursively flatten keys and trees
        """
        return self.children.items()

    def __setitem__(self, key: str, node: TChild):
        """
        Set the node indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_keys = key.split("/")
        parent_key = "/".join(split_keys[:-1])
        child_key = split_keys[-1]
        if parent_key == "":
            self.children[child_key] = node
        else:
            self[parent_key].children[child_key] = node

    def __getitem__(self, key: str | int | slice):
        """
        Return the value indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        if isinstance(key, str):
            return self.get_child_from_str(key)
        elif isinstance(key, (int, slice)):
            return self.get_child_from_int_or_slice(key)
        else:
            raise TypeError("")

    def get_child_from_int_or_slice(self, key: int | slice):
        return list(self.children.values())[key]

    def get_child_from_str(self, key: str):
        split_key = key.split("/", 1)
        parent_key, child_keys = split_key[0], split_key[1:]

        try:
            if len(child_keys) == 0:
                return self.get_child_from_str_nonrecursive(parent_key)
            else:
                return self.children[parent_key].get_child_from_str(child_keys[0])
        except KeyError as err:
            raise KeyError(f"{key}") from err

    def get_child_from_str_nonrecursive(self, key: str):
        return self.children[key]

    def add_child(self, key: str, child: TChild):
        """
        Add a primitive indexed by a slash-separated key

        Parameters
        ----------
        key: str
            A slash-separated key, for example 'Box/Line0/Point2'
        """
        split_key = key.split("/", 1)
        parent_key, child_keys = split_key[0], split_key[1:]

        try:
            if len(child_keys) == 0:
                self.add_child_nonrecursive(parent_key, child)
            else:
                self.children[parent_key].add_child(child_keys[0], child)

        except KeyError as err:
            raise KeyError(f"{key}") from err

    def add_child_nonrecursive(self, key: str, child: TChild):
        """
        Add a primitive indexed by a key

        Base case of recursive `add_child`
        """
        if key in self.children:
            raise KeyError(f"{key}")
        else:
            self.children[key] = child


TItem = TypeVar("TItem")


class ItemCounter(Generic[TItem]):
    """
    Count items by a prefix
    """

    @staticmethod
    def __classname(item: TItem) -> str:
        return type(item).__name__

    def __init__(self, gen_prefix: Callable[[TItem], str] = __classname):
        self._prefix_to_count = {}
        self._gen_prefix = gen_prefix

    @property
    def prefix_to_count(self):
        return self._prefix_to_count

    def __contains__(self, key):
        return key in self._p

    def gen_prefix(self, item: TItem) -> str:
        return self._gen_prefix(item)

    def add_item(self, item: TItem) -> str:
        prefix = self.gen_prefix(item)
        if prefix in self.prefix_to_count:
            self.prefix_to_count[prefix] += 1
        else:
            self.prefix_to_count[prefix] = 1

        postfix = self.prefix_to_count[prefix] - 1
        return f"{prefix}{postfix}"

    def add_item_until_valid(self, item: TItem, valid: Callable[[str], bool]):

        key = self.add_item(item)
        while not valid(key):
            key = self.add_item(item)

        return key

    def add_item_to_nodes(self, item: TItem, *nodes: tuple[Node, ...]):
        def valid(key):
            key_notin_nodes = (key not in node for node in nodes)
            return all(key_notin_nodes)
        return self.add_item_until_valid(item, valid)


## Manual flattening/unflattening implementation
FlatNodeStructure = tuple[type[TNode], str, TValue, int]

def iter_flat(key: str, node: TNode):
    """
    Return a flat iterator over all nodes
    """
    num_child = len(node)

    if num_child == 0:
        nodes = [(key, node)]
    else:
        # TODO: mypy says there's something wrong with the typing here?
        cnodes = [
            iter_flat("/".join((key, ckey)), cnode) for ckey, cnode in node.items()
        ]
        cnodes = itertools.chain(cnodes)

        nodes = itertools.chain([(key, node)], *cnodes)
    return nodes


def flatten(key: str, node: TNode) -> list[FlatNodeStructure]:
    node_structs = [
        (type(_node), _key, _node.value, len(_node))
        for _key, _node in iter_flat(key, node)
    ]
    return node_structs


def unflatten(
    node_structs: list[FlatNodeStructure],
) -> tuple[TNode, list[FlatNodeStructure]]:
    node_type, pkey, value, num_child = node_structs[0]
    node_structs = node_structs[1:]

    if num_child == 0:
        node = node_type.from_tree(value, {})
    else:
        ckeys = []
        children = []
        for _ in range(num_child):
            child_struct = node_structs[0]

            ckey = child_struct[1][len(pkey) + 1 :]
            child, node_structs = unflatten(node_structs)

            ckeys.append(ckey)
            children.append(child)

        node = node_type.from_tree(
            value, {key: child for key, child in zip(ckeys, children)}
        )

    return node, node_structs


## pytree flattening/unflattening implementation
# These functions register `Node` classes as a `jax.pytree` so jax can flatten/unflatten
# them
FlatNode = tuple[TValue, dict[str, TNode]]
AuxData = Any

def _make_flatten_unflatten(node_type: type[TNode]):

    def _flatten_node(node: TNode) -> tuple[FlatNode, AuxData]:
        flat_node = (node.value, node.children)
        aux_data = None
        return (flat_node, aux_data)

    def _unflatten_node(aux_data: AuxData, flat_node: FlatNode) -> TNode:
        value, children = flat_node
        return node_type.from_tree(value, children)

    return _flatten_node, _unflatten_node


## Register `Node` as `jax.pytree`
for _NodeType in [Node]:
    _flatten, _unflatten = _make_flatten_unflatten(_NodeType)
    jax.tree_util.register_pytree_node(_NodeType, _flatten, _unflatten)
