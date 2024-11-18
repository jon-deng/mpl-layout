"""
Tree class (`Node`) definition and related utilities for trees

This module defines a tree class `Node` and related utilities.
This class is used by itself as well as to define geometric primitives and constraints.
"""

from typing import TypeVar, Generic, Any, Iterable, Callable

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
        """
        Return any `Node` subclass from its value and children

        This method is needed because some `Node` subclasses have different `__init__`
        signatures.
        `from_tree` can be used to recreate any `Node` subclass using just a known value
        and children.
        This is particularly important for flattening and unflattening a tree.
        """
        node = super().__new__(cls)
        Node.__init__(node, value, children)
        return node

    @property
    def value(self) -> TValue:
        """
        Return the node value
        """
        return self._value

    @property
    def children(self) -> dict[str, TChild]:
        """
        Return all child nodes
        """
        return self._children

    ## Tree methods

    def node_height(self) -> int:
        """
        Return the height of a node

        Returns
        -------
        int
            The node height

            The node height is the number of edges from the current node to the
            "furthest" child node.
        """
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

    def keys(self) -> list[str]:
        """
        Return child keys
        """
        return list(self.children.keys())

    def values(self) -> list[TChild]:
        """
        Return child nodes
        """
        return list(self.children.values())

    def items(self):
        """
        Return an iterator of (child key, child node) pairs
        """
        return self.children.items()

    def __setitem__(self, key: str, node: TChild):
        """
        Set the child node at the given key

        Raises an error if the key doesn't exist.

        Parameters
        ----------
        key: str
            A child node key

            see `__getitem__`
        node: TChild
            The node to set
        """
        # This splits `key = 'a/b/c/d'`
        # into `parent_key = 'a/b/c'` and `child_key = 'd'`
        split_keys = key.split("/")
        parent_key = "/".join(split_keys[:-1])
        child_key = split_keys[-1]

        if key not in self:
            raise KeyError(key)
        else:
            if parent_key == "":
                parent = self
            else:
                parent = self[parent_key]
            parent.children[child_key] = node

    def __getitem__(self, key: str | int | slice) -> TChild | list[TChild]:
        """
        Return the value indexed by a slash-separated key

        Parameters
        ----------
        key: str | int | slice
            A child node key

            The interpretation depends on the key type:
            - `str` keys indicate a child key and can be slash separated to denote
                child keys of child keys,
                for example, 'childa/grandchildb/greatgrandchildc'.
            - `int` keys indicate a child by integer index.
            - `slice` keys indicate a range of children.
        """
        if isinstance(key, str):
            return self._get_child_from_str(key)
        elif isinstance(key, (int, slice)):
            return self._get_child_from_int_or_slice(key)
        else:
            raise TypeError("")

    def _get_child_from_int_or_slice(self, key: int | slice) -> TChild | list[TChild]:
        return list(self.children.values())[key]

    def _get_child_from_str(self, key: str) -> TChild:
        split_key = key.split("/", 1)
        parent_key, child_keys = split_key[0], split_key[1:]

        try:
            if len(child_keys) == 0:
                return self._get_child_from_str_nonrecursive(parent_key)
            else:
                return self.children[parent_key]._get_child_from_str(child_keys[0])
        except KeyError as err:
            raise KeyError(f"{key}") from err

    def _get_child_from_str_nonrecursive(self, key: str) -> TChild:
        return self.children[key]

    def add_child(self, key: str, child: TChild):
        """
        Add a child node at the given key

        Raises an error if the key already exists.

        Parameters
        ----------
        key: str
            A child node key

            see `__getitem__`
        node: TChild
            The node to set
        """
        split_key = key.split("/", 1)
        parent_key, child_keys = split_key[0], split_key[1:]

        try:
            if len(child_keys) == 0:
                self._add_child_nonrecursive(parent_key, child)
            else:
                self.children[parent_key].add_child(child_keys[0], child)

        except KeyError as err:
            raise KeyError(f"{key}") from err

    def _add_child_nonrecursive(self, key: str, child: TChild):
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
    Count the number of added items by category (a string)

    This is used to generate unique string keys for objects
    (see `Layout.add_constraint`).

    Parameters
    ----------
    categorize: Callable[[TItem], str]
        A function that returns the category (string) of an item

    Attributes
    ----------
    category_counts: dict[str, int]
        The number of items added to each category
    """

    @staticmethod
    def categorize_by_classname(item: TItem) -> str:
        return type(item).__name__

    def __init__(self, categorize: Callable[[TItem], str] = categorize_by_classname):
        self._category_counts = {}
        self._categorize = categorize

    @property
    def category_counts(self) -> dict[str, int]:
        return self._category_counts

    def __contains__(self, key):
        return key in self._p

    def categorize(self, item: TItem) -> str:
        """
        Return the category string of an item
        """
        return self._categorize(item)

    def add_item(self, item: TItem) -> str:
        """
        Add an item

        Parameters
        ----------
        item: TItem
            The item to add

        Returns
        -------
        str
            A string identifying the added item's category and count
        """
        category = self.categorize(item)
        if category in self.category_counts:
            self.category_counts[category] += 1
        else:
            self.category_counts[category] = 1

        count = self.category_counts[category] - 1
        return f"{category}{count}"

    def add_item_until_valid(self, item: TItem, valid: Callable[[str], bool]) -> str:
        """
        Add an item until the return item key is valid

        This can be used to keep adding items until a unique key is generated for some
        existing collection.
        For example, if a dictionary of items already exists, this function can be used
        to generate new item keys until one that doesn't already exist in the dictionary
        is found.

        Parameters
        ----------
        item: TItem
            The item to add
        valid: Callable[[str], bool]
            The condition the generated item key must satisfy

        Returns
        -------
        str
            A string identifying the added item's category and count
        """
        key = self.add_item(item)
        while not valid(key):
            key = self.add_item(item)

        return key

    def add_item_to_nodes(self, item: TItem, *nodes: tuple[Node, ...]) -> str:
        """
        Add an item until the item key is unique within a set of trees

        This is used generate a unique key for an item for a set of existing trees
        (`Node`).

        Parameters
        ----------
        item: TItem
            The item to add
        *nodes: tuple[Node, ...]
            The set of trees

            The returned item key should exist in these trees.

        Returns
        -------
        str
            A string identifying the added item's category and count
        """
        def valid(key):
            key_notin_nodes = (key not in node for node in nodes)
            return all(key_notin_nodes)
        return self.add_item_until_valid(item, valid)


## Manual flattening/unflattening implementation

def iter_flat(root_key: str, root_node: TNode) -> Iterable[tuple[str, TNode]]:
    """
    Return an iterable over all nodes in the root node (recursively depth-first)

    Parameters
    ----------
    root_key: str
        A key for the node

        All child node keys will be appended to this key with a '/' separator.
    root_node: TNode
        The root node

    Returns
    -------
    Iterable[tuple[str, TNode]]
        An iterable over all nodes in the root node
    """
    num_child = len(root_node)

    if num_child == 0:
        nodes = [(root_key, root_node)]
    else:
        # TODO: mypy says there's something wrong with the typing here?
        cnodes = [
            iter_flat("/".join((root_key, ckey)), cnode) for ckey, cnode in root_node.items()
        ]
        cnodes = itertools.chain(cnodes)

        nodes = itertools.chain([(root_key, root_node)], *cnodes)
    # TODO: Rewrite this in a tail-call form?
    return nodes

# TODO: Refactor `flatten` and `unflatten`?
# The current implementation seems weird
FlatNodeStructure = tuple[type[TNode], str, TValue, int]

def flatten(root_key: str, root_node: TNode) -> list[FlatNodeStructure]:
    """
    Return a flattened list of node structures for a root node (recursively depth-first)

    Parameters
    ----------
    root_key: str
        A key for the node
    root_node: TNode
        The root node

    Returns
    -------
    list[FlatNodeStructure]
        A list of node structures

        Each node structure is a tuple representing the node.
    """
    node_structs = [
        (type(node), key, node.value, len(node))
        for key, node in iter_flat(root_key, root_node)
    ]
    return node_structs

def unflatten(
    node_structs: list[FlatNodeStructure],
) -> tuple[TNode, list[FlatNodeStructure]]:
    """
    Return the root node from a flat representation

    Parameters
    ----------
    node_structs: list[FlatNodeStructure]
        The flat representation

    Returns
    -------
    TNode
        The root node
    list[FlatNodeStructure]
        A "leftover" flat node representation

        This list should be empty if the flat node representation only contains
        nodes that belong to the root node.
    """
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
