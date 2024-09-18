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
        self,
        value: tp.Union[None, T],
        children: tp.List["Node"],
        keys: tp.List[str]
    ):
        self._value = value
        self._children = children
        self._keys = keys

        if len(children) == len(keys):
            self._key_to_child = {
                label: cnode for label, cnode in zip(keys, children)
            }
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
    def value(self):
        """
        Return the value
        """
        return self._value

    ## Flattened interface

    ## String

    def __repr__(self):
        keys_repr = ', '.join(self.keys())
        children_repr = ', '.join([node.__repr__() for node in self.children])
        return f"{type(self).__name__}({self.value}, ({children_repr}), ({keys_repr}))"

    def __str__(self):
        return self.__repr__()

    ## Dict-like interface

    def __len__(self):
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

    def items(self, flat: bool = False) -> tp.List[tp.Tuple[str, T]]:
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


NodeType = tp.Type[Node]
FlatNodeStructure = tp.Tuple[NodeType, str, T, int]

def iter_flat(
    key: str, node: Node
) -> tp.Iterable[tp.Tuple[str, Node]]:
    """
    Return a flat iterator over all nodes
    """
    num_child = len(node)

    if num_child == 0:
        nodes = [(key, node)]
    else:
        cnodes = [
            iter_flat('/'.join((key, ckey)), cnode)
            for ckey, cnode in zip(node.keys(), node.values())
        ]
        cnodes = itertools.chain(cnodes)

        nodes = itertools.chain([(key, node)], *cnodes)
    return nodes

def flatten(
    key: str, node: Node
) -> tp.List[FlatNodeStructure]:
    node_structs = [
        (type(_node), _key, _node.value, len(_node))
        for _key, _node in iter_flat(key, node)
    ]
    return node_structs

def unflatten(
    node_structs: tp.List[FlatNodeStructure]
) -> tp.Tuple[Node, tp.List[FlatNodeStructure]]:
    NodeType, pkey, value, num_child = node_structs[0]

    if num_child == 0:
        node = NodeType(value, (), ())
    else:
        ckeys = [struct[1][len(pkey)+1:] for struct in node_structs[1:num_child+1]]
        children = []
        node_structs = node_structs[1:]
        for _ in range(num_child):
            child, node_structs = unflatten(node_structs)
            children.append(child)

        node = NodeType(value, children, ckeys)

    return node, node_structs

class LabelledContainer(tp.Generic[T]):
    """
    A generic container with both string and integer indices

    Parameters
    ----------
    items: tp.Optional[tp.List[T]]
        A list of items in the container
    keys: tp.Optional[tp.List[str]]
        A list of keys for each item
    """

    def __init__(
        self,
        items: tp.Optional[tp.List[T]] = None,
        keys: tp.Optional[tp.List[str]] = None,
    ):
        # This stores the count of items of each type in the container,
        # which is used for automatically generating names.
        self._type_to_count = Counter()

        # Add the `self._items` and `self._label_to_idx` attributes.
        # These
        if items is None and keys is None:
            self._items = []
            self._label_to_idx = {}
        elif items is not None and keys is None:
            self._items = []
            self._label_to_idx = {}
            for item in items:
                append(self._items, self._label_to_idx, item, self._type_to_count)
        elif items is None and keys is not None:
            raise ValueError("No `items` supplied")
        else:
            # Check that there's one key for each item
            # and that there are no duplicate keys
            assert len(items) == len(keys)
            assert len(set(keys)) == len(keys)
            self._items = items
            self._label_to_idx = {key: ii for ii, key in enumerate(keys)}

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.values())}, {list(self.keys())})"

    def __str__(self):
        return str(self._items)

    ## List/Dict interface
    def __len__(self):
        return len(self._items)

    def __getitem__(self, key: tp.Union[str, int]) -> T:
        key = self.key_to_idx(key)
        return self._items[key]

    def keys(self):
        return self._label_to_idx.keys()

    def values(self):
        return self._items

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def key_to_idx(self, key: tp.Union[str, int, slice]):
        """
        Return the integer index (indices) corresponding to a string label
        """
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self._label_to_idx[key]
        else:
            raise TypeError(f"`key` must be `str` or `int`, not `{type(key)}`")


class LabelledList(LabelledContainer[T]):
    """
    A list with both string and integer indices

    Parameters
    ----------
    items: tp.Optional[tp.List[T]]
        A list of items in the container
    keys: tp.Optional[tp.List[str]]
        A list of keys for each item
    """

    def __init__(
        self,
        items: tp.Optional[tp.List[T]] = None,
        keys: tp.Optional[tp.List[str]] = None,
    ):

        super().__init__(items, keys)

        self._items = list(self._items)

    def append(self, item: T, label: tp.Optional[str] = None) -> str:
        label, *_ = append(
            self._items, self._label_to_idx, item, self._type_to_count, label
        )
        return label


class LabelledTuple(LabelledContainer[T]):
    """
    A tuple with both string and integer indices

    Parameters
    ----------
    items: tp.Optional[tp.List[T]]
        A list of items in the container
    keys: tp.Optional[tp.List[str]]
        A list of keys for each item
    """

    def __init__(
        self,
        items: tp.Optional[tp.List[T]] = None,
        keys: tp.Optional[tp.List[str]] = None,
    ):

        super().__init__(items, keys)

        self._items = tuple(self._items)


class Counter:
    """
    A class used to count keys added to it

    This is used to create unique names for added items in `LabelledContainer`
    type objects.
    """

    def __init__(self):
        self._count: tp.Mapping[str, int] = {}

    @property
    def count(self):
        return self._count

    def __in__(self, key):
        return key in self.count

    def __getitem__(self, key):
        return self.count.get(key, 0)

    def add(self, key: str):
        """
        Add a string to the `Counter` instance

        If the string already exists in the counter, then its count is
        incremented by 1.

        Parameters
        ----------
        key: str
        """
        if key in self.count:
            self.count[key] += 1
        else:
            self.count[key] = 1


def append(
    items: tp.List[T],
    label_to_idx: tp.Mapping[str, int],
    item: T,
    counter: Counter,
    label: tp.Optional[str] = None,
) -> tp.Tuple[str, tp.List[T], tp.Mapping[str, int]]:
    """
    Add an item to a labelled list with unique string labels

    New items are added

    Parameters
    ----------
    items: tp.List[T]
        The list of items to append to
    label_to_idx: tp.Mapping[str, int]
        A dictionary of string labels for each integer index in `items`
    item: T
        The item to add to `items`
    counter: Counter
        A `Counter` instance tracking what strings have been added to `items`
        and `label_to_idx`

        What items have been added to `items` are categorized into string labels
        by the class name. For example, if a `ClassA` instance were added to
        `items` without a specified label and no other `ClassA` instances
        were in `counter`, then an automatic string label of `ClassA0` would
        be created. If 5 other `ClassA` instances had been added and tracked in
        `counter`, then an automatic string label of `ClassA5` would be created.
    label: tp.Optional[str]
        The string label for the added item

        If not provided, an automatic label will be created based on category
        counts tracked through `counter`.

    Returns
    -------
    label: str
        The string label for the added item
    items: tp.List[T]
        The new list of items
    label_to_idx: tp.Mapping[str, int]
        The new dictionary of string labels for each integer index in `items`
    """
    item_class_name = item.__class__.__name__
    counter.add(item_class_name)

    if label is None:
        n = counter[item_class_name] - 1
        label = f"{item_class_name}{n:d}"

    assert label not in label_to_idx
    items.append(item)
    label_to_idx[label] = len(items) - 1
    return label, items, label_to_idx
