import typing as typ

T = typ.TypeVar('T')

class LabelledContainer(typ.Generic[T]):
    """
    A generic container object with both string and integer indices
    """
    def __init__(
            self,
            items: typ.Optional[typ.List[T]]=None,
            keys: typ.Optional[typ.List[str]]=None
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

    def __getitem__(self, key: typ.Union[str, int]) -> T:
        key = self.key_to_idx(key)
        return self._items[key]

    def keys(self):
        return self._label_to_idx.keys()

    def values(self):
        return self._items

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def key_to_idx(self, key: typ.Union[str, int, slice]):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self._label_to_idx[key]
        else:
            raise TypeError(f"`key` must be `str` or `int`, not `{type(key)}`")

class LabelledList(LabelledContainer[T]):
    """
    A list with both string and integer indices
    """

    def __init__(
            self,
            items: typ.Optional[typ.List[T]]=None,
            keys: typ.Optional[typ.List[str]]=None
        ):

        super().__init__(items, keys)

        self._items = list(self._items)

    def append(self, item: T, label: typ.Optional[str]=None) -> str:
        label, *_ = append(self._items, self._label_to_idx, item, self._type_to_count, label)
        return label

class LabelledTuple(LabelledContainer[T]):
    """
    A tuple with both string and integer indices
    """

    def __init__(
            self,
            items: typ.Optional[typ.List[T]]=None,
            keys: typ.Optional[typ.List[str]]=None
        ):

        super().__init__(items, keys)

        self._items = tuple(self._items)

class Counter:
    """
    A class used to count the nunber of added objects by class

    This is used to create unique names for added items in `LabelledContainer`
    type objects.
    """

    def __init__(self):
        self._count = {}

    @property
    def count(self):
        return self._count

    def __in__(self, key):
        return key in self.count

    def __getitem__(self, key):
        return self.count.get(key, 0)

    def add(self, item):
        if item in self.count:
            self.count[item] += 1
        else:
            self.count[item] = 1

def append(
        items: typ.List[T], label_to_idx: typ.Mapping[str, int],
        item: T,
        counter: Counter,
        label: typ.Optional[str]=None
    ) -> typ.Tuple[typ.List[T], str]:
    """
    Add an item to a list with labelled indices
    """
    item_class_name = item.__class__.__name__
    counter.add(item_class_name)

    if label is None:
        n =  counter[item_class_name] - 1
        label = f'{item_class_name}{n:d}'

    assert label not in label_to_idx
    items.append(item)
    label_to_idx[label] = len(items)-1
    return label, items, label_to_idx
