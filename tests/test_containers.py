"""
Tests for `mpllayout.containers`
"""

import pytest

from timeit import timeit

from mpllayout import containers as cn



class TestNode:

    @pytest.fixture()
    def node(self):
        node = cn.Node(0, (cn.Node(1, (), ()), cn.Node(2, (), ()), cn.Node(3, (), ())), ('a', 'b', 'c'))
        return node

    def test_flatten_unflatten(self, node: cn.Node):
        fnode_structs = cn.flatten('root', node)

        _node, _ = cn.unflatten(fnode_structs)

        N = int(1e6)
        duration = timeit("cn.flatten('root', node)", globals={**globals(), **locals()}, number=N)
        print(f"Flattening duration: {duration/N: .2e} s")

        N = int(1e6)
        duration = timeit("cn.unflatten(fnode_structs)", globals={**globals(), **locals()}, number=N)
        print(f"Unflattening duration: {duration/N: .2e} s")



