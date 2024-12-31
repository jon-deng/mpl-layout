"""
Tests for `mpllayout.containers`
"""

import pytest

from timeit import timeit

import numpy as np

from mpllayout import containers as cn


class TestNode:

    # NOTE: Test using randomly generated `Node`?
    @pytest.fixture()
    def node(self):
        childd = cn.Node.from_tree(99, {})
        childe = cn.Node.from_tree(9, {})
        childb = cn.Node.from_tree(2, {"d": childd, "e": childe})

        childa = cn.Node.from_tree(1, {})
        childc = cn.Node.from_tree(3, {})
        node = cn.Node.from_tree(0, {"a": childa, "b": childb, "c": childc})
        return node

    def test_node_height(self, node: cn.Node):
        assert node.node_height() == 2

    def test_repr(self, node: cn.Node):
        print(node)

    def test_iter_flat(self, node: cn.Node):
        print([{key: _node} for key, _node in cn.iter_flat("", node)])

    def test_flatten_unflatten_python(self, node: cn.Node):
        fnode_structs = cn.flatten("root", node)
        reconstructed_node, _ = cn.unflatten(fnode_structs)

        print(fnode_structs)

        assert str(node) == str(reconstructed_node)
        print(node)
        print(reconstructed_node)

        N = int(1e5)
        timeit_kwargs = {"globals": {**globals(), **locals()}, "number": N}

        duration = timeit("cn.flatten('root', node)", **timeit_kwargs)
        print(f"Flattening duration: {duration/N: .2e} s")

        duration = timeit("cn.unflatten(fnode_structs)", **timeit_kwargs)
        print(f"Unflattening duration: {duration/N: .2e} s")

    def test_flatten_unflatten_jax(self, node: cn.Node):
        import jax

        flat_tree, flat_tree_def = jax.tree_util.tree_flatten(node)
        reconstructed_node = jax.tree_util.tree_unflatten(flat_tree_def, flat_tree)

        assert str(node) == str(reconstructed_node)
        print(node)
        print(reconstructed_node)

        N = int(1e5)
        timeit_kwargs = {"globals": {**globals(), **locals()}, "number": N}

        duration = timeit("jax.tree_util.tree_flatten(node)", **timeit_kwargs)
        print(f"Flattening duration: {duration/N: .2e} s")

        duration = timeit(
            "jax.tree_util.tree_unflatten(flat_tree_def, flat_tree)", **timeit_kwargs
        )
        print(f"Unflattening duration: {duration/N: .2e} s")

class TestFunctions:

    def test_map(self):
        def fun(x):
            return x+1
        node = cn.Node(1, {'a': cn.Node(1, {}), 'b': cn.Node(1, {}), 'c': cn.Node(1, {})})

        new_node = cn.map(fun, node)

        test_node_values = [fun(_node.value) for _key, _node in cn.iter_flat('', node)]
        new_node_values = [_node.value for _key, _node in cn.iter_flat('', new_node)]
        assert np.all(np.isclose(new_node_values, test_node_values))

    def test_accumulate(self):
        # TODO: Think of how to test this properly
        # Could have two tests to check that recursion works + iterating over different numbers of children works?
        def fun(x, y):
            return x + y

        import string
        values = 5*[1]
        node = cn.Node(0, {key: cn.Node(value, {}) for key, value in zip(string.ascii_lowercase, values)})

        new_node = cn.accumulate(fun, node, 0)

        assert np.isclose(new_node.value, np.sum(values))
