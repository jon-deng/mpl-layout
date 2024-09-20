"""
Tests for `mpllayout.containers`
"""

import pytest

from timeit import timeit

from mpllayout import containers as cn



class TestNode:

    @pytest.fixture()
    def node(self):
        childd = cn.Node(99, (), ())
        childe = cn.Node(9, (), ())
        childb = cn.Node(2, (childd, childe), ('d', 'e'))


        childa = cn.Node(1, (), ())
        childc = cn.Node(3, (), ())
        node = cn.Node(0, (childa, childb, childc), ('a', 'b', 'c'))
        return node

    def test_repr(self, node: cn.Node):
        print(node)

    def test_iter_flat(self, node: cn.Node):
        print([{key: _node} for key, _node in cn.iter_flat('', node)])

    def test_flatten_unflatten_python(self, node: cn.Node):
        fnode_structs = cn.flatten('root', node)
        reconstructed_node, _ = cn.unflatten(fnode_structs)

        print(fnode_structs)

        assert str(node) == str(reconstructed_node)
        print(node)
        print(reconstructed_node)


        N = int(1e5)
        timeit_kwargs = {
            'globals': {**globals(), **locals()},
            'number': N
        }

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
        timeit_kwargs = {
            'globals': {**globals(), **locals()},
            'number': N
        }

        duration = timeit(
            "jax.tree_util.tree_flatten(node)", **timeit_kwargs
        )
        print(f"Flattening duration: {duration/N: .2e} s")

        duration = timeit(
            "jax.tree_util.tree_unflatten(flat_tree_def, flat_tree)", **timeit_kwargs
        )
        print(f"Unflattening duration: {duration/N: .2e} s")

