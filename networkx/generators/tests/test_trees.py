import pytest

import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
import random


@pytest.mark.parametrize("prefix_tree_fn",
                         (nx.prefix_tree, nx.prefix_tree_recursive))
def test_basic_prefix_tree(prefix_tree_fn):
    # This example is from the Wikipedia article "Trie"
    # <https://en.wikipedia.org/wiki/Trie>.
    strings = ["a", "to", "tea", "ted", "ten", "i", "in", "inn"]
    T = prefix_tree_fn(strings)
    root, NIL = 0, -1

    def source_label(v):
        return T.nodes[v]["source"]

    # First, we check that the tree has the expected
    # structure. Recall that each node that corresponds to one of
    # the input strings has an edge to the NIL node.
    #
    # Consider the three children at level 1 in the trie.
    a, i, t = sorted(T[root], key=source_label)
    # Check the 'a' branch.
    assert len(T[a]) == 1
    nil = arbitrary_element(T[a])
    assert len(T[nil]) == 0
    # Check the 'i' branch.
    assert len(T[i]) == 2
    nil, in_ = sorted(T[i], key=source_label)
    assert len(T[nil]) == 0
    assert len(T[in_]) == 2
    nil, inn = sorted(T[in_], key=source_label)
    assert len(T[nil]) == 0
    assert len(T[inn]) == 1
    nil = arbitrary_element(T[inn])
    assert len(T[nil]) == 0
    # Check the 't' branch.
    te, to = sorted(T[t], key=source_label)
    assert len(T[to]) == 1
    nil = arbitrary_element(T[to])
    assert len(T[nil]) == 0
    tea, ted, ten = sorted(T[te], key=source_label)
    assert len(T[tea]) == 1
    assert len(T[ted]) == 1
    assert len(T[ten]) == 1
    nil = arbitrary_element(T[tea])
    assert len(T[nil]) == 0
    nil = arbitrary_element(T[ted])
    assert len(T[nil]) == 0
    nil = arbitrary_element(T[ten])
    assert len(T[nil]) == 0

    # Next, we check that the "sources" of each of the nodes is the
    # rightmost letter in the string corresponding to the path to
    # that node.
    assert source_label(root) is None
    assert source_label(a) == "a"
    assert source_label(i) == "i"
    assert source_label(t) == "t"
    assert source_label(in_) == "n"
    assert source_label(inn) == "n"
    assert source_label(to) == "o"
    assert source_label(te) == "e"
    assert source_label(tea) == "a"
    assert source_label(ted) == "d"
    assert source_label(ten) == "n"
    assert source_label(NIL) == "NIL"


@pytest.mark.parametrize(
    "strings",
    (
        ["a", "to", "tea", "ted", "ten", "i", "in", "inn"],
        ["ab", "abs", "ad"],
        ["ab", "abs", "ad", ""],
        ["distant", "disparaging", "distant", "diamond", "ruby"],
    ),
)
def test_implementations_consistent(strings):
    """Ensure results are consistent between prefix_tree implementations."""
    assert graphs_equal(nx.prefix_tree(strings),
                        nx.prefix_tree_recursive(strings))


def test_random_tree():
    """Tests that a random tree is in fact a tree."""
    T = nx.random_tree(10, seed=1234)
    assert nx.is_tree(T)
    T = nx.random_tree(10, number_of_trees=5, seed=43)
    assert len(T) == 5
    for t in T:
        assert nx.is_tree(t)


def test_random_directed_tree():
    """Generates a directed tree."""
    T = nx.random_tree(10, seed=1234, create_using=nx.DiGraph())
    assert T.is_directed()


def test_random_tree_n_zero():
    """Tests if n = 0 then the NetworkXPointlessConcept exception is raised."""
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_tree(0, labeled=True, seed=1234)
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_tree(0, labeled=False, seed=1234)
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_rooted_tree(0, labeled=True, seed=1234)
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_rooted_tree(0, labeled=False, seed=1234)


def test_random_tree_using_generator():
    """Tests that creating a ramdom tree with a generator works"""
    G = nx.Graph()
    T = nx.random_tree(10, seed=1234, create_using=G)
    assert nx.is_tree(T)


def test_random_unlabeled_rooted_tree():
    for i in range(1, 10):
        t1 = nx.random_rooted_tree(i, labeled=False, seed=42)
        t2 = nx.random_rooted_tree(i, labeled=False, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        assert nx.is_tree(t1)
        assert "root" in t1.graph
        assert "roots" not in t1.graph
    random.seed(43)
    t = nx.random_rooted_tree(15, labeled=False, number_of_trees=10, seed=43)
    random.seed(43)
    s = nx.random_rooted_tree(15, labeled=False, number_of_trees=10, seed=43)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        assert nx.is_tree(t[i])
        assert "root" in t[i].graph
        assert "roots" not in t[i].graph


def test_random_labeled_rooted_tree():
    for i in range(1, 10):
        t1 = nx.random_rooted_tree(i, seed=42)
        t2 = nx.random_rooted_tree(i, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        assert nx.is_tree(t1)
        assert "root" in t1.graph
        assert "roots" not in t1.graph
    random.seed(43)
    t = nx.random_rooted_tree(15, number_of_trees=10, seed=43)
    random.seed(43)
    s = nx.random_rooted_tree(15, number_of_trees=10, seed=43)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        assert nx.is_tree(t[i])
        assert "root" in t[i].graph
        assert "roots" not in t[i].graph


def test_random_unlabeled_rooted_forest():
    raised = False
    try:
        nx.random_rooted_forest(10, 0, labeled=False, seed=42)
    except ValueError:
        raised = True
    assert raised
    for i in range(1, 10):
        for q in range(1, i + 1):
            t1 = nx.random_rooted_forest(i, q, labeled=False, seed=42)
            t2 = nx.random_rooted_forest(i, q, labeled=False, seed=42)
            assert nx.utils.misc.graphs_equal(t1, t2)
            for c in nx.connected_components(t1):
                assert nx.is_tree(t1.subgraph(c))
                assert len(c) <= q
            assert "root" not in t1.graph
            assert "roots" in t1.graph
    random.seed(43)
    t = nx.random_rooted_forest(
        15, number_of_forests=10, labeled=False, seed=random)
    random.seed(43)
    s = nx.random_rooted_forest(
        15, number_of_forests=10, labeled=False, seed=random)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        for c in nx.connected_components(t[i]):
            assert nx.is_tree(t[i].subgraph(c))
        assert "root" not in t[i].graph
        assert "roots" in t[i].graph


def test_random_labeled_rooted_forest():
    raised = False
    for i in range(1, 10):
        t1 = nx.random_rooted_forest(i, labeled=True, seed=42)
        t2 = nx.random_rooted_forest(i, labeled=True, seed=42)
        print(t1.edges(), t1.graph["roots"])
        print(t2.edges(), t2.graph["roots"])
        assert nx.utils.misc.graphs_equal(t1, t2)
        for c in nx.connected_components(t1):
            assert nx.is_tree(t1.subgraph(c))
        assert "root" not in t1.graph
        assert "roots" in t1.graph
    random.seed(43)
    t = nx.random_rooted_forest(
        15, number_of_forests=10, labeled=True, seed=random)
    random.seed(43)
    s = nx.random_rooted_forest(
        15, number_of_forests=10, labeled=True, seed=random)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        for c in nx.connected_components(t[i]):
            assert nx.is_tree(t[i].subgraph(c))
        assert "root" not in t[i].graph
        assert "roots" in t[i].graph


def test_random_unlabeled_tree():
    for i in range(1, 10):
        t1 = nx.random_tree(i, labeled=False, seed=42)
        t2 = nx.random_tree(i, labeled=False, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        assert nx.is_tree(t1)
        assert "root" not in t1.graph
        assert "roots" not in t1.graph
    random.seed(43)
    t = nx.random_tree(10, labeled=False, number_of_trees=10, seed=random)
    random.seed(43)
    s = nx.random_tree(10, labeled=False, number_of_trees=10, seed=random)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        assert nx.is_tree(t[i])
        assert "root" not in t[i].graph
        assert "roots" not in t[i].graph


def test_multiple_created_using_instance():
    with pytest.raises(nx.NetworkXException):
        T = nx.random_tree(0, labeled=True, number_of_trees=2,
                           seed=1234, create_using=nx.Graph())
    with pytest.raises(nx.NetworkXException):
        T = nx.random_tree(0, labeled=False, number_of_trees=2,
                           seed=1234, create_using=nx.Graph())
    with pytest.raises(nx.NetworkXException):
        T = nx.random_rooted_tree(
            0, labeled=True, number_of_trees=2, seed=1234, create_using=nx.Graph())
    with pytest.raises(nx.NetworkXException):
        T = nx.random_rooted_tree(
            0, labeled=False, number_of_trees=2, seed=1234, create_using=nx.Graph())
    with pytest.raises(nx.NetworkXException):
        T = nx.random_rooted_forest(
            0, number_of_forests=2, seed=1234, create_using=nx.Graph())


def test_random_q_labeled_forests():
    with pytest.raises(nx.NetworkXException):
        T = nx.random_rooted_forest(0, number_of_forests=2, q=1, seed=1234)


def test_random_forest_n_zero():
    """Tests generation of empty forests."""
    F = nx.random_rooted_forest(0, labeled=False, seed=1234)
    assert (len(F) == 0)
    assert (len(F.graph["roots"]) == 0)
    F = nx.random_rooted_forest(0, labeled=True, seed=1234)
    assert (len(F) == 0)
    assert (len(F.graph["roots"]) == 0)
