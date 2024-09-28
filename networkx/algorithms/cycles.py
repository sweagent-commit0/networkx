"""
========================
Cycle finding algorithms
========================
"""
from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['cycle_basis', 'simple_cycles', 'recursive_simple_cycles', 'find_cycle', 'minimum_cycle_basis', 'chordless_cycles', 'girth']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def cycle_basis(G, root=None):
    """Returns a list of cycles which form a basis for cycles of G.

    A basis for cycles of a network is a minimal collection of
    cycles such that any cycle in the network can be written
    as a sum of cycles in the basis.  Here summation of cycles
    is defined as "exclusive or" of the edges. Cycle bases are
    useful, e.g. when deriving equations for electric circuits
    using Kirchhoff's Laws.

    Parameters
    ----------
    G : NetworkX Graph
    root : node, optional
       Specify starting node for basis.

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.cycle_basis(G, 0)
    [[3, 4, 5, 0], [1, 2, 3, 0]]

    Notes
    -----
    This is adapted from algorithm CACM 491 [1]_.

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    See Also
    --------
    simple_cycles
    minimum_cycle_basis
    """
    pass

@nx._dispatchable
def simple_cycles(G, length_bound=None):
    """Find simple cycles (elementary circuits) of a graph.

    A `simple cycle`, or `elementary circuit`, is a closed path where
    no node appears twice.  In a directed graph, two simple cycles are distinct
    if they are not cyclic permutations of each other.  In an undirected graph,
    two simple cycles are distinct if they are not cyclic permutations of each
    other nor of the other's reversal.

    Optionally, the cycles are bounded in length.  In the unbounded case, we use
    a nonrecursive, iterator/generator version of Johnson's algorithm [1]_.  In
    the bounded case, we use a version of the algorithm of Gupta and
    Suzumura[2]_. There may be better algorithms for some cases [3]_ [4]_ [5]_.

    The algorithms of Johnson, and Gupta and Suzumura, are enhanced by some
    well-known preprocessing techniques.  When G is directed, we restrict our
    attention to strongly connected components of G, generate all simple cycles
    containing a certain node, remove that node, and further decompose the
    remainder into strongly connected components.  When G is undirected, we
    restrict our attention to biconnected components, generate all simple cycles
    containing a particular edge, remove that edge, and further decompose the
    remainder into biconnected components.

    Note that multigraphs are supported by this function -- and in undirected
    multigraphs, a pair of parallel edges is considered a cycle of length 2.
    Likewise, self-loops are considered to be cycles of length 1.  We define
    cycles as sequences of nodes; so the presence of loops and parallel edges
    does not change the number of simple cycles in a graph.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None, optional (default=None)
       If length_bound is an int, generate all simple cycles of G with length at
       most length_bound.  Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    >>> G = nx.DiGraph(edges)
    >>> sorted(nx.simple_cycles(G))
    [[0], [0, 1, 2], [0, 2], [1, 2], [2]]

    To filter the cycles so that they don't include certain nodes or edges,
    copy your graph and eliminate those nodes or edges before calling.
    For example, to exclude self-loops from the above example:

    >>> H = G.copy()
    >>> H.remove_edges_from(nx.selfloop_edges(G))
    >>> sorted(nx.simple_cycles(H))
    [[0, 1, 2], [0, 2], [1, 2]]

    Notes
    -----
    When length_bound is None, the time complexity is $O((n+e)(c+1))$ for $n$
    nodes, $e$ edges and $c$ simple circuits.  Otherwise, when length_bound > 1,
    the time complexity is $O((c+n)(k-1)d^k)$ where $d$ is the average degree of
    the nodes of G and $k$ = length_bound.

    Raises
    ------
    ValueError
        when length_bound < 0.

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007
    .. [2] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094
    .. [3] Enumerating the cycles of a digraph: a new preprocessing strategy.
       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.
    .. [4] A search strategy for the elementary cycles of a directed graph.
       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,
       v. 16, no. 2, 192-204, 1976.
    .. [5] Optimal Listing of Cycles and st-Paths in Undirected Graphs
        R. Ferreira and R. Grossi and A. Marino and N. Pisanti and R. Rizzi and
        G. Sacomoto https://arxiv.org/abs/1205.2766

    See Also
    --------
    cycle_basis
    chordless_cycles
    """
    pass

def _directed_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for directed graphs.

    We generate all cycles of G through binary partition.

        1. Pick a node v in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the node v.
            b. Recursively generate all cycles of G \\ v.

    This is accomplished through the following:

        1. Compute the strongly connected components SCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each simple cycle P containing v in G[C], yield P.
        4. Add the biconnected components of G[C \\ v] to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to
    simple cycles of length at most length_bound.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """
    pass

def _undirected_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for undirected graphs.

    We generate all cycles of G through binary partition.

        1. Pick an edge (u, v) in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the edge (u, v)
            b. Recursively generate all cycles of G \\ (u, v)

    This is accomplished through the following:

        1. Compute the biconnected components BCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each (v -> u) path P remaining in G[C] \\ (u, v), yield P.
        4. Add the biconnected components of G[C] \\ (u, v) to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to simple paths
    of length at most length_bound.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """
    pass

class _NeighborhoodCache(dict):
    """Very lightweight graph wrapper which caches neighborhoods as list.

    This dict subclass uses the __missing__ functionality to query graphs for
    their neighborhoods, and store the result as a list.  This is used to avoid
    the performance penalty incurred by subgraph views.
    """

    def __init__(self, G):
        self.G = G

    def __missing__(self, v):
        Gv = self[v] = list(self.G[v])
        return Gv

def _johnson_cycle_search(G, path):
    """The main loop of the cycle-enumeration algorithm of Johnson.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
        .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007

    """
    pass

def _bounded_cycle_search(G, path, length_bound):
    """The main loop of the cycle-enumeration algorithm of Gupta and Suzumura.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound: int
        A length bound.  All cycles generated will have length at most length_bound.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094

    """
    pass

@nx._dispatchable
def chordless_cycles(G, length_bound=None):
    """Find simple chordless cycles of a graph.

    A `simple cycle` is a closed path where no node appears twice.  In a simple
    cycle, a `chord` is an additional edge between two nodes in the cycle.  A
    `chordless cycle` is a simple cycle without chords.  Said differently, a
    chordless cycle is a cycle C in a graph G where the number of edges in the
    induced graph G[C] is equal to the length of `C`.

    Note that some care must be taken in the case that G is not a simple graph
    nor a simple digraph.  Some authors limit the definition of chordless cycles
    to have a prescribed minimum length; we do not.

        1. We interpret self-loops to be chordless cycles, except in multigraphs
           with multiple loops in parallel.  Likewise, in a chordless cycle of
           length greater than 1, there can be no nodes with self-loops.

        2. We interpret directed two-cycles to be chordless cycles, except in
           multi-digraphs when any edge in a two-cycle has a parallel copy.

        3. We interpret parallel pairs of undirected edges as two-cycles, except
           when a third (or more) parallel edge exists between the two nodes.

        4. Generalizing the above, edges with parallel clones may not occur in
           chordless cycles.

    In a directed graph, two chordless cycles are distinct if they are not
    cyclic permutations of each other.  In an undirected graph, two chordless
    cycles are distinct if they are not cyclic permutations of each other nor of
    the other's reversal.

    Optionally, the cycles are bounded in length.

    We use an algorithm strongly inspired by that of Dias et al [1]_.  It has
    been modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None, optional (default=None)
       If length_bound is an int, generate all simple cycles of G with length at
       most length_bound.  Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> sorted(list(nx.chordless_cycles(nx.complete_graph(4))))
    [[1, 0, 2], [1, 0, 3], [2, 0, 3], [2, 1, 3]]

    Notes
    -----
    When length_bound is None, and the graph is simple, the time complexity is
    $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$ chordless cycles.

    Raises
    ------
    ValueError
        when length_bound < 0.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    See Also
    --------
    simple_cycles
    """
    pass

def _chordless_cycle_search(F, B, path, length_bound):
    """The main loop for chordless cycle enumeration.

    This algorithm is strongly inspired by that of Dias et al [1]_.  It has been
    modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    F : _NeighborhoodCache
       A graph of forward edges to follow in constructing cycles

    B : _NeighborhoodCache
       A graph of blocking edges to prevent the production of chordless cycles

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound : int
       A length bound.  All cycles generated will have length at most length_bound.


    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    """
    pass

@not_implemented_for('undirected')
@nx._dispatchable(mutates_input=True)
def recursive_simple_cycles(G):
    """Find simple cycles (elementary circuits) of a directed graph.

    A `simple cycle`, or `elementary circuit`, is a closed path where
    no node appears twice. Two elementary circuits are distinct if they
    are not cyclic permutations of each other.

    This version uses a recursive algorithm to build a list of cycles.
    You should probably use the iterator version called simple_cycles().
    Warning: This recursive version uses lots of RAM!
    It appears in NetworkX for pedagogical value.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    Returns
    -------
    A list of cycles, where each cycle is represented by a list of nodes
    along the cycle.

    Example:

    >>> edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    >>> G = nx.DiGraph(edges)
    >>> nx.recursive_simple_cycles(G)
    [[0], [2], [0, 1, 2], [0, 2], [1, 2]]

    Notes
    -----
    The implementation follows pp. 79-80 in [1]_.

    The time complexity is $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$
    elementary circuits.

    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007

    See Also
    --------
    simple_cycles, cycle_basis
    """
    pass

@nx._dispatchable
def find_cycle(G, source=None, orientation=None):
    """Returns a cycle found via depth-first traversal.

    The cycle is a list of edges indicating the cyclic path.
    Orientation of directed edges is controlled by `orientation`.

    Parameters
    ----------
    G : graph
        A directed/undirected graph/multigraph.

    source : node, list of nodes
        The node from which the traversal begins. If None, then a source
        is chosen arbitrarily and repeatedly until all edges from each node in
        the graph are searched.

    orientation : None | 'original' | 'reverse' | 'ignore' (default: None)
        For directed graphs and directed multigraphs, edge traversals need not
        respect the original orientation of the edges.
        When set to 'reverse' every edge is traversed in the reverse direction.
        When set to 'ignore', every edge is treated as undirected.
        When set to 'original', every edge is treated as directed.
        In all three cases, the yielded edge tuples add a last entry to
        indicate the direction in which that edge was traversed.
        If orientation is None, the yielded edge has no direction indicated.
        The direction is respected, but not reported.

    Returns
    -------
    edges : directed edges
        A list of directed edges indicating the path taken for the loop.
        If no cycle is found, then an exception is raised.
        For graphs, an edge is of the form `(u, v)` where `u` and `v`
        are the tail and head of the edge as determined by the traversal.
        For multigraphs, an edge is of the form `(u, v, key)`, where `key` is
        the key of the edge. When the graph is directed, then `u` and `v`
        are always in the order of the actual directed edge.
        If orientation is not None then the edge tuple is extended to include
        the direction of traversal ('forward' or 'reverse') on that edge.

    Raises
    ------
    NetworkXNoCycle
        If no cycle was found.

    Examples
    --------
    In this example, we construct a DAG and find, in the first call, that there
    are no directed cycles, and so an exception is raised. In the second call,
    we ignore edge orientations and find that there is an undirected cycle.
    Note that the second call finds a directed cycle while effectively
    traversing an undirected graph, and so, we found an "undirected cycle".
    This means that this DAG structure does not form a directed tree (which
    is also known as a polytree).

    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    >>> nx.find_cycle(G, orientation="original")
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXNoCycle: No cycle found.
    >>> list(nx.find_cycle(G, orientation="ignore"))
    [(0, 1, 'forward'), (1, 2, 'forward'), (0, 2, 'reverse')]

    See Also
    --------
    simple_cycles
    """
    pass

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def minimum_cycle_basis(G, weight=None):
    """Returns a minimum weight cycle basis for G

    Minimum weight means a cycle basis for which the total weight
    (length for unweighted graphs) of all the cycles is minimum.

    Parameters
    ----------
    G : NetworkX Graph
    weight: string
        name of the edge attribute to use for edge weights

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G. Note that the nodes are not
    necessarily returned in a order by which they appear in the cycle

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.minimum_cycle_basis(G)
    [[5, 4, 3, 0], [3, 2, 1, 0]]

    References:
        [1] Kavitha, Telikepalli, et al. "An O(m^2n) Algorithm for
        Minimum Cycle Basis of Graphs."
        http://link.springer.com/article/10.1007/s00453-007-9064-z
        [2] de Pina, J. 1995. Applications of shortest path methods.
        Ph.D. thesis, University of Amsterdam, Netherlands

    See Also
    --------
    simple_cycles, cycle_basis
    """
    pass

def _min_cycle(G, orth, weight):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    Use (u, 1) to indicate the lifted copy of u (denoted u' in paper).
    """
    pass

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def girth(G):
    """Returns the girth of the graph.

    The girth of a graph is the length of its shortest cycle, or infinity if
    the graph is acyclic. The algorithm follows the description given on the
    Wikipedia page [1]_, and runs in time O(mn) on a graph with m edges and n
    nodes.

    Parameters
    ----------
    G : NetworkX Graph

    Returns
    -------
    int or math.inf

    Examples
    --------
    All examples below (except P_5) can easily be checked using Wikipedia,
    which has a page for each of these famous graphs.

    >>> nx.girth(nx.chvatal_graph())
    4
    >>> nx.girth(nx.tutte_graph())
    4
    >>> nx.girth(nx.petersen_graph())
    5
    >>> nx.girth(nx.heawood_graph())
    6
    >>> nx.girth(nx.pappus_graph())
    6
    >>> nx.girth(nx.path_graph(5))
    inf

    References
    ----------
    .. [1] `Wikipedia: Girth <https://en.wikipedia.org/wiki/Girth_(graph_theory)>`_

    """
    pass