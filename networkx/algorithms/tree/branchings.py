"""
Algorithms for finding optimum branchings and spanning arborescences.

This implementation is based on:

    J. Edmonds, Optimum branchings, J. Res. Natl. Bur. Standards 71B (1967),
    233–240. URL: http://archive.org/details/jresv71Bn4p233

"""
import string
from dataclasses import dataclass, field
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
__all__ = ['branching_weight', 'greedy_branching', 'maximum_branching', 'minimum_branching', 'minimal_branching', 'maximum_spanning_arborescence', 'minimum_spanning_arborescence', 'ArborescenceIterator', 'Edmonds']
KINDS = {'max', 'min'}
STYLES = {'branching': 'branching', 'arborescence': 'arborescence', 'spanning arborescence': 'arborescence'}
INF = float('inf')

@nx._dispatchable(edge_attrs={'attr': 'default'})
def branching_weight(G, attr='weight', default=1):
    """
    Returns the total weight of a branching.

    You must access this function through the networkx.algorithms.tree module.

    Parameters
    ----------
    G : DiGraph
        The directed graph.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.

    Returns
    -------
    weight: int or float
        The total weight of the branching.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([(0, 1, 2), (1, 2, 4), (2, 3, 3), (3, 4, 2)])
    >>> nx.tree.branching_weight(G)
    11

    """
    pass

@py_random_state(4)
@nx._dispatchable(edge_attrs={'attr': 'default'}, returns_graph=True)
def greedy_branching(G, attr='weight', default=1, kind='max', seed=None):
    """
    Returns a branching obtained through a greedy algorithm.

    This algorithm is wrong, and cannot give a proper optimal branching.
    However, we include it for pedagogical reasons, as it can be helpful to
    see what its outputs are.

    The output is a branching, and possibly, a spanning arborescence. However,
    it is not guaranteed to be optimal in either case.

    Parameters
    ----------
    G : DiGraph
        The directed graph to scan.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.
    kind : str
        The type of optimum to search for: 'min' or 'max' greedy branching.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    B : directed graph
        The greedily obtained branching.

    """
    pass

class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

    Why do we need this? Edmonds algorithm requires that we track edges, even
    as we change the head and tail of an edge, and even changing the weight
    of edges. We must reliably track edges across graph mutations.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        cls = super()
        cls.__init__(incoming_graph_data=incoming_graph_data, **attr)
        self._cls = cls
        self.edge_index = {}
        import warnings
        msg = 'MultiDiGraph_EdgeKey has been deprecated and will be removed in NetworkX 3.4.'
        warnings.warn(msg, DeprecationWarning)

    def add_edge(self, u_for_edge, v_for_edge, key_for_edge, **attr):
        """
        Key is now required.

        """
        pass

def get_path(G, u, v):
    """
    Returns the edge keys of the unique path between u and v.

    This is not a generic function. G must be a branching and an instance of
    MultiDiGraph_EdgeKey.

    """
    pass

class Edmonds:
    """
    Edmonds algorithm [1]_ for finding optimal branchings and spanning
    arborescences.

    This algorithm can find both minimum and maximum spanning arborescences and
    branchings.

    Notes
    -----
    While this algorithm can find a minimum branching, since it isn't required
    to be spanning, the minimum branching is always from the set of negative
    weight edges which is most likely the empty set for most graphs.

    References
    ----------
    .. [1] J. Edmonds, Optimum Branchings, Journal of Research of the National
           Bureau of Standards, 1967, Vol. 71B, p.233-240,
           https://archive.org/details/jresv71Bn4p233

    """

    def __init__(self, G, seed=None):
        self.G_original = G
        self.store = True
        self.edges = []
        self.template = random_string(seed=seed) + '_{0}'
        import warnings
        msg = 'Edmonds has been deprecated and will be removed in NetworkX 3.4. Please use the appropriate minimum or maximum branching or arborescence function directly.'
        warnings.warn(msg, DeprecationWarning)

    def _init(self, attr, default, kind, style, preserve_attrs, seed, partition):
        """
        So we need the code in _init and find_optimum to successfully run edmonds algorithm.
        Responsibilities of the _init function:
        - Check that the kind argument is in {min, max} or raise a NetworkXException.
        - Transform the graph if we need a minimum arborescence/branching.
          - The current method is to map weight -> -weight. This is NOT a good approach since
            the algorithm can and does choose to ignore negative weights when creating a branching
            since that is always optimal when maximzing the weights. I think we should set the edge
            weights to be (max_weight + 1) - edge_weight.
        - Transform the graph into a MultiDiGraph, adding the partition information and potoentially
          other edge attributes if we set preserve_attrs = True.
        - Setup the buckets and union find data structures required for the algorithm.
        """
        pass

    def find_optimum(self, attr='weight', default=1, kind='max', style='branching', preserve_attrs=False, partition=None, seed=None):
        """
        Returns a branching from G.

        Parameters
        ----------
        attr : str
            The edge attribute used to in determining optimality.
        default : float
            The value of the edge attribute used if an edge does not have
            the attribute `attr`.
        kind : {'min', 'max'}
            The type of optimum to search for, either 'min' or 'max'.
        style : {'branching', 'arborescence'}
            If 'branching', then an optimal branching is found. If `style` is
            'arborescence', then a branching is found, such that if the
            branching is also an arborescence, then the branching is an
            optimal spanning arborescences. A given graph G need not have
            an optimal spanning arborescence.
        preserve_attrs : bool
            If True, preserve the other edge attributes of the original
            graph (that are not the one passed to `attr`)
        partition : str
            The edge attribute holding edge partition data. Used in the
            spanning arborescence iterator.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        H : (multi)digraph
            The branching.

        """
        pass

@nx._dispatchable(preserve_edge_attrs=True, mutates_input=True, returns_graph=True)
def minimal_branching(G, /, *, attr='weight', default=1, preserve_attrs=False, partition=None):
    """
    Returns a minimal branching from `G`.

    A minimal branching is a branching similar to a minimal arborescence but
    without the requirement that the result is actually a spanning arborescence.
    This allows minimal branchinges to be computed over graphs which may not
    have arborescence (such as multiple components).

    Parameters
    ----------
    G : (multi)digraph-like
        The graph to be searched.
    attr : str
        The edge attribute used in determining optimality.
    default : float
        The value of the edge attribute used if an edge does not have
        the attribute `attr`.
    preserve_attrs : bool
        If True, preserve the other attributes of the original graph (that are not
        passed to `attr`)
    partition : str
        The key for the edge attribute containing the partition
        data on the graph. Edges can be included, excluded or open using the
        `EdgePartition` enum.

    Returns
    -------
    B : (multi)digraph-like
        A minimal branching.
    """
    pass
docstring_branching = '\nReturns a {kind} {style} from G.\n\nParameters\n----------\nG : (multi)digraph-like\n    The graph to be searched.\nattr : str\n    The edge attribute used to in determining optimality.\ndefault : float\n    The value of the edge attribute used if an edge does not have\n    the attribute `attr`.\npreserve_attrs : bool\n    If True, preserve the other attributes of the original graph (that are not\n    passed to `attr`)\npartition : str\n    The key for the edge attribute containing the partition\n    data on the graph. Edges can be included, excluded or open using the\n    `EdgePartition` enum.\n\nReturns\n-------\nB : (multi)digraph-like\n    A {kind} {style}.\n'
docstring_arborescence = docstring_branching + '\nRaises\n------\nNetworkXException\n    If the graph does not contain a {kind} {style}.\n\n'
maximum_branching.__doc__ = docstring_branching.format(kind='maximum', style='branching')
minimum_branching.__doc__ = docstring_branching.format(kind='minimum', style='branching') + '\nSee Also\n--------\n    minimal_branching\n'
maximum_spanning_arborescence.__doc__ = docstring_arborescence.format(kind='maximum', style='spanning arborescence')
minimum_spanning_arborescence.__doc__ = docstring_arborescence.format(kind='minimum', style='spanning arborescence')

class ArborescenceIterator:
    """
    Iterate over all spanning arborescences of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges). It generates minimum spanning
    arborescences using a modified Edmonds' Algorithm which respects the
    partition of edges. For arborescences with the same weight, ties are
    broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning arborescence of the
        partition dict.
        """
        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            return ArborescenceIterator.Partition(self.mst_weight, self.partition_dict.copy())

    def __init__(self, G, weight='weight', minimum=True, init_partition=None):
        """
        Initialize the iterator

        Parameters
        ----------
        G : nx.DiGraph
            The directed graph which we need to iterate trees over

        weight : String, default = "weight"
            The edge attribute used to store the weight of the edge

        minimum : bool, default = True
            Return the trees in increasing order while true and decreasing order
            while false.

        init_partition : tuple, default = None
            In the case that certain edges have to be included or excluded from
            the arborescences, `init_partition` should be in the form
            `(included_edges, excluded_edges)` where each edges is a
            `(u, v)`-tuple inside an iterable such as a list or set.

        """
        self.G = G.copy()
        self.weight = weight
        self.minimum = minimum
        self.method = minimum_spanning_arborescence if minimum else maximum_spanning_arborescence
        self.partition_key = 'ArborescenceIterators super secret partition attribute name'
        if init_partition is not None:
            partition_dict = {}
            for e in init_partition[0]:
                partition_dict[e] = nx.EdgePartition.INCLUDED
            for e in init_partition[1]:
                partition_dict[e] = nx.EdgePartition.EXCLUDED
            self.init_partition = ArborescenceIterator.Partition(0, partition_dict)
        else:
            self.init_partition = None

    def __iter__(self):
        """
        Returns
        -------
        ArborescenceIterator
            The iterator object for this graph
        """
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)
        if self.init_partition is not None:
            self._write_partition(self.init_partition)
        mst_weight = self.method(self.G, self.weight, partition=self.partition_key, preserve_attrs=True).size(weight=self.weight)
        self.partition_queue.put(self.Partition(mst_weight if self.minimum else -mst_weight, {} if self.init_partition is None else self.init_partition.partition_dict))
        return self

    def __next__(self):
        """
        Returns
        -------
        (multi)Graph
            The spanning tree of next greatest weight, which ties broken
            arbitrarily.
        """
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration
        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_arborescence = self.method(self.G, self.weight, partition=self.partition_key, preserve_attrs=True)
        self._partition(partition, next_arborescence)
        self._clear_partition(next_arborescence)
        return next_arborescence

    def _partition(self, partition, partition_arborescence):
        """
        Create new partitions based of the minimum spanning tree of the
        current minimum partition.

        Parameters
        ----------
        partition : Partition
            The Partition instance used to generate the current minimum spanning
            tree.
        partition_arborescence : nx.Graph
            The minimum spanning arborescence of the input partition.
        """
        pass

    def _write_partition(self, partition):
        """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree. Also, if one incoming edge is included, mark all others
        as excluded so that if that vertex is merged during Edmonds' algorithm
        we cannot still pick another of that vertex's included edges.

        Parameters
        ----------
        partition : Partition
            A Partition dataclass describing a partition on the edges of the
            graph.
        """
        pass

    def _clear_partition(self, G):
        """
        Removes partition data from the graph
        """
        pass