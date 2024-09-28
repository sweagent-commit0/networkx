"""Basic algorithms for breadth-first searching the nodes of a graph."""
import networkx as nx
__all__ = ['bfs_beam_edges']

@nx._dispatchable
def bfs_beam_edges(G, source, value, width=None):
    """Iterates over edges in a beam search.

    The beam search is a generalized breadth-first search in which only
    the "best" *w* neighbors of the current node are enqueued, where *w*
    is the beam width and "best" is an application-specific
    heuristic. In general, a beam search with a small beam width might
    not visit each node in the graph.

    .. note::

       With the default value of ``width=None`` or `width` greater than the
       maximum degree of the graph, this function equates to a slower
       version of `~networkx.algorithms.traversal.breadth_first_search.bfs_edges`.
       All nodes will be visited, though the order of the reported edges may
       vary. In such cases, `value` has no effect - consider using `bfs_edges`
       directly instead.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    value : function
        A function that takes a node of the graph as input and returns a
        real number indicating how "good" it is. A higher value means it
        is more likely to be visited sooner during the search. When
        visiting a new node, only the `width` neighbors with the highest
        `value` are enqueued (in decreasing order of `value`).

    width : int (default = None)
        The beam width for the search. This is the number of neighbors
        (ordered by `value`) to enqueue when visiting each new node.

    Yields
    ------
    edge
        Edges in the beam search starting from `source`, given as a pair
        of nodes.

    Examples
    --------
    To give nodes with, for example, a higher centrality precedence
    during the search, set the `value` function to return the centrality
    value of the node:

    >>> G = nx.karate_club_graph()
    >>> centrality = nx.eigenvector_centrality(G)
    >>> list(nx.bfs_beam_edges(G, source=0, value=centrality.get, width=3))
    [(0, 2), (0, 1), (0, 8), (2, 32), (1, 13), (8, 33)]
    """
    pass