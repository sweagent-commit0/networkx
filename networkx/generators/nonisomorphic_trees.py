"""
Implementation of the Wright, Richmond, Odlyzko and McKay (WROM)
algorithm for the enumeration of all non-isomorphic free trees of a
given order.  Rooted trees are represented by level sequences, i.e.,
lists in which the i-th element specifies the distance of vertex i to
the root.

"""
__all__ = ['nonisomorphic_trees', 'number_of_nonisomorphic_trees']
import networkx as nx

@nx._dispatchable(graphs=None, returns_graph=True)
def nonisomorphic_trees(order, create='graph'):
    """Generates lists of nonisomorphic trees

    Parameters
    ----------
    order : int
       order of the desired tree(s)

    create : one of {"graph", "matrix"} (default="graph")
       If ``"graph"`` is selected a list of ``Graph`` instances will be returned,
       if matrix is selected a list of adjacency matrices will be returned.

       .. deprecated:: 3.3

          The `create` argument is deprecated and will be removed in NetworkX
          version 3.5. In the future, `nonisomorphic_trees` will yield graph
          instances by default. To generate adjacency matrices, call
          ``nx.to_numpy_array`` on the output, e.g.::

             [nx.to_numpy_array(G) for G in nx.nonisomorphic_trees(N)]

    Yields
    ------
    list
       A list of nonisomorphic trees, in one of two formats depending on the
       value of the `create` parameter:
       - ``create="graph"``: yields a list of `networkx.Graph` instances
       - ``create="matrix"``: yields a list of list-of-lists representing adjacency matrices
    """
    pass

@nx._dispatchable(graphs=None)
def number_of_nonisomorphic_trees(order):
    """Returns the number of nonisomorphic trees

    Parameters
    ----------
    order : int
      order of the desired tree(s)

    Returns
    -------
    length : Number of nonisomorphic graphs for the given order

    References
    ----------

    """
    pass

def _next_rooted_tree(predecessor, p=None):
    """One iteration of the Beyer-Hedetniemi algorithm."""
    pass

def _next_tree(candidate):
    """One iteration of the Wright, Richmond, Odlyzko and McKay
    algorithm."""
    pass

def _split_tree(layout):
    """Returns a tuple of two layouts, one containing the left
    subtree of the root vertex, and one containing the original tree
    with the left subtree removed."""
    pass

def _layout_to_matrix(layout):
    """Create the adjacency matrix for the tree specified by the
    given layout (level sequence)."""
    pass

def _layout_to_graph(layout):
    """Create a NetworkX Graph for the tree specified by the
    given layout(level sequence)"""
    pass