"""Algorithms to calculate reciprocity in a directed graph."""
import networkx as nx
from networkx import NetworkXError
from ..utils import not_implemented_for
__all__ = ['reciprocity', 'overall_reciprocity']

@not_implemented_for('undirected', 'multigraph')
@nx._dispatchable
def reciprocity(G, nodes=None):
    """Compute the reciprocity in a directed graph.

    The reciprocity of a directed graph is defined as the ratio
    of the number of edges pointing in both directions to the total
    number of edges in the graph.
    Formally, $r = |{(u,v) \\in G|(v,u) \\in G}| / |{(u,v) \\in G}|$.

    The reciprocity of a single node u is defined similarly,
    it is the ratio of the number of edges in both directions to
    the total number of edges attached to node u.

    Parameters
    ----------
    G : graph
       A networkx directed graph
    nodes : container of nodes, optional (default=whole graph)
       Compute reciprocity for nodes in this container.

    Returns
    -------
    out : dictionary
       Reciprocity keyed by node label.

    Notes
    -----
    The reciprocity is not defined for isolated nodes.
    In such cases this function will return None.

    """
    pass

def _reciprocity_iter(G, nodes):
    """Return an iterator of (node, reciprocity)."""
    pass

@not_implemented_for('undirected', 'multigraph')
@nx._dispatchable
def overall_reciprocity(G):
    """Compute the reciprocity for the whole graph.

    See the doc of reciprocity for the definition.

    Parameters
    ----------
    G : graph
       A networkx graph

    """
    pass