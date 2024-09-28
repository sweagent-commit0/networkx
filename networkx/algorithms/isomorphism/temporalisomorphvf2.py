"""
*****************************
Time-respecting VF2 Algorithm
*****************************

An extension of the VF2 algorithm for time-respecting graph isomorphism
testing in temporal graphs.

A temporal graph is one in which edges contain a datetime attribute,
denoting when interaction occurred between the incident nodes. A
time-respecting subgraph of a temporal graph is a subgraph such that
all interactions incident to a node occurred within a time threshold,
delta, of each other. A directed time-respecting subgraph has the
added constraint that incoming interactions to a node must precede
outgoing interactions from the same node - this enforces a sense of
directed flow.

Introduction
------------

The TimeRespectingGraphMatcher and TimeRespectingDiGraphMatcher
extend the GraphMatcher and DiGraphMatcher classes, respectively,
to include temporal constraints on matches. This is achieved through
a semantic check, via the semantic_feasibility() function.

As well as including G1 (the graph in which to seek embeddings) and
G2 (the subgraph structure of interest), the name of the temporal
attribute on the edges and the time threshold, delta, must be supplied
as arguments to the matching constructors.

A delta of zero is the strictest temporal constraint on the match -
only embeddings in which all interactions occur at the same time will
be returned. A delta of one day will allow embeddings in which
adjacent interactions occur up to a day apart.

Examples
--------

Examples will be provided when the datetime type has been incorporated.


Temporal Subgraph Isomorphism
-----------------------------

A brief discussion of the somewhat diverse current literature will be
included here.

References
----------

[1] Redmond, U. and Cunningham, P. Temporal subgraph isomorphism. In:
The 2013 IEEE/ACM International Conference on Advances in Social
Networks Analysis and Mining (ASONAM). Niagara Falls, Canada; 2013:
pages 1451 - 1452. [65]

For a discussion of the literature on temporal networks:

[3] P. Holme and J. Saramaki. Temporal networks. Physics Reports,
519(3):97–125, 2012.

Notes
-----

Handles directed and undirected graphs and graphs with parallel edges.

"""
import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
__all__ = ['TimeRespectingGraphMatcher', 'TimeRespectingDiGraphMatcher']

class TimeRespectingGraphMatcher(GraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a TimeRespectingGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> G2 = nx.Graph(nx.path_graph(4, create_using=nx.Graph()))

        >>> GM = isomorphism.TimeRespectingGraphMatcher(G1, G2, "date", timedelta(days=1))
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def one_hop(self, Gx, Gx_node, neighbors):
        """
        Edges one hop out from a node in the mapping should be
        time-respecting with respect to each other.
        """
        pass

    def two_hop(self, Gx, core_x, Gx_node, neighbors):
        """
        Paths of length 2 from Gx_node should be time-respecting.
        """
        pass

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        pass

class TimeRespectingDiGraphMatcher(DiGraphMatcher):

    def __init__(self, G1, G2, temporal_attribute_name, delta):
        """Initialize TimeRespectingDiGraphMatcher.

        G1 and G2 should be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        To create a TimeRespectingDiGraphMatcher which checks for
        syntactic and semantic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> from datetime import timedelta
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))

        >>> GM = isomorphism.TimeRespectingDiGraphMatcher(G1, G2, "date", timedelta(days=1))
        """
        self.temporal_attribute_name = temporal_attribute_name
        self.delta = delta
        super().__init__(G1, G2)

    def get_pred_dates(self, Gx, Gx_node, core_x, pred):
        """
        Get the dates of edges from predecessors.
        """
        pass

    def get_succ_dates(self, Gx, Gx_node, core_x, succ):
        """
        Get the dates of edges to successors.
        """
        pass

    def one_hop(self, Gx, Gx_node, core_x, pred, succ):
        """
        The ego node.
        """
        pass

    def two_hop_pred(self, Gx, Gx_node, core_x, pred):
        """
        The predecessors of the ego node.
        """
        pass

    def two_hop_succ(self, Gx, Gx_node, core_x, succ):
        """
        The successors of the ego node.
        """
        pass

    def test_one(self, pred_dates, succ_dates):
        """
        Edges one hop out from Gx_node in the mapping should be
        time-respecting with respect to each other, regardless of
        direction.
        """
        pass

    def test_two(self, pred_dates, succ_dates):
        """
        Edges from a dual Gx_node in the mapping should be ordered in
        a time-respecting manner.
        """
        pass

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
        pass