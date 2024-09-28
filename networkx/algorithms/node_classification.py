""" This module provides the functions for node classification problem.

The functions in this module are not imported
into the top level `networkx` namespace.
You can access these functions by importing
the `networkx.algorithms.node_classification` modules,
then accessing the functions as attributes of `node_classification`.
For example:

  >>> from networkx.algorithms import node_classification
  >>> G = nx.path_graph(4)
  >>> G.edges()
  EdgeView([(0, 1), (1, 2), (2, 3)])
  >>> G.nodes[0]["label"] = "A"
  >>> G.nodes[3]["label"] = "B"
  >>> node_classification.harmonic_function(G)
  ['A', 'A', 'B', 'B']

References
----------
Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
Semi-supervised learning using gaussian fields and harmonic functions.
In ICML (Vol. 3, pp. 912-919).
"""
import networkx as nx
__all__ = ['harmonic_function', 'local_and_global_consistency']

@nx.utils.not_implemented_for('directed')
@nx._dispatchable(node_attrs='label_name')
def harmonic_function(G, max_iter=30, label_name='label'):
    """Node classification by Harmonic function

    Function for computing Harmonic function algorithm by Zhu et al.

    Parameters
    ----------
    G : NetworkX Graph
    max_iter : int
        maximum number of iterations allowed
    label_name : string
        name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.harmonic_function(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """
    pass

@nx.utils.not_implemented_for('directed')
@nx._dispatchable(node_attrs='label_name')
def local_and_global_consistency(G, alpha=0.99, max_iter=30, label_name='label'):
    """Node classification by Local and Global Consistency

    Function for computing Local and global consistency algorithm by Zhou et al.

    Parameters
    ----------
    G : NetworkX Graph
    alpha : float
        Clamping factor
    max_iter : int
        Maximum number of iterations allowed
    label_name : string
        Name of target labels to predict

    Returns
    -------
    predicted : list
        List of length ``len(G)`` with the predicted labels for each node.

    Raises
    ------
    NetworkXError
        If no nodes in `G` have attribute `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.local_and_global_consistency(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    pass

def _get_label_info(G, label_name):
    """Get and return information of labels from the input graph

    Parameters
    ----------
    G : Network X graph
    label_name : string
        Name of the target label

    Returns
    -------
    labels : numpy array, shape = [n_labeled_samples, 2]
        Array of pairs of labeled node ID and label ID
    label_dict : numpy array, shape = [n_classes]
        Array of labels
        i-th element contains the label corresponding label ID `i`
    """
    pass