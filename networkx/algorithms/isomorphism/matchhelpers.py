"""Functions which help end users define customize node_match and
edge_match functions to use during isomorphism checks.
"""
import math
import types
from itertools import permutations
__all__ = ['categorical_node_match', 'categorical_edge_match', 'categorical_multiedge_match', 'numerical_node_match', 'numerical_edge_match', 'numerical_multiedge_match', 'generic_node_match', 'generic_edge_match', 'generic_multiedge_match']

def copyfunc(f, name=None):
    """Returns a deepcopy of a function."""
    pass

def allclose(x, y, rtol=1e-05, atol=1e-08):
    """Returns True if x and y are sufficiently close, elementwise.

    Parameters
    ----------
    rtol : float
        The relative error tolerance.
    atol : float
        The absolute error tolerance.

    """
    pass
categorical_doc = '\nReturns a comparison function for a categorical node attribute.\n\nThe value(s) of the attr(s) must be hashable and comparable via the ==\noperator since they are placed into a set([]) object.  If the sets from\nG1 and G2 are the same, then the constructed function returns True.\n\nParameters\n----------\nattr : string | list\n    The categorical node attribute to compare, or a list of categorical\n    node attributes to compare.\ndefault : value | list\n    The default value for the categorical node attribute, or a list of\n    default values for the categorical node attributes.\n\nReturns\n-------\nmatch : function\n    The customized, categorical `node_match` function.\n\nExamples\n--------\n>>> import networkx.algorithms.isomorphism as iso\n>>> nm = iso.categorical_node_match("size", 1)\n>>> nm = iso.categorical_node_match(["color", "size"], ["red", 2])\n\n'
categorical_edge_match = copyfunc(categorical_node_match, 'categorical_edge_match')
categorical_node_match.__doc__ = categorical_doc
categorical_edge_match.__doc__ = categorical_doc.replace('node', 'edge')
tmpdoc = categorical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('categorical_edge_match', 'categorical_multiedge_match')
categorical_multiedge_match.__doc__ = tmpdoc
numerical_doc = '\nReturns a comparison function for a numerical node attribute.\n\nThe value(s) of the attr(s) must be numerical and sortable.  If the\nsorted list of values from G1 and G2 are the same within some\ntolerance, then the constructed function returns True.\n\nParameters\n----------\nattr : string | list\n    The numerical node attribute to compare, or a list of numerical\n    node attributes to compare.\ndefault : value | list\n    The default value for the numerical node attribute, or a list of\n    default values for the numerical node attributes.\nrtol : float\n    The relative error tolerance.\natol : float\n    The absolute error tolerance.\n\nReturns\n-------\nmatch : function\n    The customized, numerical `node_match` function.\n\nExamples\n--------\n>>> import networkx.algorithms.isomorphism as iso\n>>> nm = iso.numerical_node_match("weight", 1.0)\n>>> nm = iso.numerical_node_match(["weight", "linewidth"], [0.25, 0.5])\n\n'
numerical_edge_match = copyfunc(numerical_node_match, 'numerical_edge_match')
numerical_node_match.__doc__ = numerical_doc
numerical_edge_match.__doc__ = numerical_doc.replace('node', 'edge')
tmpdoc = numerical_doc.replace('node', 'edge')
tmpdoc = tmpdoc.replace('numerical_edge_match', 'numerical_multiedge_match')
numerical_multiedge_match.__doc__ = tmpdoc
generic_doc = '\nReturns a comparison function for a generic attribute.\n\nThe value(s) of the attr(s) are compared using the specified\noperators. If all the attributes are equal, then the constructed\nfunction returns True.\n\nParameters\n----------\nattr : string | list\n    The node attribute to compare, or a list of node attributes\n    to compare.\ndefault : value | list\n    The default value for the node attribute, or a list of\n    default values for the node attributes.\nop : callable | list\n    The operator to use when comparing attribute values, or a list\n    of operators to use when comparing values for each attribute.\n\nReturns\n-------\nmatch : function\n    The customized, generic `node_match` function.\n\nExamples\n--------\n>>> from operator import eq\n>>> from math import isclose\n>>> from networkx.algorithms.isomorphism import generic_node_match\n>>> nm = generic_node_match("weight", 1.0, isclose)\n>>> nm = generic_node_match("color", "red", eq)\n>>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])\n\n'
generic_edge_match = copyfunc(generic_node_match, 'generic_edge_match')

def generic_multiedge_match(attr, default, op):
    """Returns a comparison function for a generic attribute.

    The value(s) of the attr(s) are compared using the specified
    operators. If all the attributes are equal, then the constructed
    function returns True. Potentially, the constructed edge_match
    function can be slow since it must verify that no isomorphism
    exists between the multiedges before it returns False.

    Parameters
    ----------
    attr : string | list
        The edge attribute to compare, or a list of node attributes
        to compare.
    default : value | list
        The default value for the edge attribute, or a list of
        default values for the edgeattributes.
    op : callable | list
        The operator to use when comparing attribute values, or a list
        of operators to use when comparing values for each attribute.

    Returns
    -------
    match : function
        The customized, generic `edge_match` function.

    Examples
    --------
    >>> from operator import eq
    >>> from math import isclose
    >>> from networkx.algorithms.isomorphism import generic_node_match
    >>> nm = generic_node_match("weight", 1.0, isclose)
    >>> nm = generic_node_match("color", "red", eq)
    >>> nm = generic_node_match(["weight", "color"], [1.0, "red"], [isclose, eq])

    """
    pass
generic_node_match.__doc__ = generic_doc
generic_edge_match.__doc__ = generic_doc.replace('node', 'edge')