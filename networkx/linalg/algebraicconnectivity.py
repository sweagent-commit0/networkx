"""
Algebraic connectivity and Fiedler vectors of undirected graphs.
"""
from functools import partial
import networkx as nx
from networkx.utils import not_implemented_for, np_random_state, reverse_cuthill_mckee_ordering
__all__ = ['algebraic_connectivity', 'fiedler_vector', 'spectral_ordering', 'spectral_bisection']

class _PCGSolver:
    """Preconditioned conjugate gradient method.

    To solve Ax = b:
        M = A.diagonal() # or some other preconditioner
        solver = _PCGSolver(lambda x: A * x, lambda x: M * x)
        x = solver.solve(b)

    The inputs A and M are functions which compute
    matrix multiplication on the argument.
    A - multiply by the matrix A in Ax=b
    M - multiply by M, the preconditioner surrogate for A

    Warning: There is no limit on number of iterations.
    """

    def __init__(self, A, M):
        self._A = A
        self._M = M

class _LUSolver:
    """LU factorization.

    To solve Ax = b:
        solver = _LUSolver(A)
        x = solver.solve(b)

    optional argument `tol` on solve method is ignored but included
    to match _PCGsolver API.
    """

    def __init__(self, A):
        import scipy as sp
        self._LU = sp.sparse.linalg.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'Equil': True, 'SymmetricMode': True})

def _preprocess_graph(G, weight):
    """Compute edge weights and eliminate zero-weight edges."""
    pass

def _rcm_estimate(G, nodelist):
    """Estimate the Fiedler vector using the reverse Cuthill-McKee ordering."""
    pass

def _tracemin_fiedler(L, X, normalized, tol, method):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph. This function starts with the Laplacian L, not the Graph.

    Parameters
    ----------
    L : Laplacian of a possibly weighted or normalized, but undirected graph

    X : Initial guess for a solution. Usually a matrix of random numbers.
        This function allows more than one column in X to identify more than
        one eigenvector if desired.

    normalized : bool
        Whether the normalized Laplacian matrix is used.

    tol : float
        Tolerance of relative residual in eigenvalue computation.
        Warning: There is no limit on number of iterations.

    method : string
        Should be 'tracemin_pcg' or 'tracemin_lu'.
        Otherwise exception is raised.

    Returns
    -------
    sigma, X : Two NumPy arrays of floats.
        The lowest eigenvalues and corresponding eigenvectors of L.
        The size of input X determines the size of these outputs.
        As this is for Fiedler vectors, the zero eigenvalue (and
        constant eigenvector) are avoided.
    """
    pass

def _get_fiedler_func(method):
    """Returns a function that solves the Fiedler eigenvalue problem."""
    pass

@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatchable(edge_attrs='weight')
def algebraic_connectivity(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Returns the algebraic connectivity of an undirected graph.

    The algebraic connectivity of a connected undirected graph is the second
    smallest eigenvalue of its Laplacian matrix.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    algebraic_connectivity : float
        Algebraic connectivity.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    For undirected graphs algebraic connectivity can tell us if a graph is connected or not
    `G` is connected iff  ``algebraic_connectivity(G) > 0``:

    >>> G = nx.complete_graph(5)
    >>> nx.algebraic_connectivity(G) > 0
    True
    >>> G.add_node(10)  # G is no longer connected
    >>> nx.algebraic_connectivity(G) > 0
    False

    """
    pass

@not_implemented_for('directed')
@np_random_state(5)
@nx._dispatchable(edge_attrs='weight')
def fiedler_vector(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Returns the Fiedler vector of a connected undirected graph.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    fiedler_vector : NumPy array of floats.
        Fiedler vector.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes or is not connected.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    Given a connected graph the signs of the values in the Fiedler vector can be
    used to partition the graph into two components.

    >>> G = nx.barbell_graph(5, 0)
    >>> nx.fiedler_vector(G, normalized=True, seed=1)
    array([-0.32864129, -0.32864129, -0.32864129, -0.32864129, -0.26072899,
            0.26072899,  0.32864129,  0.32864129,  0.32864129,  0.32864129])

    The connected components are the two 5-node cliques of the barbell graph.
    """
    pass

@np_random_state(5)
@nx._dispatchable(edge_attrs='weight')
def spectral_ordering(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Compute the spectral_ordering of a graph.

    The spectral ordering of a graph is an ordering of its nodes where nodes
    in the same weakly connected components appear contiguous and ordered by
    their corresponding elements in the Fiedler vector of the component.

    Parameters
    ----------
    G : NetworkX graph
        A graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    spectral_ordering : NumPy array of floats.
        Spectral ordering of nodes.

    Raises
    ------
    NetworkXError
        If G is empty.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix
    """
    pass

@nx._dispatchable(edge_attrs='weight')
def spectral_bisection(G, weight='weight', normalized=False, tol=1e-08, method='tracemin_pcg', seed=None):
    """Bisect the graph using the Fiedler vector.

    This method uses the Fiedler vector to bisect a graph.
    The partition is defined by the nodes which are associated with
    either positive or negative values in the vector.

    Parameters
    ----------
    G : NetworkX Graph

    weight : str, optional (default: weight)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    bisection : tuple of sets
        Sets with the bisection of nodes

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.spectral_bisection(G)
    ({0, 1, 2}, {3, 4, 5})

    References
    ----------
    .. [1] M. E. J Newman 'Networks: An Introduction', pages 364-370
       Oxford University Press 2011.
    """
    pass