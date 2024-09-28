import networkx as nx

class InverseLaplacian:

    def __init__(self, L, width=None, dtype=None):
        global np
        import numpy as np
        n, n = L.shape
        self.dtype = dtype
        self.n = n
        if width is None:
            self.w = self.width(L)
        else:
            self.w = width
        self.C = np.zeros((self.w, n), dtype=dtype)
        self.L1 = L[1:, 1:]
        self.init_solver(L)

class FullInverseLaplacian(InverseLaplacian):
    pass

class SuperLUInverseLaplacian(InverseLaplacian):
    pass

class CGInverseLaplacian(InverseLaplacian):
    pass