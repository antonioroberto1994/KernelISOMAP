"""Isomap for manifold learning"""

# Author: Antonio Roberto -- <https://github.com/antonioroberto1994>
# License: GNU General Public License v3.0

import numpy as np
from sklearn.manifold import Isomap
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import rbf_kernel

def myRBF(X, Y=None, gamma=None):
    '''
Support method for RBF distance.
    '''
    X1 = np.array([X])
    Y1 = Y if Y is None else np.array([Y])
    return rbf_kernel(X1, Y1, gamma=gamma)[0]


class KernelIsomap(Isomap):
    """Isomap Embedding
    
    Non-linear dimensionality reduction through Isometric Mapping.
    The algorithm uses a kernelized (RBF - Radial Basis Function) distance measure.

    Parameters
    ----------
    n_neighbors : integer
        number of neighbors to consider for each point.
    n_components : integer
        number of coordinates for the manifold
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    eigen_solver : ['auto'|'arpack'|'dense']
        'auto' : Attempt to choose the most efficient solver
        for the given problem.
        'arpack' : Use Arnoldi decomposition to find the eigenvalues
        and eigenvectors.
        'dense' : Use a direct solver (i.e. LAPACK)
        for the eigenvalue decomposition.
    tol : float
        Convergence tolerance passed to arpack or lobpcg.
        not used if eigen_solver == 'dense'.
    max_iter : integer
        Maximum number of iterations for the arpack solver.
        not used if eigen_solver == 'dense'.
    path_method : string ['auto'|'FW'|'D']
        Method to use in finding shortest path.
        'auto' : attempt to choose the best algorithm automatically.
        'FW' : Floyd-Warshall algorithm.
        'D' : Dijkstra's algorithm.
    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    References
    ----------
    .. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
           framework for nonlinear dimensionality reduction. Science 290 (5500)
    """
    def __init__(self, n_neighbors=2, n_components=2, gamma=None, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=None):
        self.gamma_ = gamma 
        super().__init__(n_neighbors=n_neighbors, n_components=n_components, eigen_solver=eigen_solver,
                         tol=tol, max_iter=max_iter, path_method=path_method,
                         neighbors_algorithm=neighbors_algorithm, n_jobs=n_jobs)

    def _fit_transform(self, X):
        X = check_array(X, accept_sparse='csr')
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      metric=myRBF,
                                      metric_params={'gamma': self.gamma_},
                                      n_jobs=self.n_jobs)

        self.nbrs_.fit(X)
        self.training_data_ = self.nbrs_._fit_X
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,
                                     kernel="precomputed",
                                     eigen_solver=self.eigen_solver,
                                     tol=self.tol, max_iter=self.max_iter,
                                     n_jobs=self.n_jobs)

        kng = self.nbrs_.kneighbors_graph(
            X, mode='distance', n_neighbors=self.n_neighbors)

        self.dist_matrix_ = graph_shortest_path(kng,
                                                method=self.path_method,
                                                directed=False)
        G = self.dist_matrix_ ** 2
        G *= -0.5

        self.embedding_ = self.kernel_pca_.fit_transform(G)