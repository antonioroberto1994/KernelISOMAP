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

# Posso implementare anche ISOMAP con radius che non c'è


def myRBF(X, Y=None, gamma=None):
    X1 = np.array([X])
    Y1 = Y if Y is None else np.array([Y])
    return rbf_kernel(X1, Y1, gamma=gamma)[0]


class KernelIsomap(Isomap):
    def __init__(self, n_neighbors=2, n_components=2, gamma=None, eigen_solver='auto',
                 tol=0, max_iter=None, path_method='auto',
                 neighbors_algorithm='auto', n_jobs=None):
        self.gamma_ = gamma  # da usare in metric parameters per NN e graph
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


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    digits = load_digits(n_class=6)
    X = digits.data
    X_embedded = KernelIsomap(n_jobs=-1).fit_transform(X)
    print(X_embedded.shape)
