# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import import_graph, is_almost_symmetric
from .base_mase import BaseEmbedMulti
from .svd import select_dimension, selectSVD


class MultipleASE(BaseEmbedMulti):
    r"""
    Multiple Adjacency Spectral Embedding (MASE) embeds arbitrary number of input 
    graphs with matched vertex sets.

    For a population of undirected graphs, MASE assumes that the population of graphs 
    is sampled from :math:`VR^{(i)}V^T` where :math:`V \in \mathbb{R}^{n\times d}` and 
    :math:`R^{(i)} \in \mathbb{R}^{d\times d}`. Score matrices, :math:`R^{(i)}`, are 
    allowed to vary for each graph, but are symmetric. All graphs share a common a 
    latent position matrix :math:`V`. 
    
    For a population of directed graphs, MASE assumes that the population is sampled
    from :math:`UR^{(i)}V^T` where :math:`U \in \mathbb{R}^{n\times d_1}`, 
    :math:`V \in \mathbb{R}^{n\times d_2}`, and 
    :math:`R^{(i)} \in \mathbb{R}^{d_1\times d_2}`. In this case, score matrices 
    :math:`R^{(i)}` can be assymetric and non-square, but all graphs still share a 
    common latent position matrices :math:`U` and :math:`V`.

    Parameters
    ----------
    n_components : int or None, (default=None)
        Desired dimensionality of output data. If algorithm=="full", 
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        `select_dimension`.
    n_elbows : int, optional (default=2)
        If `n_compoents=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.
    algorithm : {'full', 'truncated', 'randomized' (default)}, optional
        SVD solver to use:

        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
        - 'randomized'
            Computes randomized svd using 
            ``sklearn.utils.extmath.randomized_svd``
    n_iter : int, optional (default=5)
        Number of iterations for randomized SVD solver. Not used by 'full' or 
        'truncated'. The default is larger than the default in randomized_svd 
        to handle sparse matrices that may have large slowly decaying spectrum.
    scaled : bool, optional (default=False)
        Whether to scale individual eigenvectors with eigenvalues in first embedding 
        stage.

    Attributes
    ----------
    n_graphs_ : int
        Number of graphs
    n_vertices_ : int
        Number of vertices in each graph
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph. 
    latent_right_ : array, shape (n_samples, n_components), or None
        Estimated right latent positions of the graph. Only computed when the an input 
        graph is directed, or adjacency matrix is assymetric. Otherwise, None.
    scores_ : array, shape (n_samples, n_components, n_components)
        Estimated :math:`\hat{R}` matrices for each input graph.

    Notes
    -----
    When an input graph is directed, `n_components` of `latent_left_` may not be equal
    to `n_components` of `latent_right_`.
    """

    def __init__(
        self,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        scaled=False,
    ):
        if not isinstance(scaled, bool):
            msg = "scaled must be a boolean, not {}".format(scaled)
            raise TypeError(msg)

        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
        )
        self.scaled = scaled

    def _reduce_dim(self, graphs):
        # first embed into log2(n_vertices) for each graph
        n_components = int(np.ceil(np.log2(np.min(self.n_vertices_))))

        # embed individual graphs
        embeddings = [
            selectSVD(
                graph,
                n_components=n_components,
                algorithm=self.algorithm,
                n_iter=self.n_iter,
            )
            for graph in graphs
        ]
        Us, Ds, Vs = zip(*embeddings)

        # Choose the best embedding dimension for each graphs
        if self.n_components is None:
            embedding_dimensions = []
            for D in Ds:
                elbows, _ = select_dimension(D, n_elbows=self.n_elbows)
                embedding_dimensions.append(elbows[-1])

            # Choose the max of all of best embedding dimension of all graphs
            best_dimension = int(np.ceil(np.max(embedding_dimensions)))
        else:
            best_dimension = self.n_components

        if not self.scaled:
            Us = np.hstack([U[:, :best_dimension] for U in Us])
            Vs = np.hstack([V.T[:, :best_dimension] for V in Vs])
        else:
            # Equivalent to ASE
            Us = np.hstack(
                [
                    U[:, :best_dimension] @ np.diag(np.sqrt(D[:best_dimension]))
                    for U, D in zip(Us, Ds)
                ]
            )
            Vs = np.hstack(
                [
                    V.T[:, :best_dimension] @ np.diag(np.sqrt(D[:best_dimension]))
                    for V, D in zip(Vs, Ds)
                ]
            )

        # Second SVD for vertices
        # The notation is slightly different than the paper
        Uhat, _, _ = selectSVD(
            Us,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )

        Vhat, _, _ = selectSVD(
            Vs,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
        )
        return Uhat, Vhat

    def fit(self, graphs, y=None):
        """
        Fit the model with graphs.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
        
        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        graphs = self._check_input_graphs(graphs)

        # Check if undirected
        undirected = all(is_almost_symmetric(g) for g in graphs)
        self.undirected = undirected
        # embed
        Uhat, Vhat = self._reduce_dim(graphs)
        self.latent_left_ = Uhat
        if not undirected:
            self.latent_right_ = Vhat
            self.scores_ = Uhat.T @ graphs @ Vhat
        else:
            self.latent_right_ = None
            self.scores_ = Uhat.T @ graphs @ Uhat

        return self

    def fit_transform(self, graphs, y=None):
        """
        Fit the model with graphs and apply the embedding on graphs. 
        n_components is either automatically determined or based on user input.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        y : Ignored

        Returns
        -------
        out : array-like, shape (n_graphs, n_vertices, n_components) if input 
            graphs were symmetric. If graphs were directed, returns tuple of 
            two arrays (same shape as above) where the first corresponds to the
            left latent positions, and the right to the right latent positions
        """
        return self._fit_transform(graphs)

    def transform(self, graphs):
        """
        Fit the model with graphs if the graphs is not fitted and apply the embedding on graphs. 
        n_components is either automatically determined or based on user input.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        y : Ignored

        Returns
        -------
        out : array-like, shape (n_graphs, n_vertices, n_components) if input 
            graphs were symmetric. If graphs were directed, returns tuple of 
            two arrays (same shape as above) where the first corresponds to the
            left latent positions, and the right to the right latent positions
        """
        graphs = self._check_input_graphs(graphs)
        # Check if undirected
        undirected = all(is_almost_symmetric(g) for g in graphs)
        if self.undirected != undirected:
            raise TypeError('The input graphs and the fitted graphs are not both undirected or directed.')
        try:
            self.latent_left_
        except NameError:
            raise NameError('You need to fit this MASE first.')
        else:
            if not undirected:
                scores = self.latent_left_.T @ graphs @ self.latent_right_
            else:
                scores = self.latent_left_.T @ graphs @ self.latent_left_
        return self._fit_transform(graphs)

    def get_latent(self):
        """
        Return the score matrices
        Parameters
        Returns
        -------
        out : array-like, shape (n_graphs, n_vertices, n_components)
        """
        if self.latent_right_ is None:
            return self.latent_left_
        else:
            return self.latent_left_, self.latent_right_
