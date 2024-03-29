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
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from .svd import selectSVD
from ..utils import is_symmetric


def _get_centering_matrix(n):
    """
    Compute the centering array

    Parameters
    ----------
    n : int
        Size of centering array.

    Returns
    -------
    out : 2d-array
        Outputs a centering array of shape (n, n)
    """
    out = np.identity(n) - (1 / n) * np.ones((n, n))

    return out


class ClassicalMDS(BaseEstimator):
    """
    Classical multidimensional scaling (cMDS).

    cMDS  seeks a low-dimensional representation of the data in
    which the distances respect well the distances in the original
    high-dimensional space.

    Parameters
    ----------
    n_components : int, or None
        Number of components to keep. If None, then it will run
        ``select_dimension`` to find the optimal embedding dimension.

    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    Attributes
    ----------
    n_components : int
        Equals the parameter n_components. If input n_components was None,
        then equals the optimal embedding dimension.

    components_ : array, shape (n_components, n_features)
        Principal axes in feature space.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        
    dissimilarity_matrix_ : array, shape (n_features, n_features)
        Dissimilarity matrix 

    References
    ----------
    Wickelmaier, Florian. "An introduction to MDS." Sound Quality Research Unit, 
    Aalborg University, Denmark 46.5 (2003).
    """

    def __init__(self, n_components=None, dissimilarity="euclidean"):
        # Check inputs
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must be an integer, not {}.".format(
                    type(n_components)
                )
                raise TypeError(msg)
            elif n_components <= 0:
                msg = "n_components must be >= 1 or None."
                raise ValueError(msg)
        self.n_components = n_components

        if dissimilarity not in ["euclidean", "precomputed"]:
            msg = "Dissimilarity measure must be either 'euclidean' or 'precomputed'."
            raise ValueError(msg)
        self.dissimilarity = dissimilarity

    def _compute_euclidean_distances(self, X):
        """
        Computes pairwise distance between row vectors or matrices

        Parameters
        ----------
        X : array_like
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be 2d-array 
            with shape (n_samples, n_features) or a 3d-array with shape 
            (n_samples, n_features_1, n_features_2).
        
        Returns
        -------
        out : 2d-array, shape (n_samples, n_samples)
            A dissimilarity matrix based on Frobenous norms between pairs of
            matrices or vectors.
        """
        shape = X.shape
        n_samples = shape[0]

        if X.ndim == 2:
            order = 2
            axis = 1
        else:
            order = "fro"
            axis = (1, 2)

        out = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            out[i] = np.linalg.norm(X - X[i], axis=axis, ord=order)

        return out

    def fit(self, X, y=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : array_like
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be 2d-array 
            with shape (n_samples, n_features) or a 3d-array with shape 
            (n_samples, n_features_1, n_features_2).
        
        y : Ignored
        """
        # Check X type
        if not isinstance(X, np.ndarray):
            msg = "X must be a numpy array, not {}.".format(type(X))
            raise ValueError(msg)

        if self.n_components is not None:
            n_samples = X.shape[0]
            if self.n_components > n_samples:
                msg = "n_components must be <= n_samples."
                raise ValueError(msg)

        # Handle dissimilarity
        if self.dissimilarity == "precomputed":
            dissimilarity_matrix = check_array(X, ensure_2d=True, allow_nd=False)

            # Must be symmetric
            if not is_symmetric(dissimilarity_matrix):
                msg = "X must be a symmetric array if precomputed dissimilarity matrix."
                raise ValueError(msg)
        elif self.dissimilarity == "euclidean":
            X = check_array(X, ensure_2d=True, allow_nd=True)
            dissimilarity_matrix = self._compute_euclidean_distances(X=X)

        J = _get_centering_matrix(dissimilarity_matrix.shape[0])
        B = J @ (dissimilarity_matrix ** 2) @ J * -0.5

        n_components = self.n_components

        if n_components == 1:
            algorithm = "full"
        else:
            algorithm = "randomized"
        U, D, V = selectSVD(B, algorithm=algorithm, n_components=n_components)

        self.n_components_ = len(D)
        self.components_ = U
        self.singular_values_ = D ** 0.5
        self.dissimilarity_matrix_ = dissimilarity_matrix

        return self

    def fit_transform(self, X, y=None):
        """
        Fit the data from X, and returns the embedded coordinates.

        Parameters
        ----------
        X : nd-array
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be array 
            with shape (n_samples, n_features) or a nd-array with shape 
            (n_samples, n_features_1, n_features_2, ..., n_features_d). First
            axis of nd-array must be ``n_samples``.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Embedded input.
        """
        self.fit(X)

        X_new = self.components_ @ np.diag(self.singular_values_)

        return X_new
    def transform(self, X, y=None):
        """
        Fit the data from X, and returns the embedded coordinates.

        Parameters
        ----------
        X : nd-array
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be array 
            with shape (n_samples, n_features) or a nd-array with shape 
            (n_samples, n_features_1, n_features_2, ..., n_features_d). First
            axis of nd-array must be ``n_samples``.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Embedded input.
        """
        try:
                self.components_
        except ValueError:
                return RuntimeError('You should fit it first')
        else:
                self.fit(X)
                X_new = self.components_ @ np.diag(self.singular_values_)

                return X_new
