"""KNN classifier that leverages a trainable feature extractor."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TrainableFeatureKNN(BaseEstimator, ClassifierMixin):
    """Sklearn-compliant KNN classifier with a trainable feature extractor.

    Parameters
    ----------
    trainable_feature_extractor : object
            Object exposing ``fit`` and ``predict`` methods. ``predict`` must return a
            2D array-like of shape (n_samples, n_features).
    n_neighbors : int, default=5
            Number of neighbors to use.
    weights : {'uniform', 'distance'} or callable, default='uniform'
            Weight function used in prediction.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors.
    leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.
    p : int, default=2
            Power parameter for the Minkowski metric.
    metric : str or callable, default='minkowski'
            Distance metric used for the tree.
    metric_params : dict, default=None
            Additional metric parameters.
    n_jobs : int, default=None
            Number of parallel jobs to run for neighbors search.
    """

    def __init__(
        self,
        *,
        trainable_feature_extractor: Any,
        n_neighbors: int = 5,
        weights: str | Callable[[np.ndarray], np.ndarray] = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str | Callable[..., float] = "minkowski",
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.trainable_feature_extractor = trainable_feature_extractor
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainableFeatureKNN":
        """Fit the feature extractor and the KNN classifier."""

        if self.trainable_feature_extractor is None:
            raise ValueError("trainable_feature_extractor must be provided.")

        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=False,
            allow_nd=True,
        )

        self.feature_extractor_ = clone(self.trainable_feature_extractor)
        self.feature_extractor_ = self.feature_extractor_.fit(X_checked)
        features = self.feature_extractor_.transform(X_checked)

        self._knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )

        self._knn.fit(features, y_checked)
        self.classes_ = self._knn.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the provided data."""

        features = self._transform_for_inference(X)
        return self._knn.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the provided data."""

        features = self._transform_for_inference(X)
        if not hasattr(self._knn, "predict_proba"):
            raise AttributeError(
                "Underlying KNN classifier does not support predict_proba."
            )
        return self._knn.predict_proba(features)

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ):
        """Find the K-neighbors of a point."""

        features = self._transform_for_inference(X)
        return self._knn.kneighbors(
            features, n_neighbors=n_neighbors, return_distance=return_distance
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _transform(self, X: np.ndarray) -> np.ndarray:
        features = self.feature_extractor_.predict(X)

        if hasattr(features, "to_numpy"):
            features = features.to_numpy()

        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        return check_array(features, accept_sparse=False, ensure_2d=True)

    def _transform_for_inference(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_extractor_", "_knn"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=False, allow_nd=True)
        return self._transform(X_checked)
