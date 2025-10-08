"""Custom kernel SVC implementation."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.spatial.distance import pdist, cdist, squareform


KernelFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Default linear kernel used when no other kernel is provided."""

    return np.matmul(x, y.T)


class CustomKernelSVC(BaseEstimator, ClassifierMixin):
    """SVC classifier that relies on a user-provided kernel function."""

    def __init__(
        self,
        *,
        gamma_mult: float = 1.0,
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = -1,
        probability: bool = False,
        class_weight: Optional[dict] = None,
        cache_size: float = 200.0,
        shrinking: bool = True,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.gamma_mult = gamma_mult
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.probability = probability
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.shrinking = shrinking
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, ensure_2d=True)
        gram_matrix = self.compute_Ktrtr(X_checked)

        self._svc = SVC(
            C=self.C,
            kernel="precomputed",
            tol=self.tol,
            max_iter=self.max_iter,
            probability=self.probability,
            class_weight=self.class_weight,
            cache_size=self.cache_size,
            shrinking=self.shrinking,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        self._svc.fit(gram_matrix, y_checked, sample_weight=sample_weight)
        self._X_fit_ = X_checked
        self._y_fit_ = y_checked

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_svc", "_X_fit_", "_y_fit_"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        gram_matrix = self.compute_Ktrtr(X_checked)
        return self._svc.predict(gram_matrix)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_svc", "_X_fit_", "_y_fit_"])
        if not self.probability:
            raise AttributeError(
                "Probability estimates are not available when probability=False."
            )

        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        gram_matrix = self.compute_Ktrtr(X_checked)
        return self._svc.predict_proba(gram_matrix)

    def compute_Ktrtr(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        # pairwise distance
        pdistance_trtr = squareform(pdist(X, metric="euclidean"))
        # RBF gamma median estimator
        RBF_gamma = self.gamma_mult * np.median(pdistance_trtr)
        # Kernel
        Ktrtr = np.exp(-((pdistance_trtr) ** 2) / (2 * RBF_gamma**2))
        return Ktrtr
