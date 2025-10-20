from logging import getLogger
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from src.utils.data_preparation import prepare_data_for_sktime

logger = getLogger(__name__)


class MiniRocket(BaseEstimator, TransformerMixin):
    """
    A class to extract MiniRocket features from EDA signals using the MiniRocketMultivariate transformer.
    """

    def __init__(self, random_state: Optional[int] = None, verbose: int = 1):
        """
        Initializes the MiniRocket feature extractor.

        Parameters
        ----------
        random_state : int
            Seed for random number generator.
        """
        self.random_state = random_state
        self.minirocket_ = None  # type: Optional[MiniRocketMultivariate]
        self.n_features_ = None  # type: Optional[int]
        self.verbose = verbose

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Extracts MiniRocket features from the EDA dataset.

        Parameters
        ----------
        data : EDADataset
            The dataset containing EDA signals.

        Returns
        -------
        EDADataset
            The dataset with extracted MiniRocket features.
        """
        if self.verbose:
            logger.info("Extracting MiniRocket features from EDA signals.")
        data_sktime: pd.DataFrame = prepare_data_for_sktime(X)

        self.minirocket_ = MiniRocketMultivariate(random_state=self.random_state)
        self.minirocket_.fit(data_sktime)
        transformed = self.minirocket_.transform(data_sktime)
        features: np.ndarray = np.asarray(transformed)
        self.n_features_ = features.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input data into MiniRocket features.

        Parameters
        ----------
        X : np.ndarray
            Input EDA signals.

        Returns
        -------
        np.ndarray
            Extracted MiniRocket features.
        """

        check_is_fitted(self, "minirocket_")

        data_sktime: pd.DataFrame = prepare_data_for_sktime(X)
        assert self.minirocket_ is not None
        transformed = self.minirocket_.transform(data_sktime)
        features: np.ndarray = np.asarray(transformed)
        return features

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)
