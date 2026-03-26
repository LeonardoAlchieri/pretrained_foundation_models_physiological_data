from logging import getLogger
import numpy as np

from numpy.ma import masked_invalid


logger = getLogger(__name__)


class NoneFeatureExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(self, channel: int | None = None, preserve_shape: bool = False):
        """
        Initializes the NoneFeatureExtractor.

        Parameters
        ----------
        channel : int | None, optional
            The channel to extract features from. If None, all channels are used.
        """
        self.channel = channel
        self.preserve_shape = preserve_shape

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
        }

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """
        Extracts features from an input EDA array.

        Parameters
        ----------
        arr : np.ndarray
            Input EDA array.

        Returns
        -------
        np.ndarray
            Extracted features.
        """

        logger.info("Extracting handcrafted features from EDA signals.")
        features = arr
        features = masked_invalid(features, copy=False)
        if self.channel is not None:
            features = features[..., self.channel]
        if not self.preserve_shape:
            features = features.reshape(features.shape[0], -1)
        return features
