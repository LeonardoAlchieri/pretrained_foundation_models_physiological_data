from logging import getLogger

from numpy.ma import masked_invalid

from edamame_downstream.data import EDADataset
from edamame_downstream.utils.typing import DataInfo


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

    def __call__(self, data: DataInfo) -> DataInfo:
        """
        Extracts features from the EDA dataset.

        Parameters
        ----------
        data : EDADataset
            The dataset containing EDA signals.

        Returns
        -------
        EDADataset
            The dataset with extracted features.
        """

        logger.info("Extracting handcrafted features from EDA signals.")
        features = data["values"]
        features = masked_invalid(features, copy=False)
        if self.channel is not None:
            features = features[..., self.channel]
        if not self.preserve_shape:
            features = features.reshape(features.shape[0], -1)
        data["features"] = features
        data["feature_names"] = None
        return data
