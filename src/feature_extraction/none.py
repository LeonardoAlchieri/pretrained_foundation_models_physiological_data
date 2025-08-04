from logging import getLogger

from numpy.ma import masked_invalid

from src.data import EDADataset
from src.utils.typing import DataInfo


logger = getLogger(__name__)

class NoneFeatureExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """
    
        
    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
        }

    def __call__(self, data: DataInfo) -> EDADataset:
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
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data