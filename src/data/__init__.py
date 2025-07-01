import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.typing import DataInfo
import pprint
from logging import getLogger

logger = getLogger(__name__)


class EDADataset:
    """
    A class representing the USI Laughs dataset.
    """

    def __init__(
        self,
        path_to_data: str,
        validation_method: object,
        feature_extractor: object,
    ):
        """
        Initialize the USI Laughs dataset.

        :param path_to_data: Path to the dataset.
        """
        self.path_to_data = path_to_data
        self.data = self._load_data(path_to_data)
        self.validation_method = validation_method
        self.extracted_features: bool = False
        self.feature_extractor = feature_extractor
        self.cache_path = self._get_cache_path()

    def _get_cache_path(self) -> str:
        """
        Get the cache path based on the data file and feature extractor hash.
        """
        os.makedirs(Path(self.path_to_data).parent / ".cache", exist_ok=True)
        
        # Create a hash of the feature_extractor
        feature_extractor_dict = self.feature_extractor.to_dict()
        print("Feature extractor dict:")
        # Print the feature extractor dict in light blue
        print("\033[94m")  # Light blue ANSI escape code
        pprint.pprint(feature_extractor_dict, indent=2, width=80, compact=False)
        print("\033[0m")   # Reset color

        feature_extractor_str = str(feature_extractor_dict)
        feature_hash = hashlib.md5(feature_extractor_str.encode()).hexdigest()
        print(f"\033[94mFeature extractor hash: {feature_hash}\033[0m")
        
        
        # Combine data file stem with feature extractor hash
        cache_filename = f"{str(Path(self.path_to_data).stem)}_{feature_hash}.npy"
        
        return str(
            Path(self.path_to_data).parent / ".cache" / cache_filename
        )

    def _load_data(self, path: str) -> DataInfo:
        """
        Load the USI Laughs dataset.
        This method should be implemented to load the actual dataset.
        """
        loaded_data = np.load(path, allow_pickle=True)
        return dict(loaded_data)

    def _check_and_load_from_cache(self):
        if (Path(self.cache_path).exists()) and (not self.extracted_features):
            logger.info(f"Loading cached features from {self.cache_path}")
            self.data: dict[str, np.ndarray] = np.load(
                self.cache_path, allow_pickle=True
            ).item()
            self.extracted_features = True
            
            return True
        else:
            logger.info(f"No cached features found at {self.cache_path}. Computing...")
            return False

    def extract_features(self, inplace: bool = False):
        """
        Extract features from the dataset using the provided feature extractor.
        """
        if not self._check_and_load_from_cache():
            if not self.extracted_features:
                self.data = self.feature_extractor(self.data)
                np.save(
                    self.cache_path,
                    self.data,
                )
                self.extracted_features = True

        if not inplace:
            return self

    def train_test_split(self, inplace: bool = False):
        if not self.extracted_features:
            raise RuntimeError(
                "Features must be extracted before splitting the dataset."
            )

        self.train_data_folds, self.test_data_folds = self.validation_method(self.data)

        if not inplace:
            return self
