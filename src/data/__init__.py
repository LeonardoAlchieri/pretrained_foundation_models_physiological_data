import hashlib
import os
from logging import getLogger
from pathlib import Path
from pprint import pprint
from typing import Optional, Callable

import numpy as np
from sklearn.base import TransformerMixin

from src.utils.typing import DataInfo

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
        label_processor: TransformerMixin,
        feature_scaling_method: TransformerMixin,
        sample_scaling_method: Callable | None = None,
        recompute_features: bool = False,
        channels: Optional[list[int] | int] = None,
        debug: bool = False,
    ):
        """
        Initialize the USI Laughs dataset.

        :param path_to_data: Path to the dataset.
        """
        self.path_to_data = path_to_data
        self.label_processor = label_processor

        self.sample_scaling_method = sample_scaling_method
        self.data = self._load_data(path_to_data, channels=list(channels))
        self.validation_method = validation_method
        self.extracted_features: bool = False
        self.feature_extractor = feature_extractor
        self.cache_path = self._get_cache_path()
        self.recompute_features = recompute_features

        self.feature_scaling_method = feature_scaling_method

        self.debug = debug

    def _get_cache_path(self) -> str:
        """
        Get the cache path based on the data file and feature extractor hash.
        """
        os.makedirs(Path(self.path_to_data).parent / ".cache", exist_ok=True)

        # Create a hash of the feature_extractor
        feature_extractor_dict = self.feature_extractor.to_dict()
        # TODO: this is ugly, we need to find a way to fix it
        label_processor_dict = vars(self.label_processor)
        label_processor_dict["name"] = self.label_processor.__class__.__name__
        print("Feature extractor dict:")
        # Print the feature extractor dict in light blue
        print("\033[94m")  # Light blue ANSI escape code
        pprint(feature_extractor_dict, indent=2, width=80, compact=False)
        print("\033[0m")  # Reset color

        feature_extractor_str = str(feature_extractor_dict)
        label_processor_str = str(label_processor_dict)
        feature_hash = hashlib.md5(
            (feature_extractor_str + label_processor_str + self.path_to_data).encode()
        ).hexdigest()
        print(f"\033[94mFeature extractor hash: {feature_hash}\033[0m")

        # Combine data file stem with feature extractor hash
        cache_filename = f"{str(Path(self.path_to_data).stem)}_{feature_hash}.npy"

        return str(Path(self.path_to_data).parent / ".cache" / cache_filename)

    @staticmethod
    def remove_masked_labels(data: DataInfo) -> DataInfo:
        """
        Remove samples with masked labels from the dataset.
        """
        if np.ma.is_masked(data["labels"]):
            mask = ~data["labels"].mask
            for key in data:
                if key == "name":
                    continue
                if isinstance(data[key], np.ndarray):
                    data[key] = np.asarray(data[key][mask])
            # data["labels"] = data["labels"].data[mask]  # convert to regular ndarray
        return data

    def _load_data(self, path: str, channels: Optional[list[int]] = None) -> DataInfo:
        """
        Load the dataset.
        This method should be implemented to load the actual dataset.
        """
        loaded_data = dict(np.load(path, allow_pickle=True))

        loaded_data["labels"] = (
            self.label_processor.fit_transform(loaded_data["labels"].reshape(-1, 1))
            .reshape(-1)
            .astype(int)
        )
        loaded_data = self.remove_masked_labels(loaded_data)

        if loaded_data["labels"].shape[0] != loaded_data["values"].shape[0]:
            # TODO: add more info to the error (e.g. shapes and label_processor name)
            raise ValueError(
                f"""Labels shape got messed up when using the label processor. Are you sure you used the correct one?\n
                Received labels shape: {loaded_data["labels"].shape}, values shape: {loaded_data["values"].shape}, label processor: {self.label_processor.__class__.__name__}"""
            )

        if channels is not None:
            loaded_data["values"] = loaded_data["values"][..., channels]

        if self.sample_scaling_method is not None:
            loaded_data["values"] = self.sample_scaling_method(
                loaded_data["values"], loaded_data["groups"]
            )

        logger.info(f"Data shape after loading: {loaded_data['values'].shape}")

        return loaded_data

    def _check_and_load_from_cache(self):

        if (Path(self.cache_path).exists()) and (not self.recompute_features):
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
            self.data = self.feature_extractor(self.data)
            self.data["features"] = self.feature_scaling_method.fit_transform(
                self.data["features"]
            )
            np.save(
                self.cache_path,
                self.data,
            )
            self.extracted_features = True

        if not inplace:
            return self

    def _reduce_size_for_debugging(self):
        first_iteration = True
        for key in self.data:
            if isinstance(self.data[key], np.ndarray):
                if self.data[key].ndim >= 1:
                    if first_iteration:
                        original_size = self.data[key].shape[0]
                        reduced_size = min(original_size, 300)
                        random_indices = np.random.choice(
                            original_size, size=reduced_size, replace=False
                        )
                    self.data[key] = self.data[key][random_indices]
        logger.info(
            f"Reduced dataset size for debugging. Original size: {original_size}, Reduced size: {reduced_size}"
        )

    def train_test_split(self, inplace: bool = False):
        if not self.extracted_features:
            raise RuntimeError(
                "Features must be extracted before splitting the dataset."
            )
        if self.debug:
            self._reduce_size_for_debugging()

        self.train_data_folds, self.test_data_folds = self.validation_method(self.data)

        if not inplace:
            return self
