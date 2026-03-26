import hashlib
import os
from logging import getLogger
from pathlib import Path
from pprint import pprint
from random import sample
from typing import Callable

import numpy as np
from sklearn.base import TransformerMixin

from edamame_downstream.utils.typing import DataInfo

logger = getLogger(__name__)

ACCEPTED_SIGNAL_KWARGS = ["channels"]


class EDAMAMEDataset:
    """
    A class representing the USI Laughs dataset.
    """

    def __init__(
        self,
        path_to_data: str,
        signals: list[str],
        label_name: str,
        validation_method: object,
        feature_extractors: dict[str, object],
        feature_scaling_methods: dict[str, TransformerMixin],
        sample_scaling_methods: dict[str, Callable],
        label_processor: TransformerMixin,
        name: str | None = None,
        recompute_features: bool = False,
        signal_kwargs: dict | None = None,
        debug: bool = False,
    ):
        """
        Initialize the USI Laughs dataset.

        :param path_to_data: Path to the dataset.
        """
        self.name = name
        self.path_to_data = path_to_data
        self.label_name = label_name
        self.label_processor = label_processor
        self.signals = signals
        self.signal_kwargs = self._check_signal_kwargs(signals, signal_kwargs or {})

        self.sample_scaling_methods = sample_scaling_methods
        self.feature_scaling_methods = feature_scaling_methods

        self.data = self._load_data(path_to_data)
        self.validation_method = validation_method
        self.extracted_features: bool = False
        self.feature_extractors = feature_extractors

        if len(feature_extractors) > 0:
            self.cache_paths = self._get_cache_path()
        else:
            logger.info(
                "No feature extractors provided. Please, use set_feature_extractors()."
            )

        self.recompute_features = recompute_features
        self.debug = debug

    @staticmethod
    def _check_signal_kwargs(signals: list[str], signal_kwargs: dict) -> dict:
        for signal in signals:
            if signal not in signal_kwargs.keys():
                raise ValueError(
                    f"Missing kwargs for signal {signal} in signal_kwargs."
                )
            if (
                signal_kwargs[signal] is None
                or set(signal_kwargs[signal].keys()) != set(ACCEPTED_SIGNAL_KWARGS)
            ):
                raise ValueError(
                    f"""Invalid kwargs for signal {signal} in signal_kwargs. 
                    Expected keys: {ACCEPTED_SIGNAL_KWARGS}. 
                    Received: {signal_kwargs[signal].keys() if signal_kwargs[signal] is not None else None}"""
                )
        return signal_kwargs

    def set_feature_extractors(self, feature_extractors: dict[str, object]):
        """
        Set the feature extractor for the dataset.
        """
        if feature_extractors is None or len(feature_extractors) == 0:
            raise ValueError("Feature extractors cannot be None or empty.")
        self.feature_extractors = feature_extractors
        self.cache_paths = self._get_cache_path()

    def _get_cache_path(self) -> dict[str, str]:
        """
        Get the cache path based on the data file and feature extractor hash.
        """
        os.makedirs(Path(self.path_to_data).parent / ".cache", exist_ok=True)

        paths: dict[str, str] = {}
        for signal in self.signals:
            if signal not in self.feature_extractors:
                raise ValueError(
                    f"""Missing feature extractor for signal {signal} in feature_extractors. 
                    Received keys: {self.feature_extractors.keys()}"""
                )
            os.makedirs(Path(self.path_to_data).parent / ".cache" / signal, exist_ok=True)
            feature_extractor_dict = self.feature_extractors[signal].to_dict()
            # TODO: this is ugly, we need to find a way to fix it
            label_processor_dict = dict(getattr(self.label_processor, "__dict__", {}))
            label_processor_dict["name"] = self.label_processor.__class__.__name__
            print("Feature extractor dict:")
            # Print the feature extractor dict in light blue
            print("\033[94m")  # Light blue ANSI escape code
            pprint(feature_extractor_dict, indent=2, width=80, compact=False)
            print("\033[0m")  # Reset color

            feature_extractor_str = str(feature_extractor_dict)
            label_processor_str = str(label_processor_dict)

            feature_scaling_method_str = str(vars(self.feature_scaling_methods[signal]))
            sample_scaling_method_str = (
                str(vars(self.sample_scaling_methods[signal]))
                if signal in self.sample_scaling_methods
                and self.sample_scaling_methods[signal] is not None
                else "None"
            )

            feature_hash = hashlib.md5(
                (
                    signal
                    + feature_extractor_str
                    + label_processor_str
                    + self.path_to_data
                    + feature_scaling_method_str
                    + sample_scaling_method_str
                ).encode()
            ).hexdigest()

            print(
                f"\033[94mFeature extractor hash for signal {signal}: {feature_hash}\033[0m"
            )

            # Combine data file stem with feature extractor hash
            cache_filename = f"{str(Path(self.path_to_data).stem)}_{feature_hash}.npy"
            paths[signal] = str(
                Path(self.path_to_data).parent / ".cache" / signal / cache_filename
            )
        return paths

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

    def _prepare_labels(self, data: DataInfo) -> DataInfo:
        if self.label_name not in data:
            raise ValueError(
                f"Label {self.label_name} not found in the data. Available keys: {data.keys()}"
            )

        labels = np.asarray(data[self.label_name])
        if np.issubdtype(labels.dtype, np.number):
            labels = labels.astype(np.float32)

        data["labels"] = self.label_processor.fit_transform(labels.reshape(-1, 1)).reshape(
            -1
        )
        data = self.remove_masked_labels(data)
        data["labels"] = data["labels"].astype(np.int32)
        return data

    def _prepare_signals(self, data: DataInfo) -> DataInfo:
        for signal in self.signals:
            if signal not in data:
                raise ValueError(
                    f"Signal {signal} not found in the data. Available keys: {data.keys()}"
                )
            signal_channels = self.signal_kwargs[signal]["channels"]
            data[signal] = data[signal][..., signal_channels]

        if (
            len(
                {data["labels"].shape[0]}
                | {data[signal].shape[0] for signal in self.signals}
            )
            != 1
        ):
            raise ValueError(
                f"""Signal and labels sample dimensions do not match.\n
                Received shapes: labels={data['labels'].shape}, """
                + ", ".join(f"{signal}={data[signal].shape}" for signal in self.signals)
                + f", label processor: {self.label_processor.__class__.__name__}"
            )

        return data

    def _apply_sample_scaling(self, data: DataInfo) -> DataInfo:
        for signal in self.signals:
            if signal not in self.sample_scaling_methods:
                raise ValueError(
                    f"Missing sample scaling method for signal {signal} in sample_scaling_methods."
                )
            else:
                data[signal] = self.sample_scaling_methods[signal](
                    data[signal], data["groups"]
                )
        return data

    def _load_data(self, path: str) -> DataInfo:
        """
        Load the dataset.
        This method should be implemented to load the actual dataset.
        """
        loaded_data = dict(np.load(path, allow_pickle=True))

        loaded_data = self._prepare_labels(loaded_data)
        loaded_data = self._prepare_signals(loaded_data)

        loaded_data = self._apply_sample_scaling(loaded_data)

        logger.info(
            f"Data shape after loading: "
            + ", ".join(
                f"{signal}={loaded_data[signal].shape}" for signal in self.signals
            )
        )

        return loaded_data

    def _check_and_load_from_cache(self, signal: str):

        cache_path = Path(self.cache_paths[signal])

        if cache_path.exists() and (not self.recompute_features):
            logger.info(f"Loading cached features from {cache_path}")
            self.data: dict[str, np.ndarray] = np.load(
                cache_path, allow_pickle=True
            ).item()
            self.extracted_features = True

            return True
        else:
            logger.info(f"No cached features found at {cache_path}. Computing...")
            return False

    def extract_features(self, inplace: bool = False):
        """
        Extract features from the dataset using the provided feature extractor.
        """
        for signal in self.signals:
            if not self._check_and_load_from_cache(signal):
                self.data[f'features_{signal}'] = self.feature_extractors[signal](
                    self.data[signal]
                )
                self.data[f"features_{signal}"] = self.feature_scaling_methods[
                    signal
                ].fit_transform(self.data[f"features_{signal}"])
                np.save(
                    self.cache_paths[signal],
                    self.data,
                )
            self.extracted_features = True

        self.data["features"] = np.concatenate(
            [self.data[f"features_{signal}"] for signal in self.signals], axis=1
        )

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
                        first_iteration = False
                    self.data[key] = self.data[key][random_indices]
        logger.info(
            f"""
            Reduced dataset size for debugging. Original size: {original_size}, 
            Reduced size: {reduced_size}"""
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
