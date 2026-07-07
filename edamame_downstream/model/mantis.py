"""Mantis used as an end-to-end (fine-tunable) classifier.

Unlike :class:`edamame_downstream.feature_extraction.mantis.MantisExtractor`, which
only uses Mantis as a frozen feature extractor, this module exposes Mantis as a
*model* in the sense of ``configs/classification/model/``: a scikit-learn-style
classifier that is fine-tuned on the downstream task via ``MantisTrainer.fit`` and
predicts labels directly with ``MantisTrainer.predict``/``predict_proba``.

This lets Mantis be swept and evaluated by the existing
:class:`edamame_downstream.engine.Engine` (i.e. ``GridSearchCV``) exactly like any
other sklearn classifier defined under ``edamame_downstream.model``.
"""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer

logger = getLogger(__name__)

# Cache the loaded foundation-model weights per ``model_name`` so that the repeated
# ``fit`` calls issued by GridSearchCV (one per fold and per hyper-parameter point)
# do not re-read the checkpoint from disk / re-download it every single time.
_STATE_DICT_CACHE: dict[str, dict] = {}


class MantisClassifier(BaseEstimator, ClassifierMixin):
    """Fine-tunable Mantis classifier with a scikit-learn API.

    The estimator receives the (flattened) raw signal as produced by
    :class:`~edamame_downstream.feature_extraction.none.NoneFeatureExtractor`
    (``preserve_shape=False``), i.e. a 2D array of shape ``(n_samples, seq_len * n_channels)``,
    reshapes it back to the ``(n_samples, n_channels, seq_len)`` layout expected by
    Mantis, resamples the time axis to the network's ``seq_len`` and fine-tunes the
    foundation model on the downstream labels.

    Parameters
    ----------
    model_name : str
        Either a HuggingFace hub identifier (e.g. ``"paris-noah/Mantis-8M"``) or a
        local path to a checkpoint file. Used to initialise the pretrained weights.
    n_channels : int, default=3
        Number of channels contained in the flattened input. Required to invert the
        row-major flattening performed by ``NoneFeatureExtractor`` (which reshapes
        ``(N, T, C) -> (N, T * C)``).
    device : str, default="cpu"
        Device on which the model is placed and fine-tuned (``"cpu"``, ``"cuda"`` or
        ``"mps"``).
    fine_tuning_type : {"full", "adapter_head", "head", "scratch"}, default="full"
        Fine-tuning strategy, forwarded to ``MantisTrainer.fit``.
    num_epochs : int, default=100
        Number of fine-tuning epochs.
    batch_size : int, default=256
        Batch size used for fine-tuning and inference.
    base_learning_rate : float, default=2e-4
        Initial learning rate for the AdamW optimizer.
    learning_rate_adjusting : bool, default=True
        Whether to use Mantis' built-in learning-rate schedule.
    """

    def __init__(
        self,
        model_name: str = "paris-noah/Mantis-8M",
        *,
        n_channels: int = 3,
        device: str = "cpu",
        fine_tuning_type: str = "full",
        num_epochs: int = 100,
        batch_size: int = 256,
        base_learning_rate: float = 2e-4,
        learning_rate_adjusting: bool = True,
    ) -> None:
        self.model_name = model_name
        self.n_channels = n_channels
        self.device = device
        self.fine_tuning_type = fine_tuning_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_learning_rate = base_learning_rate
        self.learning_rate_adjusting = learning_rate_adjusting

    def to_dict(self) -> dict:
        """Return a serialisable description of the estimator configuration."""
        return {
            "name": self.__class__.__name__,
            "model_name": self.model_name,
            "n_channels": self.n_channels,
            "device": self.device,
            "fine_tuning_type": self.fine_tuning_type,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "base_learning_rate": self.base_learning_rate,
        }

    # ------------------------------------------------------------------ #
    # Network loading                                                    #
    # ------------------------------------------------------------------ #
    def _load_network(self) -> Mantis8M:
        """Instantiate a Mantis network with the requested pretrained weights.

        Mirrors the checkpoint-vs-hub logic of
        :class:`~edamame_downstream.feature_extraction.mantis.MantisExtractor`.
        """
        network = Mantis8M(device=self.device)
        local_model_path = Path(self.model_name).expanduser()

        if local_model_path.is_file():
            state_dict = _STATE_DICT_CACHE.get(self.model_name)
            if state_dict is None:
                try:
                    checkpoint = torch.load(
                        local_model_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    checkpoint = torch.load(local_model_path, map_location="cpu")
                state_dict = (
                    checkpoint.get("state_dict", checkpoint)
                    if isinstance(checkpoint, dict)
                    else checkpoint
                )
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        f"Unsupported checkpoint format at: {local_model_path}"
                    )
                # Strip DataParallel/DDP "module." prefixes if present.
                state_dict = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }
                _STATE_DICT_CACHE[self.model_name] = state_dict
            network.load_state_dict(state_dict, strict=False)
            return network

        return network.from_pretrained(self.model_name)

    # ------------------------------------------------------------------ #
    # Input reshaping                                                    #
    # ------------------------------------------------------------------ #
    def _prepare_x(self, X: np.ndarray) -> torch.Tensor:
        """Turn a flattened ``(N, T * C)`` matrix into a Mantis ``(N, C, seq_len)`` tensor."""
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        if n_features % self.n_channels != 0:
            raise ValueError(
                f"Number of features ({n_features}) is not divisible by "
                f"n_channels ({self.n_channels}); cannot reconstruct the "
                "(n_samples, n_channels, seq_len) layout."
            )
        seq_len = n_features // self.n_channels
        # NoneFeatureExtractor flattens (N, T, C) row-major, so reshape back to
        # (N, T, C) and then move the channel axis to obtain (N, C, T).
        X = X.reshape(n_samples, seq_len, self.n_channels)
        X = np.transpose(X, (0, 2, 1))  # (N, C, T)
        X = self._resize_time(X, self._seq_len)
        return torch.tensor(X, dtype=torch.float32)

    @staticmethod
    def _resize_time(x: np.ndarray, target_len: int) -> np.ndarray:
        """Resample/pad the last (time) axis of ``x`` to ``target_len``."""
        seq_len = x.shape[-1]
        if seq_len < target_len:
            pad_width = ((0, 0), (0, 0), (0, target_len - seq_len))
            return np.pad(x, pad_width, mode="constant")
        if seq_len > target_len:
            return signal.resample(x, target_len, axis=-1)
        return x

    # ------------------------------------------------------------------ #
    # scikit-learn API                                                   #
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MantisClassifier":
        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, ensure_2d=True)

        # Encode arbitrary label values to the [0, n_classes - 1] range Mantis expects.
        self._label_encoder = LabelEncoder().fit(y_checked)
        self.classes_ = self._label_encoder.classes_
        y_encoded = torch.tensor(
            self._label_encoder.transform(y_checked), dtype=torch.long
        )

        network = self._load_network()
        self._seq_len = network.seq_len
        self._trainer = MantisTrainer(device=self.device, network=network)

        x_tensor = self._prepare_x(X_checked)
        
        self._trainer.fit(
            x_tensor,
            y_encoded,
            fine_tuning_type=self.fine_tuning_type,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            base_learning_rate=self.base_learning_rate,
            learning_rate_adjusting=self.learning_rate_adjusting,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_trainer", "_label_encoder"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        x_tensor = self._prepare_x(X_checked)
        return self._trainer.predict_proba(x_tensor, batch_size=self.batch_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_trainer", "_label_encoder"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        x_tensor = self._prepare_x(X_checked)
        y_encoded = self._trainer.predict(x_tensor, batch_size=self.batch_size)
        return self._label_encoder.inverse_transform(y_encoded)
