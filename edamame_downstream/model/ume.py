"""UME (Edamame/EfficientNet) used as an end-to-end (fine-tunable) classifier.

Unlike :class:`edamame_downstream.feature_extraction.edamame.EdamameExtractor`,
which uses the pretrained UME encoder only as a frozen feature extractor, this
module exposes UME as a *model* in the sense of ``configs/classification/model/``:
a scikit-learn-style classifier that is fine-tuned on the downstream task via
``EdamamePipeline.fit`` and predicts labels directly with
``EdamamePipeline.predict`` / ``predict_proba``.

This lets UME be swept and evaluated by the existing
:class:`edamame_downstream.engine.Engine` (i.e. ``GridSearchCV``) exactly like any
other sklearn classifier defined under ``edamame_downstream.model``.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from edamame.pipeline import EdamamePipeline

logger = getLogger(__name__)


class UMEClassifier(BaseEstimator, ClassifierMixin):
    """Fine-tunable UME classifier with a scikit-learn API.

    The estimator receives the (flattened) raw signal as produced by
    :class:`~edamame_downstream.feature_extraction.none.NoneFeatureExtractor`
    (``preserve_shape=False``), i.e. a 2D array of shape
    ``(n_samples, seq_len * n_channels)``, and reshapes it back to the
    ``(n_samples, seq_len, n_channels)`` layout expected by
    :meth:`EdamamePipeline.fit`. UME is trained on EDA data at the native
    resolution, so — unlike Mantis — the sequence length is kept as-is (no
    resampling / padding).

    Parameters
    ----------
    weights_path : str
        Path to the pretrained UME checkpoint (``.ckpt``) loaded via
        :meth:`EdamamePipeline.from_pretrained`.
    model_name : str, default="efficientnet"
        Backbone identifier understood by ``EdamamePipeline``. Fine-tuning is
        only supported for the EfficientNet (UME) encoder.
    n_channels : int, default=3
        Number of channels contained in the flattened input. Required to invert
        the row-major flattening performed by ``NoneFeatureExtractor`` (which
        reshapes ``(N, T, C) -> (N, T * C)``).
    device : str, default="cpu"
        Device on which the model is placed and fine-tuned (``"cpu"``,
        ``"cuda"``, ``"mps"`` or ``"auto"``).
    normalization_fn : callable, optional
        Optional per-sample normalisation applied by the pipeline.
    model_config : dict, optional
        Optional explicit model configuration forwarded to
        ``EdamamePipeline.from_pretrained``.
    fine_tuning_type : {"full", "head", "scratch"}, default="full"
        Fine-tuning strategy forwarded to ``EdamamePipeline.fit``.
    learning_rate : float, default=1e-3
        Learning rate for the classification head (and backbone when
        ``backbone_lr`` is None).
    backbone_lr : float, optional
        Separate (typically smaller) learning rate for the encoder. Only used
        for ``"full"``/``"scratch"``.
    batch_size : int, default=64
        Mini-batch size used for fine-tuning and inference.
    num_epochs : int, default=100
        Number of fine-tuning epochs.
    weight_decay : float, default=0.0
        AdamW weight decay.
    class_weight : {"balanced", torch.Tensor}, optional
        Optional cross-entropy class weights.
    verbose : bool, default=False
        Whether the pipeline prints per-epoch loss/accuracy.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        *,
        model_name: str = "efficientnet",
        n_channels: int = 3,
        device: str = "cpu",
        normalization_fn: Optional[Callable] = None,
        model_config: Optional[dict] = None,
        fine_tuning_type: str = "full",
        learning_rate: float = 1e-3,
        backbone_lr: Optional[float] = None,
        batch_size: int = 64,
        num_epochs: int = 100,
        weight_decay: float = 0.0,
        use_layer_norm: bool = True,
        class_weight: Optional[Any] = None,
        early_stopping: bool = False,
        patience: int = 3,
        monitor: str = "val_macro_f1",
        verbose: bool = False,
    ) -> None:
        self.weights_path = weights_path
        self.model_name = model_name
        self.n_channels = n_channels
        self.device = device
        self.normalization_fn = normalization_fn
        self.model_config = model_config
        self.fine_tuning_type = fine_tuning_type
        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.verbose = verbose
        self.use_layer_norm = use_layer_norm
        self.early_stopping = early_stopping
        self.patience = patience
        self.monitor = monitor

    def to_dict(self) -> dict:
        """Return a serialisable description of the estimator configuration."""
        return {
            "name": self.__class__.__name__,
            "model_name": self.model_name,
            "weights_path": self.weights_path,
            "n_channels": self.n_channels,
            "device": self.device,
            "fine_tuning_type": self.fine_tuning_type,
            "learning_rate": self.learning_rate,
            "backbone_lr": self.backbone_lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
        }

    # ------------------------------------------------------------------ #
    # Input reshaping                                                    #
    # ------------------------------------------------------------------ #
    def _prepare_x(self, X: np.ndarray) -> np.ndarray:
        """Turn a flattened ``(N, T * C)`` matrix into ``(N, seq_len, n_channels)``.

        The UME encoder consumes EDA data at its native resolution, so the
        sequence length is preserved (no resampling). ``NoneFeatureExtractor``
        flattens ``(N, T, C)`` row-major, so a plain reshape recovers the native
        ``(N, seq_len, n_channels)`` layout expected by the pipeline.
        """
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        if n_features % self.n_channels != 0:
            raise ValueError(
                f"Number of features ({n_features}) is not divisible by "
                f"n_channels ({self.n_channels}); cannot reconstruct the "
                "(n_samples, seq_len, n_channels) layout."
            )
        seq_len = n_features // self.n_channels
        return X.reshape(n_samples, seq_len, self.n_channels)

    # ------------------------------------------------------------------ #
    # scikit-learn API                                                   #
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "UMEClassifier":
        if self.weights_path is None:
            raise ValueError("weights_path must be provided to load the pretrained UME encoder.")

        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, ensure_2d=True)

        self._pipeline = EdamamePipeline.from_pretrained(
            self.model_name,
            weights_path=self.weights_path,
            model_config=self.model_config,
            device_map=self.device,
            normalization_fn=self.normalization_fn,
        )

        x_reshaped = self._prepare_x(X_checked)
        # The pipeline encodes arbitrary labels internally and restores the
        # original values in ``predict``; no external LabelEncoder is needed.
        self._pipeline.fit(
            x_reshaped,
            y_checked,
            fine_tuning_type=self.fine_tuning_type,
            learning_rate=self.learning_rate,
            backbone_lr=self.backbone_lr,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            class_weight=self.class_weight,
            use_layer_norm=self.use_layer_norm,
            verbose=self.verbose,
            validation_split=0.2,
            early_stopping=self.early_stopping,
            patience=self.patience,
            monitor=self.monitor
        )
        self.classes_ = np.asarray(self._pipeline.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_pipeline"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        x_reshaped = self._prepare_x(X_checked)
        return self._pipeline.predict_proba(x_reshaped, batch_size=self.batch_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["_pipeline"])
        X_checked = check_array(X, accept_sparse=False, ensure_2d=True)
        x_reshaped = self._prepare_x(X_checked)
        return self._pipeline.predict(x_reshaped, batch_size=self.batch_size)
