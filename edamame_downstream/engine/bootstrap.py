import warnings
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import Logger
from scipy.stats import sem
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from edamame_downstream.data import EDAMAMEDataset
from edamame_downstream.engine import Engine

logger = getLogger(__name__)


def _fit_bootstrap_iter(
    args: tuple[
        list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
        Any,
        dict,
        Any,
        int,
        int | None,
    ],
) -> list:
    """Worker function for one bootstrap iteration.

    Parameters
    ----------
    args : tuple
        ``(preprocessed_folds, model, param_grid, scoring, inner_cv_folds,
        block_size)`` where *preprocessed_folds* is a list of
        ``(X_train, y_train, groups)`` tuples already imputed and resampled.
    """
    preprocessed_folds, model, param_grid, scoring, inner_cv_folds, block_size = args
    fold_models = []
    for X_train, y_train, groups in preprocessed_folds:
        n_train = len(X_train)
        b_idx = BootstrapEngine._make_bootstrap_indices(
            n_train, block_size=block_size, groups=groups
        )
        if len(b_idx) == 0:
            fold_models.append(None)
            continue
        clf = GridSearchCV(
            model,
            param_grid,
            scoring=make_scorer(scoring),
            cv=inner_cv_folds,
            verbose=0,
            n_jobs=1,
        )
        clf.fit(X_train[b_idx], y_train[b_idx])
        fold_models.append(clf.best_estimator_)
    return fold_models


class BootstrapEngine(Engine):
    """Engine with bootstrap cross-validation for robust performance estimation.

    Folds are fixed (provided by the datamodule, e.g. LOPO). The folds play
    the role of the CV splits. For each of ``b_boot`` bootstrap iterations,
    training data in every fold is resampled with replacement and a model is
    fitted. At test time the same bootstrap structure is used (one prediction
    per bootstrap × fold), and a random-effects model decomposes the variance
    into *within-bootstrap* (across folds) and *between-bootstrap* components.

    The resampling unit can be adapted to the data structure:

    * **i.i.d. observations** (default) — resample individual rows.
    * **Block bootstrap** (``block_size`` > 1) — split the sequence into
      consecutive blocks and resample blocks. Suitable for time-series data
      where successive observations are correlated; choose a block size large
      enough that inter-block correlation is negligible.
    * **Group bootstrap** (``bootstrap_by_group=True``) — resample at the
      group level (e.g. subjects in a longitudinal study). Groups are taken
      from the ``groups`` field of the fold data. Observations from different
      groups are treated as independent while within-group correlation is
      preserved.

    Parameters
    ----------
    b_boot : int
        Number of bootstrap iterations. Default 400.
    block_size : int | None
        Block size for block bootstrap. ``None`` (default) falls back to
        i.i.d. observation-level bootstrap.
    bootstrap_by_group : bool
        When ``True``, resample at the group/subject level using the
        ``groups`` array provided by the datamodule fold. Overrides
        ``block_size`` when applied to training data.
    *args, **kwargs
        Forwarded to :class:`Engine`.
    """

    def __init__(
        self,
        *args,
        b_boot: int = 400,
        block_size: int | None = None,
        bootstrap_by_group: bool = True,
        dumb_calculation: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.b_boot = b_boot
        self.block_size = block_size
        self.bootstrap_by_group = bootstrap_by_group
        self.dumb_calculation = dumb_calculation
        # _bootstrap_models[b][fold_idx] = fitted model (or None)
        self._bootstrap_models: list[list] = []
        self._preprocessed_train: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Bootstrap index generation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_bootstrap_indices(
        n: int,
        block_size: int | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an array of row indices forming one bootstrap resample.

        Parameters
        ----------
        n : int
            Number of observations.
        block_size : int | None
            ``None``  → i.i.d. multinomial resample of individual rows.
            ``int``   → consecutive-block bootstrap: split ``[0, n)`` into
                        blocks of length ``block_size`` and resample them
                        with replacement (last block may be shorter).
        groups : np.ndarray | None
            When provided, resample at the *group* level (e.g. subjects).
            All observations belonging to a sampled group are included,
            preserving within-group correlations. ``groups`` takes priority
            over ``block_size``.
        """
        if groups is not None:
            unique_groups = np.unique(groups)
            sampled_groups = np.random.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            return np.concatenate([np.where(groups == g)[0] for g in sampled_groups])

        if block_size is None or block_size <= 1:
            # Standard i.i.d. bootstrap
            W = np.random.multinomial(n, np.ones(n) / n)
            return np.repeat(np.arange(n), W)

        # Block bootstrap: build consecutive blocks, resample with replacement
        blocks = [
            np.arange(start, min(start + block_size, n))
            for start in range(0, n, block_size)
        ]
        sampled = np.random.choice(len(blocks), size=len(blocks), replace=True)
        return np.concatenate([blocks[i] for i in sampled])

    # ------------------------------------------------------------------
    # fit – Phase 1: preprocess & grid-search per fold
    #        Phase 2: for each bootstrap iter, resample & fit per fold
    # ------------------------------------------------------------------
    def fit(self, datamodule: EDAMAMEDataset):
        """Find best hyper-parameters per fold, then train b_boot × n_folds
        bootstrap models."""
        self.models = []
        self._preprocessed_train = []
        self._bootstrap_models = []
        self.imputers = []

        # --- Phase 1: preprocess each fold once ---
        preprocessed_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        for fold_idx, Xy_train in tqdm(
            enumerate(datamodule.train_data_folds),
            desc="Preprocessing folds",
            total=len(datamodule.train_data_folds),
        ):
            X_train, y_train = Xy_train["features"], Xy_train["labels"]

            X_train = self._trainable_feature_extraction_step_train(X_train)

            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            self.imputers.append(imputer)

            if self.resampling is not None:
                resampled = self.resampling.fit_resample(
                    X_train, y_train, group=Xy_train["groups"]
                )
                X_train, y_train = resampled[:2]

            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

            if self.subsample_train_set:
                _, X_train, _, y_train = train_test_split(
                    X_train,
                    y_train,
                    test_size=self.subsample_train_set,
                    stratify=y_train,
                    random_state=42,
                )

            groups = (
                np.asarray(Xy_train["groups"])
                if self.bootstrap_by_group and "groups" in Xy_train
                else None
            )
            preprocessed_folds.append((X_train, y_train, groups))

        # --- Phase 2: bootstrap training parallelised over b_boot ---
        worker_args = [
            (
                preprocessed_folds,
                self.model,
                self.param_grid,
                self.scoring,
                self.inner_cv_folds,
                self.block_size,
            )
            for _ in range(self.b_boot)
        ]
        self._bootstrap_models = process_map(
            _fit_bootstrap_iter,
            worker_args,
            desc="Bootstrap fits",
            max_workers=self.n_jobs,
        )

        if self.trainable_feature_extractor is not None:
            if len(self.models) != len(self.trainable_features_extractor_list):
                raise ValueError(
                    f"Number of trained models ({len(self.models)}) and feature "
                    f"extractors ({len(self.trainable_features_extractor_list)}) "
                    "do not match."
                )

    # ------------------------------------------------------------------
    # Random-effects variance estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_random_effects_variance(
        theta_star: np.ndarray,
        b_boot: int,
        n_folds: int,
    ) -> dict[str, float]:
        """Compute mean, SE, and 95 % CI from a ``(b_boot, n_folds)`` matrix.

        Uses the random-effects decomposition:
        * **within-bootstrap** variance  τ₀²  (across folds)
        * **between-bootstrap** variance σ_BT²
        """
        if np.all(np.isnan(theta_star)):
            return {
                "mean": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }

        theta_star = np.nan_to_num(theta_star, nan=np.nanmean(theta_star))

        theta_b_bar = np.mean(theta_star, axis=1)  # (b_boot,)
        theta_bar = float(np.mean(theta_b_bar))

        # Within-bootstrap variance (tau_0^2) — variance across folds
        tau_0_sq = float(
            np.sum((theta_star - theta_b_bar[:, None]) ** 2) / ((n_folds - 1) * b_boot)
        )

        # Between-bootstrap variance (sigma_BT^2)
        sigma_BT_sq = float(
            np.sum((theta_b_bar - theta_bar) ** 2) / (b_boot - 1) - tau_0_sq / n_folds
        )
        sigma_BT_sq = max(sigma_BT_sq, 0.0)

        se = float(np.sqrt(sigma_BT_sq))
        return {
            "mean": theta_bar,
            "se": se,
            "ci_lower": theta_bar - 1.96 * se,
            "ci_upper": theta_bar + 1.96 * se,
        }

    # ------------------------------------------------------------------
    # test – evaluate pre-trained bootstrap models (no fitting here)
    # ------------------------------------------------------------------
    def test(
        self,
        datamodule: EDAMAMEDataset,
        save: bool = True,
        lightning_logger: Logger | None = None,
    ) -> dict[int, dict]:
        """Evaluate pre-trained bootstrap models.

        Loops over bootstrap iterations (outer) and folds (inner).
        Produces a ``(b_boot, n_folds)`` score matrix per metric, then
        applies the random-effects variance model to obtain a single
        aggregate estimate with SE and 95 % CI.
        """
        if not self._bootstrap_models:
            raise RuntimeError("No bootstrap models found. Call fit() before test().")

        all_metrics = [self.scoring] + self.additional_metrics

        # --- Preprocess test data for every fold (done once) ---
        test_data: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_idx, Xy_test in enumerate(datamodule.test_data_folds):
            X_test, y_test = Xy_test["features"], Xy_test["labels"]
            X_test = self._trainable_feature_extraction_step_test(X_test, fold_idx)
            X_test = self.imputers[fold_idx].transform(X_test)
            test_data.append((X_test, y_test))

        n_folds = len(test_data)

        # Score matrices: (b_boot, n_folds) per metric
        theta_matrices = {
            m.__name__: np.full((self.b_boot, n_folds), np.nan) for m in all_metrics
        }

        # --- Outer = bootstrap, inner = folds ---
        for b in tqdm(range(self.b_boot), desc="Bootstrap CV testing"):
            fold_models = self._bootstrap_models[b]
            for fold_idx in range(n_folds):
                model = fold_models[fold_idx]
                if model is None:
                    continue

                X_test, y_test = test_data[fold_idx]
                n_test = len(X_test)

                # Bootstrap the test data
                b_test_idx = self._make_bootstrap_indices(
                    n_test, block_size=self.block_size
                )

                if len(b_test_idx) == 0:
                    continue

                y_pred_b = model.predict(X_test[b_test_idx])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for metric in all_metrics:
                        theta_matrices[metric.__name__][b, fold_idx] = metric(
                            y_true=y_test[b_test_idx], y_pred=y_pred_b
                        )

        # --- Random-effects variance: single aggregate result per metric ---
        self.bootstrap_stats: dict[str, dict[str, float]] = {}
        for metric in all_metrics:
            if self.dumb_calculation:
                self.bootstrap_stats[metric.__name__] = {
                    "mean": np.nanmean(theta_matrices[metric.__name__]),
                    "se": sem(
                        theta_matrices[metric.__name__], nan_policy="omit", axis=None
                    ),
                    "ci_lower": np.nanmean(theta_matrices[metric.__name__])
                    - 1.96
                    * sem(
                        theta_matrices[metric.__name__], nan_policy="omit", axis=None
                    ),
                    "ci_upper": np.nanmean(theta_matrices[metric.__name__])
                    + 1.96
                    * sem(
                        theta_matrices[metric.__name__], nan_policy="omit", axis=None
                    ),
                }
            else:
                self.bootstrap_stats[metric.__name__] = (
                    self._compute_random_effects_variance(
                        theta_matrices[metric.__name__],
                        self.b_boot,
                        n_folds,
                    )
                )

        # --- Per-fold means (for reports.csv compatibility) ---
        for fold_idx in range(n_folds):
            self.fold_reports[fold_idx] = {
                m.__name__: float(np.nanmean(theta_matrices[m.__name__][:, fold_idx]))
                for m in self.additional_metrics
            }

        # --- Logging ---
        scoring_stats = self.bootstrap_stats[self.scoring.__name__]
        logger.info(
            f"({datamodule.name}) bootstrap mean_{self.scoring.__name__}: "
            f"{scoring_stats['mean'] * 100:.2f} ± {scoring_stats['se'] * 100:.2f} %"
        )
        for metric in self.additional_metrics:
            stats = self.bootstrap_stats[metric.__name__]
            logger.info(
                f"{metric.__name__}: {stats['mean'] * 100:.2f} "
                f"± {stats['se'] * 100:.2f} %  "
                f"[{stats['ci_lower'] * 100:.2f}, {stats['ci_upper'] * 100:.2f}]"
            )
            if lightning_logger is not None:
                lightning_logger.log_metrics(
                    {
                        f"{datamodule.name}/{metric.__name__}": stats["mean"] * 100,
                        f"{datamodule.name}/{metric.__name__}_max": stats["ci_upper"]
                        * 100,
                        f"{datamodule.name}/{metric.__name__}_min": stats["ci_lower"]
                        * 100,
                    }
                )

        if save:
            self._save_local_results()
            self._save_bootstrap_results()

        return self.fold_reports

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _save_bootstrap_results(self):
        """Save aggregate bootstrap statistics."""
        save_path = HydraConfig.get().runtime.output_dir
        rows = []
        for metric_name, stats in self.bootstrap_stats.items():
            rows.append(
                {
                    "metric": metric_name,
                    "mean": stats["mean"],
                    "se": stats["se"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                }
            )
        pd.DataFrame(rows).to_csv(
            Path(save_path) / "bootstrap_reports.csv", index=False
        )
