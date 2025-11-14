from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable

import numpy as np
import omegaconf
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from imblearn.under_sampling.base import BaseUnderSampler
from joblib import parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

# import wandb
from src.data import EDADataset

logger = getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[Callable],
) -> dict[str, float]:
    results = {}
    for metric in metrics:
        results[metric.__name__] = metric(y_true, y_pred)
    return results


class Engine:
    def __init__(
        self,
        model: dict,
        scoring: partial,
        additional_metrics: list[partial],
        inner_cv_folds: int,
        resampling: BaseUnderSampler | None = None,
        trainable_feature_extractor: BaseEstimator | None = None,
        n_jobs: int = 10,
    ):
        self.model = model["model"]
        self.param_grid = dict(model["param_grid"])
        self.scoring: Callable = scoring.func
        self.inner_cv_folds = inner_cv_folds
        self.models = []  # Store best model for each fold
        self.fold_reports: dict[int, dict] = {}
        self.resampling = resampling
        self.n_jobs = n_jobs
        self.additional_metrics = [val.func for val in additional_metrics]
        self.trainable_feature_extractor = (
            trainable_feature_extractor
            if not type(trainable_feature_extractor) == omegaconf.dictconfig.DictConfig
            else None
        )
        self.trainable_features_extractor_list = []

    def _trainable_feature_extraction_step_train(
        self, X_train: np.ndarray
    ) -> np.ndarray:
        if self.trainable_feature_extractor is not None:
            feature_extractor = clone(self.trainable_feature_extractor)
            feature_extractor = feature_extractor.fit(X_train)
            X_train = feature_extractor.transform(X_train)
            self.trainable_features_extractor_list.append(feature_extractor)
        return X_train

    def _trainable_feature_extraction_step_test(
        self, X_test: np.ndarray, fold_idx: int
    ) -> np.ndarray:
        if self.trainable_feature_extractor is not None:
            feature_extractor = self.trainable_features_extractor_list[fold_idx]
            X_test = feature_extractor.transform(X_test)
        return X_test

    def fit(self, datamodule: EDADataset):
        self.models = []
        # self.fold_reports = {}
        with parallel_backend("threading", n_jobs=self.n_jobs):
            self.imputers = []  # Store imputers for each fold
            for fold_idx, (Xy_train) in tqdm(
                enumerate(datamodule.train_data_folds),
                desc="Training folds",
                total=len(datamodule.train_data_folds),
            ):
                X_train, y_train = Xy_train["features"], Xy_train["labels"]

                X_train = self._trainable_feature_extraction_step_train(X_train)

                # TODO: hard coded, to add in conf file
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

                clf = GridSearchCV(
                    self.model,
                    self.param_grid,
                    scoring=make_scorer(self.scoring),
                    cv=self.inner_cv_folds,
                    verbose=0,
                )
                clf.fit(X_train, y_train)
                # NOTE: do not use clone, since it does not return a fitted model!
                self.models.append(clf.best_estimator_)
                # Optionally, store best params or scores

        if self.trainable_feature_extractor is not None:
            if len(self.models) != len(self.trainable_features_extractor_list):
                raise ValueError(
                    f"Number of trained models and feature extractors do not match. Received: {len(self.models)} models and {len(self.trainable_features_extractor_list)} feature extractors."
                )

    def test(self, datamodule: EDADataset, save: bool = True) -> dict[int, dict]:
        all_accuracies = []
        for fold_idx, (Xy_test) in tqdm(
            enumerate(datamodule.test_data_folds),
            desc="Testing folds",
            total=len(datamodule.test_data_folds),
        ):
            X_test, y_test = Xy_test["features"], Xy_test["labels"]

            X_test = self._trainable_feature_extraction_step_test(X_test, fold_idx)

            imputer = self.imputers[fold_idx]
            X_test = imputer.transform(X_test)

            model = self.models[fold_idx]
            y_pred = model.predict(X_test)
            acc = self.scoring(y_true=y_test, y_pred=y_pred)

            report = compute_metrics(y_test, y_pred, metrics=self.additional_metrics)
            self.fold_reports[fold_idx] = report
            all_accuracies.append(acc)

            logger.info(
                {
                    f"fold_{fold_idx+1}_{self.scoring.__name__}": acc,
                    f"fold_{fold_idx+1}_report": report,
                }
            )
        logger.info({f"mean_{self.scoring.__name__}": float(np.mean(all_accuracies))})
        if save:
            self._save_local_results()
        return self.fold_reports

    def _save_local_results(self):
        save_path = HydraConfig.get().runtime.output_dir
        pd.DataFrame.from_dict(self.fold_reports, orient="index").to_csv(
            Path(save_path) / f"reports.csv"
        )
