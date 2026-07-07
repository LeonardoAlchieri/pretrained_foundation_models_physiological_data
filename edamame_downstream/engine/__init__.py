from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable
from copy import deepcopy

import numpy as np
import omegaconf
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from imblearn.under_sampling.base import BaseUnderSampler
from joblib import parallel_backend
from pytorch_lightning.loggers import Logger
from scipy.stats import sem
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# import wandb
from edamame_downstream.data import EDADataset

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
        subsample_train_set: bool | float = False,
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
        self.subsample_train_set = subsample_train_set

    def _trainable_feature_extraction_step_train(self, X_train: np.ndarray) -> np.ndarray:
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
    
    def _check_param_grid(self) -> bool:
        num_param_more_than_one: int = 0
        for val in self.param_grid.values():
            if len(val) > 1:
                num_param_more_than_one += 1
        if num_param_more_than_one == 0:
            return False
        else:
            return True
            

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
                if len(X_train.shape) > 2:
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    
                X_train = imputer.fit_transform(X_train)
                self.imputers.append(imputer)

                if self.resampling is not None:
                    resampled = self.resampling.fit_resample(
                        X_train, y_train, group=Xy_train["groups"]
                    )
                    X_train, y_train = resampled[:2]
                X_train = np.asarray(X_train)
                y_train = np.asarray(y_train)
                
                # Subsample training set if specified (stratified to maintain label distribution)
                if self.subsample_train_set:
                    _, X_train, _, y_train = train_test_split(
                        X_train, y_train, test_size=self.subsample_train_set, 
                        stratify=y_train, random_state=42
                    )
                if self._check_param_grid():
                    clf = GridSearchCV(
                        self.model,
                        self.param_grid,
                        scoring=make_scorer(self.scoring),
                        cv=self.inner_cv_folds,
                        verbose=0,
                        n_jobs=10,
                    )
                else:
                    clf = deepcopy(self.model)
                clf.fit(X_train, y_train)
                # NOTE: do not use clone, since it does not return a fitted model!
                # self.models.append(clf.best_estimator_)
                self.models.append(clf)

        if self.trainable_feature_extractor is not None:
            if len(self.models) != len(self.trainable_features_extractor_list):
                raise ValueError(
                    f"Number of trained models and feature extractors do not match. Received: {len(self.models)} models and {len(self.trainable_features_extractor_list)} feature extractors."
                )

    def test(
        self, datamodule: EDADataset, save: bool = True, lightning_logger: Logger | None = None
    ) -> dict[int, dict]:
        all_accuracies = []
        all_reports: dict[str, list] = {metric.__name__: [] for metric in self.additional_metrics}
        for fold_idx, (Xy_test) in tqdm(
            enumerate(datamodule.test_data_folds),
            desc="Testing folds",
            total=len(datamodule.test_data_folds),
        ):
            X_test, y_test = Xy_test["features"], Xy_test["labels"]

            X_test = self._trainable_feature_extraction_step_test(X_test, fold_idx)

            imputer = self.imputers[fold_idx]
            if len(X_test.shape) > 2:
                X_test = X_test.reshape(X_test.shape[0], -1)
                
            X_test = imputer.transform(X_test)

            model = self.models[fold_idx]
            y_pred = model.predict(X_test)
            acc = self.scoring(y_true=y_test, y_pred=y_pred)

            report = compute_metrics(y_test, y_pred, metrics=self.additional_metrics)
            for metric in self.additional_metrics:
                all_reports[metric.__name__].append(report[metric.__name__])
            self.fold_reports[fold_idx] = report
            all_accuracies.append(acc)

            logger.info(
                {
                    f"fold_{fold_idx+1}_{self.scoring.__name__}": acc,
                    f"fold_{fold_idx+1}_report": report,
                }
            )
        logger.info(
            f"({datamodule.name}) mean_{self.scoring.__name__}: {float(np.mean(all_accuracies))}"
        )
        mean_reports = {
            metric: (float(np.mean(values)), float(sem(values)))
            for metric, values in all_reports.items()
        }
        for metric, (mean, sem_value) in mean_reports.items():
            logger.info(f"{metric}: {mean*100:.2f} ± {sem_value*100:.2f} %")
            if lightning_logger is not None:
                lightning_logger.log_metrics({
                    f"{datamodule.name}/{metric}": mean * 100,
                    f"{datamodule.name}/{metric}_max": (mean+sem_value*1.98) * 100,
                    f"{datamodule.name}/{metric}_min": (mean-sem_value*1.98) * 100,
                })
        if save:
            self._save_local_results()
        return self.fold_reports

    def _save_local_results(self):
        save_path = HydraConfig.get().runtime.output_dir
        pd.DataFrame.from_dict(self.fold_reports, orient="index").to_csv(
            Path(save_path) / f"reports.csv"
        )
