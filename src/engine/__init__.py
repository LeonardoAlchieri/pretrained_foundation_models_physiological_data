from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

# import wandb
from src.data import EDADataset
from joblib import parallel_backend


class Engine:
    def __init__(
        self,
        model: dict,
        scoring: partial,
        inner_cv_folds: int,
        resampling: None | BaseUnderSampler | str = None,
        n_jobs: int = 10,
    ):
        self.model = model["model"]
        self.param_grid = dict(model["param_grid"])
        self.scoring: Callable = scoring.func
        self.inner_cv_folds = inner_cv_folds
        self.models = []  # Store best model for each fold
        self.fold_reports: dict[str, dict] = {}
        self.resampling = resampling
        self.n_jobs = n_jobs

    def fit(self, datamodule: EDADataset):
        self.models = []
        # self.fold_reports = {}
        with parallel_backend('loky', n_jobs=self.n_jobs):
            self.imputers = []  # Store imputers for each fold
            for fold_idx, (Xy_train) in tqdm(
                enumerate(datamodule.train_data_folds),
                desc="Training folds",
                total=len(datamodule.train_data_folds),
            ):
                X_train, y_train = Xy_train["features"], Xy_train["labels"]
                imputer = SimpleImputer(strategy="mean")
                X_train = imputer.fit_transform(X_train)
                self.imputers.append(imputer)

                if self.resampling is not None:
                    X_train, y_train = self.resampling.fit_resample(
                        X_train, y_train, group=Xy_train["groups"]
                    )
                clf = GridSearchCV(
                    self.model,
                    self.param_grid,
                    scoring=make_scorer(self.scoring),
                    cv=self.inner_cv_folds,
                    verbose=0,
                )
                clf.fit(X_train, y_train)
                self.models.append(clf.best_estimator_)
                # Optionally, store best params or scores

    def test(self, datamodule: EDADataset):
        all_accuracies = []
        for fold_idx, (Xy_test) in tqdm(
            enumerate(datamodule.test_data_folds),
            desc="Testing folds",
            total=len(datamodule.test_data_folds),
        ):
            X_test, y_test = Xy_test["features"], Xy_test["labels"]
            imputer = self.imputers[fold_idx]
            X_test = imputer.transform(X_test)
            model = self.models[fold_idx]
            y_pred = model.predict(X_test)
            acc = self.scoring(y_true=y_test, y_pred=y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.fold_reports[fold_idx] = report
            all_accuracies.append(acc)

            print(
                {
                    f"fold_{fold_idx+1}_{self.scoring.__name__}": acc,
                    f"fold_{fold_idx+1}_report": report,
                }
            )
        print({f"mean_{self.scoring.__name__}": float(np.mean(all_accuracies))})
        self._save_local_results()

    def _save_local_results(self):
        save_path = HydraConfig.get().runtime.output_dir
        pd.DataFrame.from_dict(self.fold_reports, orient="index").to_csv(
            Path(save_path) / f"reports.csv"
        )
