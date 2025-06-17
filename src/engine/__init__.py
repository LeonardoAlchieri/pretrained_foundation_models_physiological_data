from pathlib import Path

import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import wandb
from src.data import EDADataset


class Engine:
    def __init__(
        self,
        model,
        scoring,
        inner_cv_folds,
    ):
        self.model = model["model"]
        self.param_grid = dict(model["param_grid"])
        self.scoring = scoring
        self.inner_cv_folds = inner_cv_folds
        self.models = []  # Store best model for each fold
        self.fold_reports: dict[str, dict] = {}

    def fit(self, datamodule: EDADataset):
        self.models = []
        # self.fold_reports = {}
        self.imputers = []  # Store imputers for each fold
        for fold_idx, (Xy_train) in enumerate(datamodule.train_data_folds):
            X_train, y_train = Xy_train["features"], Xy_train["labels"]
            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            self.imputers.append(imputer)
            clf = GridSearchCV(
                self.model,
                self.param_grid,
                scoring=self.scoring,
                cv=self.inner_cv_folds,
            )
            clf.fit(X_train, y_train)
            self.models.append(clf.best_estimator_)
            # Optionally, store best params or scores

    def test(self, datamodule: EDADataset):
        all_accuracies = []
        for fold_idx, (Xy_test) in enumerate(datamodule.test_data_folds):
            X_test, y_test = Xy_test["features"], Xy_test["labels"]
            imputer = self.imputers[fold_idx]
            X_test = imputer.transform(X_test)
            model = self.models[fold_idx]
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.fold_reports[fold_idx] = report
            all_accuracies.append(acc)

            wandb.log(
                {
                    f"fold_{fold_idx+1}_accuracy": acc,
                    f"fold_{fold_idx+1}_report": report,
                }
            )
        wandb.log({"mean_accuracy": float(np.mean(all_accuracies))})
        self._save_local_results()

    def _save_local_results(self):
        save_path = HydraConfig.get().runtime.output_dir
        pd.DataFrame.from_dict(self.fold_reports, orient='index').to_csv(
            Path(save_path) / f"reports.csv"
        )
