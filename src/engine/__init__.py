import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegressor:
    def __init__(self, param_grid=None, scoring: str | callable ='accuracy', inner_cv_folds: int = 5):
        self.param_grid = param_grid or {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [1000]}
        self.scoring = scoring
        self.inner_cv_folds = inner_cv_folds
        self.models = []  # Store best model for each fold
        self.fold_reports = []

    def fit(self, datamodule):
        self.models = []
        self.fold_reports = []
        for fold_idx, (Xy_train) in enumerate(datamodule.train_data):
            X_train, y_train = Xy_train['values'], Xy_train['labels']
            clf = GridSearchCV(LogisticRegression(), self.param_grid, scoring=self.scoring, cv=self.inner_cv_folds)
            clf.fit(X_train, y_train)
            self.models.append(clf.best_estimator_)
            # Optionally, store best params or scores

    def test(self, datamodule):
        all_accuracies = []
        for fold_idx, (Xy_test) in enumerate(datamodule.test_data):
            X_test, y_test = Xy_test['values'], Xy_test['labels']
            model = self.models[fold_idx]
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.fold_reports.append(report)
            all_accuracies.append(acc)
            print(f"Fold {fold_idx+1} Accuracy: {acc:.4f}")
        print(f"Mean Accuracy: {np.mean(all_accuracies):.4f}")
