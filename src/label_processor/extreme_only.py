import numpy as np


class ExtremeOnlyLabelProcessor:

    def __init__(
        self, lower_threshold=0.25, upper_threshold=0.75, threshold_type="percentile"
    ):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.threshold_type = threshold_type

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_type == "percentile":
            lower_bound = np.quantile(X, self.lower_threshold, axis=0)
            upper_bound = np.quantile(X, self.upper_threshold, axis=0)
        elif self.threshold_type == "value":
            lower_bound = self.lower_threshold
            upper_bound = self.upper_threshold
        else:
            raise ValueError("threshold_type must be either 'percentile' or 'value'")

        new_vals = np.ma.array(X.copy())
        new_vals[(new_vals > lower_bound) & (new_vals < upper_bound)] = (
            np.ma.masked
        )  # mark as to be removed
        new_vals[(new_vals <= lower_bound)] = 0
        new_vals[(new_vals >= upper_bound)] = 1
        return new_vals
