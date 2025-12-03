import numpy as np


class InsideLabelProcessor:

    def __init__(
        self,
        lower_lower_threshold=0.2,
        lower_upper_threshold=0.4,
        upper_lower_threshold=0.6,
        upper_upper_threshold=0.8,
        threshold_type="percentile",
    ):
        self.lower_lower_threshold = lower_lower_threshold
        self.lower_upper_threshold = lower_upper_threshold

        self.upper_upper_threshold = upper_upper_threshold
        self.upper_lower_threshold = upper_lower_threshold

        self.threshold_type = threshold_type

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_type == "percentile":
            lower_lower_bound = np.quantile(X, self.lower_lower_threshold, axis=0)
            lower_upper_bound = np.quantile(X, self.lower_upper_threshold, axis=0)
            upper_lower_bound = np.quantile(X, self.upper_lower_threshold, axis=0)
            upper_upper_bound = np.quantile(X, self.upper_upper_threshold, axis=0)
        elif self.threshold_type == "value":
            lower_lower_bound = self.lower_lower_threshold
            lower_upper_bound = self.lower_upper_threshold
            upper_lower_bound = self.upper_lower_threshold
            upper_upper_bound = self.upper_upper_threshold
        else:
            raise ValueError("threshold_type must be either 'percentile' or 'value'")

        new_vals = np.ma.array(X.copy())
        new_vals[(new_vals < lower_lower_bound) | (new_vals > upper_upper_bound)] = (
            np.ma.masked
        )  # mark as to be removed
        new_vals[(new_vals > lower_upper_bound) & (new_vals < upper_lower_bound)] = (
            np.ma.masked
        )  # mark as to be removed
        new_vals[(new_vals >= lower_lower_bound) & (new_vals <= lower_upper_bound)] = 0
        new_vals[(new_vals >= upper_lower_bound) & (new_vals <= upper_upper_bound)] = 1
        return new_vals
