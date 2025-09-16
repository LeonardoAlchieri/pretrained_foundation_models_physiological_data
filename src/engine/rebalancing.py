import numpy as np
from imblearn.under_sampling import RandomUnderSampler


class GroupUnderSampler:
    def __init__(self, sampling_strategy="auto", random_state=None, replacement=False):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement

    def fit_resample(self, X, y, group):

        X_resampled = []
        y_resampled = []
        for g in np.unique(group):
            indices = np.where(group == g)[0]
            X_group = X[indices]
            y_group = y[indices]

            if self.check_user_one_class(y_group) is True:
                continue

            # Apply random under-sampling to each group
            rus = RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                replacement=self.replacement,
            )
            
            X_resampled_group, y_resampled_group = rus.fit_resample(X_group, y_group)
            X_resampled.append(X_resampled_group)
            y_resampled.append(y_resampled_group)

        # Concatenate the results from all groups
        X_resampled = np.vstack(X_resampled)

        y_resampled = np.concatenate(y_resampled)
        return X_resampled, y_resampled
    
    # Returns true if group contains one class only.
    @staticmethod
    def check_user_one_class(group) -> bool:
        if np.unique(group).size == 1: 
            return True
        return False


class NoUnderSampler:
    def fit_resample(self, X, y, group):
        return X, y
