from src.utils.typing import DataInfo
import numpy as np
from sklearn.model_selection import GroupKFold

# FIXME: something wrong with this one
class TACV:
    def __init__(self, num_folds: int):
        self.num_folds = num_folds

    def __call__(self, data: DataInfo) -> tuple[list[DataInfo], list[DataInfo]]:
        """
        Perform a TACV (Time-Aware Cross-Validation) split on the dataset.

        :param data: The dataset to be split.
        :return: A tuple containing the training and testing datasets for each fold.
        """
        features = data["features"]
        labels = data["labels"]
        groups = data["groups"]
        gkf = GroupKFold(n_splits=self.num_folds)

        train_folds = []
        test_folds = []

        for train_otherusers_indeces, test_user_indices in gkf.split(features, labels, groups):
            test_users = np.unique(groups[test_user_indices])
            train_indices = []
            train_indices.extend(train_otherusers_indeces)
            test_indices = []
            for user in test_users:
                user_indices = np.where(groups == user)[0]
                user_indices_sorted = np.sort(user_indices)
                n = len(user_indices_sorted)
                split_point = int(np.ceil(2 * n / 3))
                train_indices.extend(user_indices_sorted[:split_point])
                test_indices.extend(user_indices_sorted[split_point:])
            train_folds.append({
                "features": features[train_indices],
                "labels": labels[train_indices],
                "groups": groups[train_indices],
            })
            test_folds.append({
                "features": features[test_indices],
                "labels": labels[test_indices],
                "groups": groups[test_indices],
            })

        return train_folds, test_folds

class LOPO:
    def __init__(self):
        pass

    def __call__(self, data: DataInfo) -> tuple[list[DataInfo], list[DataInfo]]:
        """
        Leave-One-Participant-Out Cross Validation.
        For each unique group, use all their data as the test set and the rest as training.
        :param data: The dataset to be split.
        :return: A tuple of (train_folds, test_folds), each a list of DataInfo dicts.
        """
        features = data["features"]
        labels = data["labels"]
        groups = data["groups"]
        unique_groups = np.unique(groups)
        train_folds = []
        test_folds = []
        for group in unique_groups:
            test_idx = np.where(groups == group)[0]
            train_idx = np.where(groups != group)[0]
            train_folds.append({
                "features": features[train_idx],
                "labels": labels[train_idx],
                "groups": groups[train_idx],
            })
            test_folds.append({
                "features": features[test_idx],
                "labels": labels[test_idx],
                "groups": groups[test_idx],
            })
        return train_folds, test_folds

