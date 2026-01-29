import numpy as np


class CustomSelectorLabelProcessor:

    def __init__(self, positive_class: list[str] | str, negative_class: list[str] | str):
        if isinstance(positive_class, str):
            positive_class = [positive_class]
        if isinstance(negative_class, str):
            negative_class = [negative_class]
        self.positive_class = positive_class
        self.negative_class = negative_class
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        new_vals = np.ma.array(X.copy())
        for pos in self.positive_class:
            new_vals[X == pos] = 1
        for neg in self.negative_class:
            new_vals[X == neg] = 0
        mask = ~np.isin(X, self.positive_class + self.negative_class)
        new_vals[mask] = np.ma.masked  # mark as to be removed
        return new_vals