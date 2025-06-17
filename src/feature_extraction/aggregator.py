import numpy as np


class MeanAggregator:

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return np.mean(arr, axis=0, keepdims=False)
    
class CatAggregator:

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return np.concatenate(arr, axis=1)