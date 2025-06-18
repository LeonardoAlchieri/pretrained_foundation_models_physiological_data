import numpy as np


class MeanChanAggregator:
    # NOTE: the input is (channels, batch, time, features) or (channels, batch, features)
    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return np.mean(arr, axis=0, keepdims=False)
    
class MeanTimeAggregator:
    def __call__(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        
        if arr.ndim == 4:
            # NOTE: reshape array as (batch, channels, time, features) instead of (channels, batch, time, features)
            arr = np.transpose(arr, (1, 0, 2, 3))
            return np.mean(arr, axis=2, keepdims=False)
        elif arr.ndim == 3:
            # NOTE: reshape array as (batch, channels, features) instead of (channels, batch, features)
            arr = np.transpose(arr, (1, 0, 2))
            return arr
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}. Expected 2D or 3D array.")
    
class CatAggregator:

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return np.concatenate(arr, axis=1)