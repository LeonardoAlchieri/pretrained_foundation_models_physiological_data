import numpy as np
from tqdm import tqdm


class MeanChanAggregator:
    # NOTE: the input is (channels, batch, time, features) or (channels, batch, features)
    def __call__(self, arr: list[np.ndarray]) -> np.ndarray:
        return np.mean(arr, axis=0, keepdims=False)


class MeanTimeAggregator:
    def __call__(self, arr: np.ndarray | list) -> np.ndarray:
        if isinstance(arr, list):
            outcomes = []
            # first, do average and then concat
            for individual_arr in tqdm(arr, desc="Processing arrays", unit="array"):
                if individual_arr.ndim == 3:
                    outcomes.append(np.mean(individual_arr, axis=1, keepdims=False))
                else:
                    raise ValueError(
                        f"Unsupported array shape: {individual_arr.shape}. Expected 3D array, with shapes (N, T, D)."
                    )
            return np.concatenate(outcomes, axis=1)
        elif isinstance(arr, np.ndarray):
            if arr.ndim == 3:
                return np.mean(arr, axis=1, keepdims=False)
            elif arr.ndim == 4:
                # NOTE: reshape array as (batch, channels, time, features) instead of (channels, batch, time, features)
                arr = np.transpose(arr, (1, 0, 2, 3))
                return np.mean(arr, axis=2, keepdims=False)
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}. Expected 3D or 4D array.")
        else:
            raise TypeError(
                f"Unsupported input type: {type(arr)}. Expected np.ndarray or list of np.ndarray."
            )


class CatAggregator:

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return np.concatenate(arr, axis=1)
