from typing import TypedDict
import numpy as np

DataInfo = TypedDict(
    "DataInfo",
    {
        "name": str,
        "values": np.ndarray,
        "labels": np.ndarray,
        "groups": np.ndarray,
        "features": np.ndarray | None,
        "feature_names": list[str] | None,
    },
)