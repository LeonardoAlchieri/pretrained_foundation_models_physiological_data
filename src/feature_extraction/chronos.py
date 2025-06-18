import numpy as np
import torch
from src.utils.typing import DataInfo
from chronos import ChronosPipeline
from src.data import EDADataset
from src.utils.config import check_aggregator

class ChronosExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        aggregator: object | str = "None",
    ):
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.aggregator = check_aggregator(aggregator)

    def __call__(self, data: DataInfo) -> EDADataset:
        """
        Extracts features from the EDA dataset.

        Parameters
        ----------
        data : EDADataset
            The dataset containing EDA signals.

        Returns
        -------
        EDADataset
            The dataset with extracted features.
        """
        vals: torch.tensor = torch.tensor(data["values"], dtype=torch.float32)
        if self.aggregator == "None":
            # return an array of shape (batch_size, 1), where the value is 0
            features = np.zeros((vals.shape[0], 1), dtype=np.float32)
        else:
        # NOTE: we are performing average pool across the time dimension (axis=1), which is standard practice with foundation models
            features: np.ndarray = self.aggregator(
                [
                    self.pipeline.embed(vals[..., i])[0].numpy()
                    for i in range(vals.shape[2])
                ]
                )
            # features = np.stack(
            #     [
            #         self.pipeline.embed(vals[..., i])[0].mean(axis=1).numpy()
            #         for i in range(vals.shape[2])
            #     ],
            #     axis=2,
            # )
        features = np.ma.masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data
