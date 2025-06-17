import numpy as np
import torch
from src.utils.typing import DataInfo
from momentfm import MOMENTPipeline
from src.data import EDADataset


class MOMENTExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        aggregator: object | None = None,
    ):
        # TODO: figure out where to put the device_map and torch_dtype parameters
        self.pipeline = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.pipeline.init()
        self.aggregator = aggregator

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
        vals = torch.permute(
            vals, (0, 2, 1)
        )  # Change shape to (batch_size, channels, time)
        # NOTE: we are performing average pool across the time dimension (axis=1), which is standard practice with foundation models
        if not self.aggregator:
            features: np.ndarray = self.pipeline(x_enc=vals).embeddings.numpy()
        else:
            features: np.ndarray = self.aggregator(
                [self.pipeline(x_enc=vals[:, [i], :]).embeddings.numpy()for i in range(vals.shape[1])]
            )
        features = np.ma.masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data
