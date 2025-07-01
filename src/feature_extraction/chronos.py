import numpy as np
import torch
from src.utils.typing import DataInfo
from chronos import ChronosPipeline
from src.data import EDADataset
from src.utils.config import check_aggregator

from tqdm.auto import tqdm

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
        batch_size: int = 32,
    ):
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.aggregator = check_aggregator(aggregator)
        self.batch_size = batch_size

    def _process_channel_in_batches(self, channel_data: torch.Tensor) -> np.ndarray:
        """
        Process a single channel's data in batches to avoid memory issues.
        
        Parameters
        ----------
        channel_data : torch.Tensor
            Data for a single channel with shape (batch_size, sequence_length)
            
        Returns
        -------
        np.ndarray
            Embedded features for the channel
        """
        all_embeddings = []
        
        for i in tqdm(range(0, channel_data.shape[0], self.batch_size), desc="Batch progress"):
            batch_end = min(i + self.batch_size, channel_data.shape[0])
            batch_data = channel_data[i:batch_end]
            
            # Process the batch
            batch_embeddings = self.pipeline.embed(batch_data)[0].numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch results
        return np.concatenate(all_embeddings, axis=0)

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
            # Process each channel separately using batches to avoid memory issues
            channel_features = []
            for i in range(vals.shape[2]):
                channel_embeddings = self._process_channel_in_batches(vals[..., i])
                channel_features.append(channel_embeddings)
            
            features: np.ndarray = self.aggregator(channel_features)
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
