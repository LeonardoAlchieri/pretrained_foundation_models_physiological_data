import numpy as np
import torch
from src.utils.typing import DataInfo
from momentfm import MOMENTPipeline
from src.data import EDADataset
from src.utils.config import check_aggregator

class MOMENTExtractor:
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
        # TODO: figure out where to put the device_map and torch_dtype parameters
        self.pipeline = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.pipeline.init()
        self.aggregator = check_aggregator(aggregator)
        self.batch_size = batch_size

    def _process_in_batches(self, vals: torch.Tensor) -> np.ndarray:
        """
        Process data in batches to avoid memory issues.
        
        Parameters
        ----------
        vals : torch.Tensor
            Input data with shape (batch_size, channels, time)
            
        Returns
        -------
        np.ndarray
            Embedded features
        """
        all_embeddings = []
        
        for i in range(0, vals.shape[0], self.batch_size):
            batch_end = min(i + self.batch_size, vals.shape[0])
            batch_data = vals[i:batch_end]
            
            # Process the batch
            batch_embeddings = self.pipeline(x_enc=batch_data).embeddings.numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch results
        return np.concatenate(all_embeddings, axis=0)

    def _process_channel_in_batches(self, vals: torch.Tensor, channel_idx: int) -> np.ndarray:
        """
        Process a single channel's data in batches to avoid memory issues.
        
        Parameters
        ----------
        vals : torch.Tensor
            Input data with shape (batch_size, channels, time)
        channel_idx : int
            Index of the channel to process
            
        Returns
        -------
        np.ndarray
            Embedded features for the channel
        """
        all_embeddings = []
        
        for i in range(0, vals.shape[0], self.batch_size):
            batch_end = min(i + self.batch_size, vals.shape[0])
            batch_data = vals[i:batch_end, [channel_idx], :]
            
            # Process the batch
            batch_embeddings = self.pipeline(x_enc=batch_data).embeddings.numpy()
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
        # NOTE: Change shape to (batch_size, channels, time)
        vals = torch.permute(vals, (0, 2, 1))
        # NOTE: we are performing average pool across the time dimension (axis=1), which is standard practice with foundation models
        if self.aggregator == "None":
            features: np.ndarray = self._process_in_batches(vals)
        else:
            # Process each channel separately using batches to avoid memory issues
            channel_features = []
            for i in range(vals.shape[1]):
                channel_embeddings = self._process_channel_in_batches(vals, i)
                channel_features.append(channel_embeddings)
            
            features: np.ndarray = self.aggregator(channel_features)
        features = np.ma.masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data
