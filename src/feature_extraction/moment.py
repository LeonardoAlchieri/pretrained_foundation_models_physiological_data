import numpy as np
import torch
from src.utils.typing import DataInfo
from momentfm import MOMENTPipeline
from src.data import EDADataset
from src.utils.config import check_aggregator
from torch.utils.data import DataLoader

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
        self.device_map = device_map
        self.torch_dtype = torch_dtype

    def _create_dataloader(self, vals: torch.Tensor) -> DataLoader:
        """
        Create a DataLoader for the input data.

        Parameters
        ----------
        vals : torch.Tensor
            Input data with shape (batch_size, channels, time)

        Returns
        -------
        DataLoader
            A DataLoader instance for the input data.
        """
        dataset = torch.utils.data.TensorDataset(vals)
        return DataLoader(dataset, batch_size=self.batch_size)

    def _process_in_batches(self, vals: torch.Tensor) -> np.ndarray:
        """
        Process data in batches using DataLoader to avoid memory issues.

        Parameters
        ----------
        vals : torch.Tensor
            Input data with shape (batch_size, channels, time)

        Returns
        -------
        np.ndarray
            Embedded features
        """
        dataloader = self._create_dataloader(vals)
        all_embeddings = []

        for batch_data in dataloader:
            batch_data = batch_data[0].to(self.device_map, dtype=self.torch_dtype)
            batch_embeddings = self.pipeline(x_enc=batch_data).embeddings.numpy()
            all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _process_channel_in_batches(self, vals: torch.Tensor, channel_idx: int) -> np.ndarray:
        """
        Process a single channel's data in batches using DataLoader to avoid memory issues.

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
        channel_data = vals[:, [channel_idx], :]
        dataloader = self._create_dataloader(channel_data)
        all_embeddings = []

        for batch_data in dataloader:
            batch_data = batch_data[0].to(self.device_map, dtype=self.torch_dtype)
            batch_embeddings = self.pipeline(x_enc=batch_data).embeddings.numpy()
            all_embeddings.append(batch_embeddings)

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
        vals = torch.permute(vals, (0, 2, 1))

        if self.aggregator == "None":
            features: np.ndarray = self._process_in_batches(vals)
        else:
            channel_features = []
            for i in range(vals.shape[1]):
                channel_embeddings = self._process_channel_in_batches(vals, i)
                channel_features.append(channel_embeddings)

            features: np.ndarray = self.aggregator(channel_features)

        features = np.ma.masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data
