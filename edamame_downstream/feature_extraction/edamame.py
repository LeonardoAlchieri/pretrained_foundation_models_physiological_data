from typing import Callable, Optional

import numpy as np
import torch
from edamame.pipeline import EdamamePipeline
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from edamame_downstream.data import EDADataset
from edamame_downstream.utils.config import check_aggregator
from edamame_downstream.utils.typing import DataInfo


class EdamameExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(
        self,
        model_name: str,
        pipeline: Optional[EdamamePipeline] = None,
        checkpoint_path: str | None = None,
        device_map: str = "cpu",
        normalization_fn: Optional[Callable] = None,
        torch_dtype: torch.dtype = torch.float32,
        aggregator: object | str = "None",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        UserWarning("torch_dtype parameter is currently not used in EdamameExtractor")
        if pipeline is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided when model_name is a string.")
            self.pipeline = EdamamePipeline.from_pretrained(
                model_name,
                weights_path=checkpoint_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                normalization_fn=normalization_fn,
            )
        else:
            if checkpoint_path is not None:
                RuntimeWarning("checkpoint_path is ignored when a pipeline instance is provided.")
            self.pipeline = EdamamePipeline(
                model=pipeline,
                model_name=model_name,
                device=self.device_map,
                normalization_fn=normalization_fn,
                model_config={},
            )
        self.aggregator = check_aggregator(aggregator)
        self.batch_size = batch_size

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
            "model_name": self.model_name,
            "device_map": self.device_map,
            "torch_dtype": str(self.torch_dtype),
            "aggregator": str(self.aggregator.__class__.__name__),
            "batch_size": self.batch_size,
        }

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

        for i in tqdm(range(0, channel_data.shape[0], self.batch_size), desc="Batch progress", leave=False):
            batch_end = min(i + self.batch_size, channel_data.shape[0])
            batch_data = channel_data[i:batch_end]

            # Process the batch
            batch_embeddings = self.pipeline.embed(batch_data, mask_ratio=0, return_mask=False)[
                0
            ].numpy()
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch results
        return np.concatenate(all_embeddings, axis=0)

    def _process_channel_with_dataloader(self, channel_data: torch.Tensor) -> np.ndarray:
        """
        Process a single channel's data using PyTorch DataLoader.

        Parameters
        ----------
        channel_data : torch.Tensor
            Data for a single channel with shape (batch_size, sequence_length)

        Returns
        -------
        np.ndarray
            Embedded features for the channel
        """
        # Create a TensorDataset and DataLoader for batching
        dataset = TensorDataset(channel_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        for batch in tqdm(dataloader, desc="Batch progress", leave=False):
            batch_data = batch[0]  # Extract the data from the batch
            batch_embeddings = self.pipeline.embed(batch_data, mask_ratio=0).to("cpu").numpy()
            # batch_embeddings = np.mean(batch_embeddings, axis=1)  # Average over patch dimension
            batch_embeddings = batch_embeddings.reshape(batch_data.shape[0], -1)
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
            features = self._process_channel_with_dataloader(vals)
        else:
            # NOTE: we are performing average pool across the time dimension (axis=1), which is standard practice with foundation models
            # Process each channel separately using batches to avoid memory issues
            channel_features = []
            for i in range(vals.shape[2]):
                channel_embeddings = self._process_channel_with_dataloader(vals[..., [i]])
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
