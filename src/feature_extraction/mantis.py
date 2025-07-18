from typing import Callable
import numpy as np
import torch
from src.utils.typing import DataInfo
from mantis.trainer import MantisTrainer
from mantis.architecture import Mantis8M
from src.data import EDADataset
from src.utils.config import check_aggregator
from scipy import signal

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset


class MantisExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        aggregator: Callable | str = "None",
        batch_size: int = 32,
        channel_together: bool = True,
    ):
        self.pipeline_name = model_name
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        network = Mantis8M(device=device_map)
        self.network = network.from_pretrained(model_name)
        # network.seq_len
        self.pipeline = MantisTrainer(device=device_map, network=self.network)
        self.aggregator = check_aggregator(aggregator)
        self.batch_size = batch_size
        self.channel_together = channel_together

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
            "model_name": self.pipeline_name,
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

        for i in tqdm(
            range(0, channel_data.shape[0], self.batch_size), desc="Batch progress"
        ):
            batch_end = min(i + self.batch_size, channel_data.shape[0])
            batch_data = channel_data[i:batch_end]

            # Process the batch
            batch_embeddings = self.pipeline.transform(batch_data)[0].numpy()
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch results
        return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def _pad_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Pads the input tensor to the target length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be padded.
        target_len : int
            Target length for padding.

        Returns
        -------
        torch.Tensor
            Padded tensor.
        """
        seq_len = x.shape[1]
        if seq_len < target_len:
            pad_width = ((0, 0), (0, target_len - seq_len), (0, 0))
            X_padded = np.pad(x, pad_width, mode="constant")
        elif seq_len > target_len:
            # If the sequence is longer than the target, resample to match the target length
            x = x.numpy()
            X_padded = signal.resample(x, target_len, axis=1)
        else:
            X_padded = x
        return torch.Tensor(X_padded)

    def _process_channel_with_dataloader(
        self, channel_data: torch.Tensor
    ) -> np.ndarray:
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

        if channel_data.ndim < 3:
            channel_data = channel_data.unsqueeze(2)

        # num_real_patches = (
        #     1
        #     + (channel_data.shape[1] - self.pipeline.config.patch_length)
        #     // self.pipeline.config.patch_stride
        # )
        dataset = TensorDataset(channel_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        for batch in tqdm(dataloader, desc="Batch progress"):
            batch_data: torch.Tensor = batch[0]  # Extract the data from the batch
            batch_data = self._pad_len(batch_data, self.network.seq_len)
            batch_data = batch_data.movedim(1, 2)  # Change shape to (batch_size, channels, time)
            batch_embeddings = (
                self.pipeline.transform(batch_data)
            )
            # batch_embeddings = batch_embeddings[:, :, :num_real_patches, :]
            # swap axis 0 and 1 to have shape (channels,batch_size,time,features) from (batch_size, channels, time, features)
            # batch_embeddings = np.transpose(batch_embeddings, (1, 0, 2, 3))
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
        if self.aggregator != "None":
            # return an array of shape (batch_size, 1), where the value is 0
            UserWarning("Mantis does not require a channel aggregator")
            features = np.zeros((vals.shape[0], 1), dtype=np.float32)
        else:
            if not self.channel_together:
                # NOTE: we are performing average pool across the time dimension (axis=1), which is standard practice with foundation models
                # Process each channel separately using batches to avoid memory issues
                channel_features = []
                for i in range(vals.shape[2]):
                    channel_embeddings = self._process_channel_with_dataloader(
                        vals[..., [i]]
                    )
                    channel_features.append(channel_embeddings[0, ...])
            else:
                # Process all channels together
                channel_features = self._process_channel_with_dataloader(vals)

            features: np.ndarray = channel_features

        features = np.ma.masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = None
        return data
