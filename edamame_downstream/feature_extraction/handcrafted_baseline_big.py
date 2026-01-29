from logging import getLogger
from typing import Any
from warnings import warn

from time import time
from neurokit2.eda import eda_peaks
from pywt import wavedec
from tqdm import tqdm
from numpy import (
    apply_along_axis,
    array,
    gradient,
    isnan,
    nanmax,
    nanmean,
    nanmin,
    nansum,
    nanmedian,
    nanstd,
    log,
    zeros,
    ndarray,
    stack,
)
from numpy.ma import masked_invalid
from scipy.stats import linregress
from scipy.fft import fft
from tqdm.contrib.concurrent import process_map

from edamame_downstream.data import EDADataset
from edamame_downstream.utils.typing import DataInfo

logger = getLogger(__name__)

EDA_FEATURE_NAMES: list[str] = [
    "min_feat",
    "max_feat",
    "mean_feat",
]


def handcrafted_eda_features(
    data: ndarray, sampling_rate: int = 4, max_workers: int = 1, chunksize: int = 1
) -> ndarray:
    """This method performs the feature extraction for an EDA signal (be it mixed or phasic).
    The features extracted are: statistical features (minimum, maximum, mean, standard deviation,
    difference between maximum and minimum value or dynamic change, slope, absolute value
    of the slope, mean and standard deviation of the first derivative), number of peaks,
    peaks’ amplitude.
    The features extracted follow what done by Di Lascio et al. (2019).

    Parameters
    ----------
    data : ndarray
        eda data to extract features from.
    sampling_rate : int, optional
        sampling rate of the eda features, in Hz, by default 4.
    max_workers : int, optional
        maximum number of workers to use for parallel processing, by default 1.
    chunksize : int, optional
        chunksize for parallel processing, by default 1.

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description, i.e.,
        `[min, max, mean, std, diff_max_min, slope, absolute_slope, mean_derivative,
        std_derivative,number_peaks,peaks_amplitude]`
    """

    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return zeros(len(EDA_FEATURE_NAMES))
    else:
        min_feat: float = nanmin(data, axis=1)
        max_feat: float = nanmax(data, axis=1)
        mean_feat: float = nanmean(data, axis=1)

        return stack(
            [
                min_feat,
                max_feat,
                mean_feat,
            ],
            axis=1,
        )


class HandcraftedFeatureExtractor:
    """
    A class to extract handcrafted features from EDA signals.
    """

    def __init__(self, sampling_rate: int = 4, max_workers: int = 1, chunksize: int = 1):
        self.sampling_rate = sampling_rate
        self.max_workers = max_workers
        self.chunksize = chunksize

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
            "sampling_rate": self.sampling_rate,
        }

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

        logger.info("Extracting handcrafted features from EDA signals.")
        features = handcrafted_eda_features(
            data["values"],
            max_workers=self.max_workers,
            chunksize=self.chunksize,
            sampling_rate=self.sampling_rate,
        )
        features = masked_invalid(features, copy=False)
        data["features"] = features.reshape(features.shape[0], -1)
        data["feature_names"] = EDA_FEATURE_NAMES
        return data
