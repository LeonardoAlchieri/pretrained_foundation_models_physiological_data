from logging import getLogger
from typing import Any
from warnings import warn

from time import time
from neurokit2.eda import eda_peaks
from neurokit2.hrv import hrv_time
from neurokit2.ppg import ppg_findpeaks
from pywt import wavedec
from tqdm import tqdm
from numpy import (
    apply_along_axis,
    array,
    ediff1d,
    gradient,
    isnan,
    max,
    mean,
    min,
    nanmax,
    nanmean,
    nanmin,
    nansum,
    nanmedian,
    nanstd,
    log,
    std,
    trapz,
    zeros,
    ndarray,
    stack,
)
from numpy.ma import masked_invalid
from scipy.stats import linregress
from scipy.fft import fft
from tqdm.contrib.concurrent import process_map

logger = getLogger(__name__)

EDA_FEATURE_NAMES: list[str] = [
    "min_feat",
    "max_feat",
    "mean_feat",
    "std_feat",
    "dynamic_range_feat",
    "slope_feat",
    "absolute_slope_feat",
    "first_derivetive_mean_feat",
    "first_derivative_std_feat",
    "number_of_peaks_feat",
    "peaks_amplitude_feat",
    "dc_term",
    "sum_of_all_coefficients",
    "information_entropy",
    "spectral_energy",
]


def calculate_wavelet_features(
    signal: ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculates wavelet-based features for an EDA physiological signal.

    Parameters:
        signal (float): EDA physiological signal.

    Returns:
        dict: A tuple containing the calculated wavelet features.
            - 'mean_1hz': Mean of wavelet coefficients at 1Hz.
            - 'std_1hz': Standard deviation of wavelet coefficients at 1Hz.
            - 'mean_2hz': Mean of wavelet coefficients at 2Hz.
            - 'std_2hz': Standard deviation of wavelet coefficients at 2Hz.
            - 'mean_4hz': Mean of wavelet coefficients at 4Hz.
            - 'std_4hz': Standard deviation of wavelet coefficients at 4Hz.
    """
    # Perform discrete wavelet transform
    wavelet_coefs = wavedec(signal, wavelet="db4", level=6)

    def compute_feature_on_wavelet_coefs(wavelet_coef: ndarray):
        mean_coef: float = nanmean(wavelet_coef)
        std_coef: float = nanstd(wavelet_coef)
        minimum_coef: float = nanmin(wavelet_coef)
        maximum_coef: float = nanmax(wavelet_coef)
        dynamic_range_coef: float = maximum_coef - minimum_coef
        variance_coef: float = nanstd(wavelet_coef) ** 2
        standard_error_coef: float = nanstd(wavelet_coef) / len(wavelet_coef)
        return (
            mean_coef,
            std_coef,
            minimum_coef,
            maximum_coef,
            dynamic_range_coef,
            variance_coef,
            standard_error_coef,
        )

    (
        mean_1Hz,
        std_1Hz,
        minimum_1Hz,
        maximum_1Hz,
        dynamic_range_1Hz,
        variance_1Hz,
        standard_error_1Hz,
    ) = compute_feature_on_wavelet_coefs(wavelet_coefs[5])

    (
        mean_2Hz,
        std_2Hz,
        minimum_2Hz,
        maximum_2Hz,
        dynamic_range_2Hz,
        variance_2Hz,
        standard_error_2Hz,
    ) = compute_feature_on_wavelet_coefs(wavelet_coefs[4])

    (
        mean_4Hz,
        std_4Hz,
        minimum_4Hz,
        maximum_4Hz,
        dynamic_range_4Hz,
        variance_4Hz,
        standard_error_4Hz,
    ) = compute_feature_on_wavelet_coefs(wavelet_coefs[3])

    # mean_2hz: float = nanmean(wavelet_coefs[4])
    # std_2hz: float = nanstd(wavelet_coefs[4])
    # mean_4hz: float = nanmean(wavelet_coefs[3])
    # std_4hz: float = nanstd(wavelet_coefs[3])

    return (
        mean_1Hz,
        std_1Hz,
        minimum_1Hz,
        maximum_1Hz,
        dynamic_range_1Hz,
        variance_1Hz,
        standard_error_1Hz,
        mean_2Hz,
        std_2Hz,
        minimum_2Hz,
        maximum_2Hz,
        dynamic_range_2Hz,
        variance_2Hz,
        standard_error_2Hz,
        mean_4Hz,
        std_4Hz,
        minimum_4Hz,
        maximum_4Hz,
        dynamic_range_4Hz,
        variance_4Hz,
        standard_error_4Hz,
    )


def get_slop_linregress(arr: ndarray) -> list[float]:
    slopes = []
    for c in range(arr.shape[1]):
        slope, intercept, r_value, p_value, std_err = linregress(range(len(arr[:, c])), arr[:, c])
        slopes.append(slope)

    return slopes


def get_eda_peaks_info(args) -> list[list[int, float]]:
    arr, sampling_rate = args
    peaks = []
    for c in range(arr.shape[1]):
        try:
            eda_peaks_result = eda_peaks(
                arr[:, c],
                sampling_rate=sampling_rate,
            )
            # logger.debug(f"Calculated eda peaks for input {arr}")
        except ValueError as e:
            # NOTE: sometimes, when no peaks are detected, as ValueError is thrown by the
            # neurokit2 method. We solve this in a very simplistic way
            logger.warning(f"Could not extract EDA peaks. Reason: {e}")
            eda_peaks_result: tuple[None, dict[str, Any]] = (
                None,
                dict(SCR_Peaks=[], SCR_Amplitude=[0]),
            )

        peaks.append(
            [
                len(eda_peaks_result[1]["SCR_Peaks"]),
                sum(eda_peaks_result[1]["SCR_Amplitude"]),
            ]
        )

    return peaks


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
        std_feat: float = nanstd(data, axis=1)
        dynamic_range_feat: float = max_feat - min_feat

        slope_feat_res = process_map(
            get_slop_linregress,
            [data[n, ...] for n in range(data.shape[0])],
            max_workers=max_workers,  # Uses all available CPU cores
            chunksize=1000,
            desc="Computing slopes",
        )
        # slope_feat_res = [
        #     get_slop_linregress(data[n, ...]) for n in tqdm(range(data.shape[0]))
        # ]

        slope_feat = array(slope_feat_res)

        absolute_slope_feat: float = abs(slope_feat)

        first_derivative_data: ndarray = gradient(data, axis=1)
        first_derivetive_mean_feat: float = nanmean(first_derivative_data, axis=1)
        first_derivative_std_feat: float = nanstd(first_derivative_data, axis=1)

        start_time = time()
        peaks_result = process_map(
            get_eda_peaks_info,
            [(data[n, ...], sampling_rate) for n in range(data.shape[0])],
            max_workers=max_workers,  # Uses all available CPU cores
            chunksize=1000,
            desc="Computing EDA peaks",
        )
        peaks_result = array(peaks_result)
        number_of_peaks_feat, peaks_amplitude_feat = (
            peaks_result[..., 0],
            peaks_result[..., 1],
        )
        print(f"Time taken to compute EDA peaks: {time() - start_time} seconds")

        # start_time = time()
        # peaks_result_ = apply_along_axis(get_eda_peaks_info, axis=1, arr=data)
        # number_of_peaks_feat_, peaks_amplitude_feat_ = (
        #     peaks_result_[:, 0, :],
        #     peaks_result_[:, 1, :],
        # )
        # print(f"Time taken to compute EDA peaks (apply_along_axis): {time() - start_time} seconds")
        # breakpoint()

        # eda_peaks_result: dict[str, Any] = eda_peaks(
        #     data,
        #     sampling_rate=sampling_rate,
        # )

        # number_of_peaks_feat: int = len(eda_peaks_result[1]["SCR_Peaks"])
        # # NOTE: I am not sure that the sum of the amplitudes is the correct feature to be
        # # extracted
        # peaks_amplitude_feat: float = sum(eda_peaks_result[1]["SCR_Amplitude"])

        # frequency domain features (see Shkurta's references)
        fft_transform: ndarray = fft(data, axis=1)

        # dc term
        dc_term: ndarray = abs(nanmean(fft_transform, axis=1))
        # sum of all coefficients
        sum_of_all_coefficients: ndarray = abs(nansum(fft_transform, axis=1))

        # information entropy
        def get_information_entropy(arr: ndarray, rate: int, axis: int = 0) -> ndarray:
            # 1. Calculate the PSD of your signal by simply squaring the amplitude spectrum and scaling it by number of frequency bins.
            psd: ndarray = (arr**2) / rate
            # 2. Normalize the calculated PSD by dividing it by a total sum.
            norm_psd = psd / nansum(psd, axis=axis, keepdims=True)
            return -nansum(norm_psd * log(norm_psd), axis=axis)

        information_entropy: ndarray = get_information_entropy(arr=data, rate=sampling_rate, axis=1)
        # spectral energy
        spectral_energy = abs(nansum(data, axis=1)) / sampling_rate

        return stack(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                absolute_slope_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                number_of_peaks_feat,
                peaks_amplitude_feat,
                dc_term,
                sum_of_all_coefficients,
                information_entropy,
                spectral_energy,
            ],
            axis=1,
        )
        
        
def get_acc_features(data: ndarray, **kwargs) -> ndarray:
    """This method performs the feature extraction for an ACC signal.
    The features extracted are: minimum, maximum, mean, standard deviation, dynamic change,
    slope, absolute value of the slope, mean and standard deviation of the first and
    second derivative.
    The features extracted follow what done by Di Lascio et al. (2019).

    Parameters
    ----------
    data : ndarray
        acc data to extract features from.

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description
    """

    data: ndarray = data[~isnan(data)]
    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return zeros(11)
    else:
        min_feat: float = min(data)
        max_feat: float = max(data)
        mean_feat: float = mean(data)
        std_feat: float = std(data)
        dynamic_range_feat: float = max_feat - min_feat
        slope_feat, _, _, _, _ = linregress(range(len(data)), data)
        absolute_slope_feat: float = abs(slope_feat)
        first_derivative_data: ndarray = gradient(data)
        first_derivetive_mean_feat: float = mean(first_derivative_data)
        first_derivative_std_feat: float = std(first_derivative_data)
        second_derivative_data: ndarray = gradient(first_derivative_data)
        second_derivative_mean_feat: float = mean(second_derivative_data)
        second_derivative_std_feat: float = std(second_derivative_data)

        return array(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                absolute_slope_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                second_derivative_mean_feat,
                second_derivative_std_feat,
            ]
        )


def get_ppg_features(data: ndarray, sampling_rate: int = 64) -> ndarray:
    """This method computes the BVP features, as done by Di Lascio et al. (2019). The
    features extracted are:
    - **Time-domain**: statistical features (minimum, maximum, mean, standard deviation,
    dynamic change, slope, area under the curve, number of peaks, ratio between number
    of peaks and length of the segment, mean and standard deviation of the first and
    second derivative, difference between the highest and smallest peak), HR, statistical
    features on HRV (mean of all NN, standard deviation of all NN (SDNN), standard
    deviation of differences between NN (SDSD), the square root of the mean of the
    sum of the squares of differences between NN (RMSSD)


    Parameters
    ----------
    data : ndarray
        bvp data
    sampling_rate : int, optional
        sample rate, in Hz, of the signal, by default 64

    Returns
    -------
    ndarray
        array containing the features extracted, ordered as described above
    """
    # - **BVP pulse**: mean and standard deviation of pulses’ amplitude and pulses’ length
    data: ndarray = data[~isnan(data)]
    if len(data) == 0:
        return zeros(17)
    else:
        min_feat: float = min(data)
        max_feat: float = max(data)
        mean_feat: float = mean(data)
        std_feat: float = std(data)
        dynamic_range_feat: float = max_feat - min_feat
        slope_feat, _, _, _, _ = linregress(range(len(data)), data)
        auc_feat: float = trapz(data, dx=1 / sampling_rate)
        try:
            ppg_peaks: ndarray = ppg_findpeaks(data, sampling_rate=sampling_rate)[
                "PPG_Peaks"
            ]
        except ValueError as e:
            logger.warning(
                f"Some problems with the evaluation of PPG peaks. Most likely due to the way the data was filtered (expected different filter by neurokit2). See {e}"
            )
            return zeros(17)
        except IndexError as e:
            logger.warning(
                f"Some problems with the evaluation of PPG peaks. Most likely due to the way the data was filtered (expected different filter by neurokit2). See {e}"
            )
            return zeros(17)
        num_ppg_peaks_feat: int = len(ppg_peaks)
        ratio_peaks_segment_length_feat: float = num_ppg_peaks_feat / len(data)
        if num_ppg_peaks_feat == 0:
            ratio_highest_smallest_peak_feat = 0
        else:
            ratio_highest_smallest_peak_feat: float = max(data[ppg_peaks]) / min(
                data[ppg_peaks]
            )

        first_derivative_data: ndarray = gradient(data)
        first_derivetive_mean_feat: float = mean(first_derivative_data)
        first_derivative_std_feat: float = std(first_derivative_data)

        if len(ppg_peaks) <= 1:
            # FIXME: this is not really a solution, and some stuff could be done even if
            # there is only 1 peak
            hr_mean_feat: float = 0.0
            hrv_features: ndarray = zeros(4)
        else:
            hr: ndarray = eval_hr(ppg_peaks, sampling_rate=sampling_rate)
            hr_mean_feat: float = mean(hr)
            hrv_features = hrv_time(ppg_peaks, sampling_rate=sampling_rate)[
                ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD"]
            ].values[0, :]

        # TODO: there should be a few BVP pulse features, but I have no idea what they are
        return array(
            [
                min_feat,
                max_feat,
                mean_feat,
                std_feat,
                dynamic_range_feat,
                slope_feat,
                auc_feat,
                num_ppg_peaks_feat,
                ratio_peaks_segment_length_feat,
                ratio_highest_smallest_peak_feat,
                first_derivetive_mean_feat,
                first_derivative_std_feat,
                hr_mean_feat,
                hrv_features[0],
                hrv_features[1],
                hrv_features[2],
                hrv_features[3],
            ]
        )


def eval_hr(ppg_peaks: ndarray, sampling_rate: int = 64) -> ndarray:
    """A stripped down version of the neurokit2 library to evaluate HR. In our case, we
    do not care about interpolation, minimum length and the sort.

    Parameters
    ----------
    ppg_peaks : ndarray
        array of peaks. Should be in the same format as the output of the
    sampling_rate : int, optional
        sample rate in Hz, by default 64

    Returns
    -------
    ndarray
        array with the heart rate for each peak
    """
    period = ediff1d(ppg_peaks, to_begin=0) / sampling_rate
    period[0] = mean(period[1:])
    return 60 / period




class EdaFeatureExtractor:
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

    def __call__(self, arr: ndarray) -> ndarray:
        """
        Extracts features from an input EDA array.

        Parameters
        ----------
        arr : ndarray
            Input EDA array.

        Returns
        -------
        ndarray
            Extracted features.
        """

        logger.info("Extracting handcrafted features from EDA signals.")
        features = handcrafted_eda_features(
            arr,
            max_workers=self.max_workers,
            chunksize=self.chunksize,
            sampling_rate=self.sampling_rate,
        )
        features = masked_invalid(features, copy=False)
        return features.reshape(features.shape[0], -1)


class AccFeatureExtractor:
    """
    A class to extract handcrafted features from ACC signals.
    """

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
        }

    def __call__(self, arr: ndarray) -> ndarray:
        """
        Extracts features from an input ACC array.

        Parameters
        ----------
        arr : ndarray
            Input ACC array with shape (samples, timesteps) or
            (samples, timesteps, channels).

        Returns
        -------
        ndarray
            Extracted features.
        """

        logger.info("Extracting handcrafted features from ACC signals.")

        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(
                "ACC input must have shape (samples, timesteps) or "
                "(samples, timesteps, channels)."
            )

        channel_features = [
            array([get_acc_features(arr[n, :, c]) for n in range(arr.shape[0])])
            for c in range(arr.shape[2])
        ]

        features = stack(channel_features, axis=2)
        features = masked_invalid(features, copy=False)
        return features.reshape(features.shape[0], -1)


class PpgFeatureExtractor:
    """
    A class to extract handcrafted features from PPG/BVP signals.
    """

    def __init__(self, sampling_rate: int = 64):
        self.sampling_rate = sampling_rate

    def to_dict(self):
        """
        Returns a dictionary representation of the class.
        """
        return {
            "name": self.__class__.__name__,
            "sampling_rate": self.sampling_rate,
        }

    def __call__(self, arr: ndarray) -> ndarray:
        """
        Extracts features from an input PPG/BVP array.

        Parameters
        ----------
        arr : ndarray
            Input PPG/BVP array with shape (samples, timesteps) or
            (samples, timesteps, channels).

        Returns
        -------
        ndarray
            Extracted features.
        """

        logger.info("Extracting handcrafted features from PPG/BVP signals.")

        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(
                "PPG input must have shape (samples, timesteps) or "
                "(samples, timesteps, channels)."
            )

        channel_features = [
            array(
                [
                    get_ppg_features(arr[n, :, c], sampling_rate=self.sampling_rate)
                    for n in range(arr.shape[0])
                ]
            )
            for c in range(arr.shape[2])
        ]

        features = stack(channel_features, axis=2)
        features = masked_invalid(features, copy=False)
        return features.reshape(features.shape[0], -1)
