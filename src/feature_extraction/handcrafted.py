from logging import getLogger
from typing import Any
from warnings import warn

from neurokit2.eda import eda_peaks
from pywt import wavedec
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
    stack
)
from numpy.ma import masked_invalid
from scipy.stats import linregress
from scipy.fft import fft

from src.data import EDADataset

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


def handcrafted_eda_features(data: ndarray, sampling_rate: int = 4) -> ndarray:
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

    Returns
    -------
    ndarray
        the method returns an array of extracted features, in the order given in the
        description, i.e.,
        `[min, max, mean, std, diff_max_min, slope, absolute_slope, mean_derivative,
        std_derivative,number_peaks,peaks_amplitude]`
    """

    # Use nan_mask for further processing, or mask with np.ma:
    data = masked_invalid(data)
    logger.debug(f"Len of eda data after removal of NaN: {len(data)}")
    if len(data) == 0:
        return zeros(len(EDA_FEATURE_NAMES))
    else:
        min_feat: float = nanmin(data, axis=1)
        max_feat: float = nanmax(data, axis=1)
        mean_feat: float = nanmean(data, axis=1)
        std_feat: float = nanstd(data, axis=1)
        dynamic_range_feat: float = max_feat - min_feat

        def get_slop_linregress(arr: ndarray) -> float:
            slope, intercept, r_value, p_value, std_err = linregress(
                range(len(arr)), arr
            )
            return slope

        slope_feat = apply_along_axis(get_slop_linregress, axis=1, arr=data)
        absolute_slope_feat: float = abs(slope_feat)

        first_derivative_data: ndarray = gradient(data, axis=1)
        first_derivetive_mean_feat: float = nanmean(first_derivative_data, axis=1)
        first_derivative_std_feat: float = nanstd(first_derivative_data, axis=1)

        def get_eda_peaks_info(arr: ndarray) -> dict[str, Any]:
            try:
                eda_peaks_result = eda_peaks(
                    arr,
                    sampling_rate=sampling_rate,
                )
                logger.debug(f"Calculated eda peaks for input {arr}")
            except ValueError as e:
                # NOTE: sometimes, when no peaks are detected, as ValueError is thrown by the
                # neurokit2 method. We solve this in a very simplistic way
                logger.warning(f"Could not extract EDA peaks. Reason: {e}")
                eda_peaks_result: tuple[None, dict[str, Any]] = (
                    None,
                    dict(SCR_Peaks=[], SCR_Amplitude=[0]),
                )

            return len(eda_peaks_result[1]["SCR_Peaks"]), sum(
                eda_peaks_result[1]["SCR_Amplitude"]
            )

        
        
        peaks_result = apply_along_axis(
            get_eda_peaks_info, axis=1, arr=data
        )
        number_of_peaks_feat, peaks_amplitude_feat = peaks_result[:,0,:], peaks_result[:,1,:]

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

        information_entropy: ndarray = get_information_entropy(
            arr=data, rate=sampling_rate, axis=1
        )
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
            axis=1
        )
        