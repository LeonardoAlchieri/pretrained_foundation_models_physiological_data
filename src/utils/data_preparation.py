from sys import path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.preprocessing.decomposition import decomposition
from src.preprocessing.filter import butter_lowpass_filter_filtfilt


def segment_eda_data(
    data: pd.DataFrame,
    user: str,
    keys_to_get: list,
    segment_length: int = 4,
) -> pd.DataFrame:
    """
    Segments the EDA data into chunks of a specified length (in seconds).
    """
    segment_length = int(data.attrs["sampling_rate"]) * segment_length
    segments = []

    values_to_get = {key: [] for key in keys_to_get}
    groups = []
    for start in range(0, len(data), segment_length):
        end = start + segment_length
        segmented_data = data.iloc[start:end][["EDA", "Phasic", "Tonic"]].values
        if len(segmented_data) < segment_length:
            continue
        segments.append(segmented_data)
        for key in keys_to_get:
            values_to_get[key].append(data.attrs["audience_survey"][key])
        groups.append(user)

    return segments, *values_to_get.values(), groups


def process_empatica_data(data: np.ndarray) -> pd.DataFrame:
    initial_timestamp = data[0]
    sampling_rate = data[1]
    values = data[2:]
    timestamps = np.arange(
        initial_timestamp,
        initial_timestamp + len(values) / sampling_rate,
        1 / sampling_rate,
    )
    df = pd.DataFrame({"Timestamp": timestamps, "EDA": values})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.attrs["sampling_rate"] = sampling_rate
    return df


def split_data_into_sessions(data: np.ndarray, session_data: pd.DataFrame) -> dict:
    """
    Splits the EDA data into sessions based on the session start and end times.
    """
    sessions = {}
    data: pd.DataFrame = process_empatica_data(data)
    for _, row in session_data.iterrows():
        presentation_id = row["Presentation_Id"]
        start_time = row["Start"]
        end_time = row["End"]
        session_key = presentation_id
        selected_data = data[
            (data["Timestamp"] >= start_time) & (data["Timestamp"] <= end_time)
        ]
        if not selected_data.empty:
            sessions[session_key] = selected_data
    return sessions


# filtering
def apply_filter(
    data: pd.DataFrame, cutoff_frequency: float, butterworth_order: int
) -> pd.DataFrame:
    """
    Applies a low-pass Butterworth filter to the EDA data.
    """
    filtered_data = butter_lowpass_filter_filtfilt(
        data["EDA"].values,
        cutoff=cutoff_frequency,
        fs=data.attrs["sampling_rate"],
        order=butterworth_order,
    )
    data["EDA"] = filtered_data
    return data


# cvxeda decomposition
def decompose_signal(data: pd.DataFrame) -> pd.DataFrame | None:
    """
    Decomposes the EDA signal into tonic and phasic components.
    This is a placeholder function; replace with actual decomposition logic.
    """
    # Placeholder for actual decomposition logic
    try:
        decomposed = decomposition(data["EDA"].values)
    except:
        UserWarning(f"Decomposition failed. Returning None")
        return None
    data["Tonic"] = decomposed["tonic component"]
    data["Phasic"] = decomposed["phasic component"]
    return data


def prepare_data_for_sktime(data: np.ndarray) -> pd.DataFrame:
    """
    Prepares the EDA data for use with sktime by converting it into a DataFrame
    where each cell contains a pd.Series of the time series data for that sample.

    Parameters
    ----------
    data : np.ndarray
        The input EDA data of shape (n_samples, n_timesteps, n_channels).

    Returns
    -------
    pd.DataFrame
        A DataFrame suitable for sktime, with each cell containing a pd.Series.
    """

    if data.ndim == 2:
        # FIXME: hardecoded! This needs to be changed
        data = data.reshape(data.shape[0], -1, 3)
    if data.ndim != 3:
        raise ValueError(
            "Input data must be a 3D array of shape (n_samples, n_timesteps, n_channels)."
        )

    n_samples, n_timesteps, n_channels = data.shape
    sktime_data = pd.DataFrame()

    for channel in range(n_channels):
        channel_series = []
        for sample in range(n_samples):
            series = pd.Series(data[sample, :, channel])
            channel_series.append(series)
        sktime_data[f"dim_{channel}"] = channel_series

    return sktime_data
