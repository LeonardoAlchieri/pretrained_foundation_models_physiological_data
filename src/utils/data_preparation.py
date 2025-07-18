from src.preprocessing.filter import butter_lowpass_filter_filtfilt
import pandas as pd
import numpy as np
from sys import path
from src.preprocessing.decomposition import decomposition
from tqdm.auto import tqdm


def segment_eda_data(
    data: pd.DataFrame, user: str, keys_to_get: list, segment_length: int = 4, 
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
        initial_timestamp + len(values) * sampling_rate,
        sampling_rate,
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
def decompose_signal(data: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes the EDA signal into tonic and phasic components.
    This is a placeholder function; replace with actual decomposition logic.
    """
    # Placeholder for actual decomposition logic
    decomposed = decomposition(data["EDA"].values)
    
    data["Tonic"] = decomposed["tonic component"]
    data["Phasic"] = decomposed["phasic component"]
    return data 