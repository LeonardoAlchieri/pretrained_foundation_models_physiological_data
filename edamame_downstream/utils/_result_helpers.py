"""
Internal helpers for result loading, filtering, deduplication, and display.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from yaml import safe_load

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_COLUMNS: list[str] = [
    "Aggregator",
    "Dataset",
    "Features",
    "Label Name",
    "Model",
    "Resampling",
    "Side",
    "Validation",
    "Channels",
    "Feature Scaling",
    "Sample Scaling",
    "Subsample Train Set",
]

KNOWN_METRICS: list[str] = [
    "accuracy_score",
    "balanced_accuracy_score",
    "f1_score",
    "matthews_corrcoef",
    "roc_auc_score",
    "precision_score",
    "recall_score",
]

METRIC_RENAME: dict[str, str] = {
    "balanced_accuracy_score": "Balanced Accuracy",
    "f1_score": "F1",
    "matthews_corrcoef": "MCC",
    "roc_auc_score": "ROC AUC",
    "precision_score": "Precision",
    "recall_score": "Recall",
}

# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


def extract_features_name(conf: dict) -> str:
    """Derive a human-readable feature-extractor name from a Hydra config dict."""
    if "model_name" in conf["feature_extractor"]:
        name = conf["feature_extractor"]["model_name"]
        if "checkpoint_path" in conf["feature_extractor"]:
            stem = Path(conf["feature_extractor"]["checkpoint_path"]).stem
            name = f"{name}_{stem}"
        return name

    parts = conf["feature_extractor"]["_target_"].split(".")
    class_name = parts[-1]
    # Use module name to disambiguate generic class names (e.g. HandcraftedFeatureExtractor)
    if class_name == "HandcraftedFeatureExtractor" and len(parts) >= 2:
        return parts[-2]
    return class_name


def extract_timestamp(reports_path: Path) -> str:
    """Return a sortable timestamp string extracted from a run output path.

    Handles two formats:
    1. Single run : .../outputs/YYYY-MM-DD/HH-MM-SS/reports.csv
    2. Sweep      : .../outputs/DATASET/multirun_YYYY-MM-DD-HH-MM-SS/RUN_ID/reports.csv
    """
    parts = reports_path.parts
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            if i + 1 < len(parts) and len(parts[i + 1]) == 8:
                return f"{part}_{parts[i + 1]}"
    for part in parts:
        if part.startswith("multirun_"):
            m = re.match(r"multirun_(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2}-\d{2})", part)
            if m:
                return f"{m.group(1)}_{m.group(2)}"
    return "0000-00-00_00-00-00"


def parse_run_metadata(
    reports_path: Path,
    conf: dict,
    val_method: str,
    remove_xgboost: bool,
) -> dict | None:
    """Parse a Hydra config into a run-metadata dict.

    Returns ``None`` when the run should be skipped (wrong validation method or
    XGBoost filter active).
    """
    if conf["validation_method"]["_target_"].split(".")[-1].lower() != val_method:
        return None

    model_name: str = conf["model"]["model"]["_target_"].split(".")[-1]
    if remove_xgboost and model_name == "XGBClassifier":
        return None

    validation_method = conf["validation_method"]["_target_"].split(".")[-1]
    if "aggregator" in conf and "_target_" in conf["aggregator"]:
        aggregator = conf["aggregator"]["_target_"].split(".")[-1]
    elif "aggregator" in conf:
        aggregator = None
    else:
        aggregator = "MeanTimeAggregator"

    seed = conf["seed"]
    if isinstance(conf["dataset"], str):
        print(f"{conf['dataset']}, {reports_path} has no side information.")
    if "side" in conf["dataset"].keys():
        side = conf["dataset"]["side"]
        if side == "${side}":
            side = conf["side"]
        label_name = conf["label_name"]
        dataset = conf["dataset"]["dataset"]
    else:
        side = conf["side"]
        label_name = "None"
        dataset = conf["dataset"]

    # Dreamt: distinguish between sleep/wake and deep/light sleep tasks
    # (both share label_name: Sleep_Stage but have different class configs).
    if dataset == "Dreamt" and "label_processor" in conf:
        lp = conf["label_processor"]
        if "positive_class" in lp and "negative_class" in lp:
            pos, neg = set(lp["positive_class"]), set(lp["negative_class"])
            if pos == {"N3"} and neg == {"R"}:
                label_name = "DeepSleep_vs_REM"
            elif pos == {"R", "N3", "N2", "N1"} and neg == {"W", "P"}:
                label_name = "Sleep_vs_Wake"

    resampling = conf["resampling"]["_target_"].split(".")[-1] if "resampling" in conf else "None"
    resampling = resampling if resampling != "NoUnderSampler" else "None"

    channels = (
        str(conf["datamodule"]["channels"])
        if "channels" in conf["datamodule"]
        else "[0,1,2]"
    )
    feature_scaling = (
        conf["feature_scaling_method"]["_target_"].split(".")[-1]
        if "feature_scaling_method" in conf
        else "Unknown"
    )
    sample_scaling = (
        conf["sample_scaling_method"]["_target_"].split(".")[-1]
        if "sample_scaling_method" in conf
        else "None"
    )
    subsample_train_set = conf["engine"].get("subsample_train_set", 1)
    if subsample_train_set is False:
        subsample_train_set = 1

    return {
        "Dataset": dataset,
        "Side": side,
        "Seed": seed,
        "Label Name": label_name,
        "Model": model_name,
        "Resampling": resampling,
        "Features": extract_features_name(conf),
        "Aggregator": aggregator,
        "Validation": validation_method,
        "Channels": channels,
        "Feature Scaling": feature_scaling,
        "Sample Scaling": sample_scaling,
        "Timestamp": extract_timestamp(reports_path),
        "Subsample Train Set": subsample_train_set,
    }


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    filters: dict[str, str | list[str] | None],
    exclude_filters: dict[str, str | list[str] | None],
) -> pd.DataFrame:
    """Apply positive and negative column filters to *df* and return the result."""
    for col, value in filters.items():
        if value is None:
            continue
        if col not in df.columns:
            print(f"Warning: Filter column '{col}' not found in results. Skipping.")
            continue
        df = df[df[col].isin([value] if isinstance(value, str) else value)]

    for col, value in exclude_filters.items():
        if value is None:
            continue
        if col not in df.columns:
            print(f"Warning: Exclude filter column '{col}' not found in results. Skipping.")
            continue
        df = df[~df[col].isin([value] if isinstance(value, str) else value)]

    return df


def dedup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most-recent run per config combination and sort."""
    df = df.sort_values("Timestamp", ascending=False)
    df = df.drop_duplicates(subset=CONFIG_COLUMNS, keep="first")
    return df.sort_values(CONFIG_COLUMNS)


def format_avg_sem_columns(
    df: pd.DataFrame,
    scale: float = 1.0,
    decimals: int = 2,
) -> pd.DataFrame:
    """Pivot ``<metric> avg`` / ``<metric> sem`` column pairs into formatted strings.

    Each numeric pair becomes a ``"mean ± ci"`` string.  *scale* is applied
    before rounding (e.g. pass ``100`` to convert fractions to percentages).
    """
    fmt = f"%.{decimals}f ± %.{decimals}f"

    def _fmt_pair(x: pd.DataFrame) -> pd.Series:
        if len(x) > 1:
            return pd.Series(
                [
                    fmt % (
                        np.round(x.values[0][i] * scale, decimals),
                        np.round(x.values[1][i] * scale, decimals),
                    )
                    for i in range(x.shape[1])
                ],
                index=x.columns,
            )
        return pd.Series([str(x.values[0][i]) for i in range(x.shape[1])], index=x.columns)

    df.columns = pd.MultiIndex.from_tuples(
        [tuple(col.split(" ")) if ("avg" in col or "sem" in col) else (col,)
         for col in df.columns]
    )
    return df.T.groupby(level=0).apply(_fmt_pair).T


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_group_header(dataset: str, side: str, label_name: str, resampling: str) -> None:
    """Render a styled HTML section header for a result group."""
    display(HTML(
        f"""
        <div style='background-color:#ffe6e6; padding:18px; margin:10px 0; border-radius:8px;'>
            <h2 style='color:#b30000; margin:0; font-size:2em;'>
            Results for Dataset: <i>{dataset}</i>, Side: <i>{side}</i>,
            Label: <i>{label_name}</i>, Resampling: <i>{resampling}</i>
            </h2>
        </div>
        """
    ))
