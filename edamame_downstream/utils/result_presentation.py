"""
High-level result presentation functions for classification experiments.

Public API
----------
present_results            -- load reports.csv files and display a summary table
present_results_bootstrapz -- same, but using pre-computed bootstrap CIs
get_results_path           -- discover report files under an output directory tree
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm
from yaml import safe_load

from ._result_helpers import (
    CONFIG_COLUMNS,
    KNOWN_METRICS,
    METRIC_RENAME,
    apply_filters,
    dedup_results,
    display_group_header,
    format_avg_sem_columns,
    parse_run_metadata,
)
from .statistical_tests import aligned_friedmann_holm_test, friedmann_nemenyi_test

pd.set_option("display.max_columns", 100)


# ---------------------------------------------------------------------------
# Internal display helpers
# ---------------------------------------------------------------------------


def _mean_of_mean(x: pd.DataFrame) -> pd.Series:
    """Aggregate seed-level '%.2f ± %.2f' strings into a single combined string."""
    new_df = pd.Series(index=[col for col in x.columns if col != "Side"], dtype=str)
    for col in KNOWN_METRICS:
        if col not in x.columns:
            continue
        means = x[col].apply(lambda v: float(v.split(" ± ")[0]))
        sems = x[col].apply(lambda v: float(v.split(" ± ")[1]))
        combined_sem = (sems.apply(lambda v: v**2).sum() ** 0.5) / len(x)
        new_df[col] = "%.2f ± %.2f" % (means.mean(), combined_sem)
    return new_df


def _save_latex(df: pd.DataFrame, dataset: str, side: str, label_name: str,
                resampling: str, validation_method: str, prefix: str = "results") -> None:
    latex_path = (
        Path("../tables_latex")
        / f"{prefix}_{dataset}_{side}_{label_name}_{resampling}_{validation_method}.tex"
    )
    with open(latex_path, "w") as f:
        f.write(df.to_latex())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def present_results(
    paths: Generator,
    val_method: str = "lopo",
    remove_xgboost: bool = False,
    remove_chronos_small_from_test: bool = False,
    which_test: str = "friedmann-nemenyi",
    test_args: dict = {},
    average_over_seed: bool = False,
    show_tests: bool = True,
    filters: dict[str, str | list[str] | None] = None,
    exclude_filters: dict[str, str | list[str] | None] = None,
) -> tuple:
    """Present results from classification experiments.

    Args:
        paths: Generator of paths to ``reports.csv`` files.
        val_method: Validation method to filter by (e.g. ``"lopo"``).
        remove_xgboost: Exclude XGBoost results.
        which_test: Statistical test – ``"friedmann-nemenyi"`` or
            ``"alignedfriedmann-holm"``.
        test_args: Extra arguments forwarded to the statistical test.
        show_tests: Whether to run and display statistical tests.
        filters: Positive column filters – see :func:`apply_filters`.
        exclude_filters: Negative column filters – see :func:`apply_filters`.
    """
    filters = filters or {}
    exclude_filters = exclude_filters or {}

    results = []
    for reports_path in tqdm(list(paths), desc="Processing reports"):
        conf = safe_load(open(reports_path.parent / ".hydra/config.yaml"))
        meta = parse_run_metadata(reports_path, conf, val_method, remove_xgboost)
        if meta is None:
            continue
        report = pd.read_csv(reports_path, index_col=0)
        report_results = {
            f"{col} avg": report[col].mean() for col in report.columns
        } | {
            f"{col} sem": report[col].sem() * 1.98 for col in report.columns  # 95% CI
        }
        results.append({**meta, "Detailed Report": report, **report_results})

    df_results = apply_filters(pd.DataFrame(results), filters, exclude_filters)
    if df_results.empty:
        print("No results match the specified filters.")
        return []
    df_results = dedup_results(df_results)

    grouped_data_return = None
    for (dataset, side, label_name, resampling), group in df_results.groupby(
        ["Dataset", "Side", "Label Name", "Resampling"]
    ):
        try:
            display_group_header(dataset, side, label_name, resampling)
            grouped_data = (
                group.sort_values(by=["Model", "Features", "Aggregator"])
                .drop(columns=["Detailed Report", "Timestamp"])
                .drop_duplicates()
            )
            grouped_data = format_avg_sem_columns(grouped_data)
            grouped_data = grouped_data.groupby(CONFIG_COLUMNS).apply(
                _mean_of_mean, include_groups=False
            )
            display(grouped_data)
            grouped_data_return = grouped_data.copy()

            export = grouped_data.reset_index().drop(
                columns=[c for c in ["accuracy_score", "Dataset", "Side", "Label Name",
                                     "Resampling", "Validation", "Subsample Train Set"]
                         if c in grouped_data.reset_index().columns],
            ).rename(columns=METRIC_RENAME)
            _save_latex(export, dataset, side, label_name, resampling,
                        group["Validation"].iloc[0])

            if show_tests:
                if "aggregator" in test_args and group["Aggregator"].nunique() > 2:
                    group = group[
                        (group["Aggregator"] == test_args["aggregator"])
                        | group["Aggregator"].isnull()
                    ]
                unravelled: dict[str, dict] = {
                    m: {} for m in group["Detailed Report"].iloc[0].columns
                }
                for _, row in group.iterrows():
                    for metric in row["Detailed Report"].columns:
                        unravelled[metric][(row["Model"], row["Features"])] = (
                            row["Detailed Report"][metric]
                        )
                if which_test == "friedmann-nemenyi":
                    friedmann_nemenyi_test(unravelled)
                elif which_test == "alignedfriedmann-holm":
                    aligned_friedmann_holm_test(unravelled)
                else:
                    raise ValueError(f"Unknown test: {which_test}")

        except Exception as e:
            grouped_data_return = None
            print(f"Error processing group {dataset}, {side}, {label_name}, {resampling}: {e}")

    return results, grouped_data_return


def present_results_bootstrapz(
    paths: Generator,
    val_method: str = "lopo",
    remove_xgboost: bool = False,
    filters: dict[str, str | list[str] | None] = None,
    exclude_filters: dict[str, str | list[str] | None] = None,
) -> tuple:
    """Present results using pre-computed bootstrap confidence intervals.

    Reads ``bootstrap_reports.csv`` (columns: ``metric``, ``mean``, ``se``,
    ``ci_lower``, ``ci_upper``) from the same directory as each ``reports.csv``
    supplied via *paths*.  The half-CI width is stored as ``sem`` so the same
    display pipeline can be reused.

    Args:
        paths: Generator of paths to ``reports.csv`` files (used to locate the
            sibling ``bootstrap_reports.csv`` and the Hydra config).
        val_method: Validation method to filter by (e.g. ``"lopo"``).
        remove_xgboost: Exclude XGBoost results.
        filters: Positive column filters – see :func:`apply_filters`.
        exclude_filters: Negative column filters – see :func:`apply_filters`.
    """
    filters = filters or {}
    exclude_filters = exclude_filters or {}

    results = []
    for reports_path in tqdm(list(paths), desc="Processing bootstrap reports"):
        bootstrap_path = reports_path.parent / "bootstrap_reports.csv"
        if not bootstrap_path.exists():
            continue
        conf = safe_load(open(reports_path.parent / ".hydra/config.yaml"))
        meta = parse_run_metadata(reports_path, conf, val_method, remove_xgboost)
        if meta is None:
            continue
        report = pd.read_csv(bootstrap_path).set_index("metric")
        report_results = {
            f"{m} avg": report.loc[m, "mean"] for m in report.index
        } | {
            f"{m} sem": (report.loc[m, "ci_upper"] - report.loc[m, "ci_lower"]) / 2
            for m in report.index
        }
        results.append({**meta, **report_results})

    df_results = apply_filters(pd.DataFrame(results), filters, exclude_filters)
    if df_results.empty:
        print("No results match the specified filters.")
        return []
    df_results = dedup_results(df_results)

    grouped_data_return = None
    for (dataset, side, label_name, resampling), group in df_results.groupby(
        ["Dataset", "Side", "Label Name", "Resampling"]
    ):
        try:
            display_group_header(dataset, side, label_name, resampling)
            grouped_data = (
                group.sort_values(by=["Model", "Features", "Aggregator"])
                .drop(columns=["Timestamp"])
                .drop_duplicates()
            )
            # Multiply by 100 so values are shown as percentages
            grouped_data = format_avg_sem_columns(grouped_data, scale=100.0)
            grouped_data_return = grouped_data.copy()
            display(grouped_data)

        except Exception as e:
            grouped_data_return = None
            print(f"Error processing group {dataset}, {side}, {label_name}, {resampling}: {e}")

    return results, grouped_data_return


def get_results_path(
    results_paths: list[str] = ["../outputs/", "../outputs_adula/"],
    specific_path: str | None = None,
    only_last: bool = False,
    result_name: str = "reports.csv",
) -> list[Path]:
    """Discover result files under one or more output directory trees.

    Args:
        results_paths: Root directories to search.
        specific_path: If given, keep only paths that contain this substring.
        only_last: Keep only the most-recent run for each configuration.
        result_name: Filename to search for (default ``"reports.csv"``).
    """
    all_results: list[Path] = []
    for root in results_paths:
        if not os.path.exists(root):
            raise ValueError(f"Results path {root} does not exist.")
        # Sweep  : outputs/DATASET-SIDE/multirun_DATE/RUN_ID/reports.csv (4 levels)
        all_results += list(Path(root).glob(f"*/*/*/{result_name}"))
        # Single : outputs/DATE/TIME/reports.csv (3 levels)
        all_results += list(Path(root).glob(f"*/*/{result_name}"))

    if specific_path is not None:
        all_results = [p for p in all_results if specific_path in str(p)]

    if only_last:
        unpacked: dict[tuple, Path] = {}
        for val in all_results:
            parts = val.parts
            time_key = parts[3]
            key = tuple(p for p in parts if p != time_key)
            if key not in unpacked or time_key > unpacked[key].parts[3]:
                unpacked[key] = val
        all_results = sorted(unpacked.values())

    return all_results
