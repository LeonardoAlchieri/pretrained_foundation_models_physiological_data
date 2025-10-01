import os
import pickle
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from IPython.display import HTML, display
from scipy import stats
from yaml import safe_load
from jmetal.lab.statistical_test.functions import (
    friedman_aligned_rank_test,
    friedman_aligned_ph_test,
)


pd.set_option("display.max_columns", 100)


def friedmann_nemenyi_test(
    unravelled_detailed_results: dict[str : dict[tuple[str, str], list[float]]],
) -> None:
    for metric in unravelled_detailed_results.keys():
        metric_df = pd.DataFrame(unravelled_detailed_results[metric])
        metric_vals = [metric_df[col].values for col in metric_df.columns]
        out = stats.friedmanchisquare(*metric_vals)
        print(
            f"Friedman Test {metric}: statistic={out.statistic:.3f}, pvalue={out.pvalue:.3f}"
        )
        metric_vals = np.array(metric_vals).T
        out = sp.posthoc_nemenyi_friedman(metric_vals)
        print("Nemenyi post-hoc test")
        feature_names = metric_df.columns
        out.index = feature_names
        out.columns = feature_names
        out = out.sort_index(axis=0).sort_index(axis=1)
        display(out)


def aligned_friedmann_holm_test(
    unravelled_detailed_results: dict[str : dict[tuple[str, str], list[float]]],
) -> None:
    for metric in unravelled_detailed_results.keys():
        metric_df = pd.DataFrame(unravelled_detailed_results[metric])
        # metric_vals = np.asarray([metric_df[col].values for col in metric_df.columns])
        metric_vals = metric_df.values

        out = friedman_aligned_rank_test(metric_vals)
        chi2_stat = out.loc["Aligned Rank stat"].iloc[0]
        p_value = out.loc["p-value"].iloc[0]
        print(f"Aligned-rank Friedman χ² {metric} = {chi2_stat:.3f}, p = {p_value:.3f}")

        z_vals, out, _ = friedman_aligned_ph_test(
            metric_vals, apv_procedure="Holm"  # Holm step-down correction
        )
        print("Holm post-hoc test")
        feature_names = metric_df.columns
        out.index = feature_names
        out.columns = feature_names
        out = out.sort_index(axis=0).sort_index(axis=1)
        display(out)


def present_results(
    paths: Generator,
    val_method: str = "lopo",
    remove_xgboost: bool = False,
    remove_chronos_small_from_test: bool = False,
    which_test: str = "friedmann-nemenyi",
    test_args: dict = {},
    average_over_seed: bool = False,
) -> None:
    results = []
    for reports_path in paths:
        report = pd.read_csv(reports_path, index_col=0)
        conf = safe_load(open(reports_path.parent / ".hydra/config.yaml"))
        if conf["validation_method"]["_target_"].split(".")[-1].lower() != val_method:
            continue
        model_name: str = conf["model"]["model"]["_target_"].split(".")[-1]
        features_name = (
            conf["feature_extractor"]["_target_"].split(".")[-1]
            if "model_name" not in conf["feature_extractor"]
            else conf["feature_extractor"]["model_name"]
        )
        validation_method = conf["validation_method"]["_target_"].split(".")[-1]
        if "aggregator" not in conf:
            aggregator = "MeanTimeAggregator"
        else:
            aggregator = (
                conf["aggregator"]["_target_"].split(".")[-1]
                if "_target_" in conf["aggregator"]
                else None
            )
        report_results = {}
        for col in report.columns:
            report_results[f"{col} avg"] = report[col].mean()
            report_results[f"{col} sem"] = report[col].sem() * 1.98  # 95% CI

        seed = conf["seed"]
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
        resampling = (
            conf["resampling"]["_target_"].split(".")[-1]
            if "resampling" in conf
            else "None"
        )
        resampling = resampling if resampling != "NoUnderSampler" else "None"
        # Collect results in a list of dicts
        if remove_xgboost and model_name == "XGBClassifier":
            continue
        results.append(
            {
                "Dataset": dataset,
                "Side": side,
                "Seed": seed,
                "Label Name": label_name,
                "Model": model_name,
                "Resampling": resampling,
                "Features": features_name,
                "Aggregator": aggregator,
                "Validation": validation_method,
                "Detailed Report": report,
                **report_results,
            }
        )

    # After the loop, display as a table
    df_results = pd.DataFrame(results)
    for (dataset, side, label_name, resampling), group in df_results.groupby(
        ["Dataset", "Side", "Label Name", "Resampling"]
    ):
        try:
            display(
                HTML(
                    f"""
                <div style='background-color:#ffe6e6; padding:18px; margin:10px 0; border-radius:8px;'>
                    <h2 style='color:#b30000; margin:0; font-size:2em;'>
                    Results for Dataset: <i>{dataset}</i>, Side: <i>{side}</i>, Label: <i>{label_name}</i>, Resampling: <i>{resampling}</i>
                    </h2>
                </div>
                """
                )
            )
            grouped_data = (
                group.sort_values(by=["Model", "Features", "Aggregator"])
                .drop(columns=["Detailed Report"])
                .drop_duplicates()
            )
            # display(grouped_data)

            def cleanup_res(x):
                if len(x) > 1:
                    return pd.Series(
                        [
                            "%.2f ± %.2f"
                            % (np.round(x.values[0][i], 2), np.round(x.values[1][i], 2))
                            for i in range(x.shape[1])
                        ],
                        index=x.columns,
                    )
                else:
                    return pd.Series(
                        [f"{x.values[0][i]}" for i in range(x.shape[1])],
                        index=x.columns,
                    )

            grouped_data.columns = pd.MultiIndex.from_tuples(
                [
                    (
                        tuple(col.split(" "))
                        if "avg" in col or "sem" in col
                        else tuple([col])
                    )
                    for col in grouped_data.columns
                ]
            )

            grouped_data = grouped_data.T.groupby(level=0).apply(cleanup_res).T

            def mean_of_mean(x) -> float:
                new_df = pd.Series(
                    index=list([col for col in x.columns if col != "Side"]), dtype=str
                )
                for col in [
                    "accuracy_score",
                    "balanced_accuracy_score",
                    "f1_score",
                    "matthews_corrcoef",
                    "roc_auc_score",
                    "precision_score",
                    "recall_score",
                ]:
                    new_df[col] = "%.2f ± %.2f" % (
                        x[col].apply(lambda v: float(v.split(" ± ")[0])).mean(),
                        x[col]
                        .apply(lambda v: float(v.split(" ± ")[1]))
                        .apply(lambda v: v**2)
                        .sum()
                        ** 0.5
                        / len(x),
                    )
                return new_df

            grouped_data = grouped_data.groupby(
                [
                    "Aggregator",
                    "Dataset",
                    "Features",
                    "Label Name",
                    "Model",
                    "Resampling",
                    "Side",
                    "Validation",
                ]
            ).apply(mean_of_mean, include_groups=False)
            display(grouped_data)
            grouped_data = grouped_data.reset_index()
            grouped_data = grouped_data.drop(
                columns=[
                    "accuracy_score",
                    "Dataset",
                    "Side",
                    "Label Name",
                    "Resampling",
                    "Validation",
                ]
            )
            grouped_data = grouped_data.rename(
                columns={
                    "balanced_accuracy_score": "Balanced Accuracy",
                    "f1_score": "F1",
                    "matthews_corrcoef": "MCC",
                    "roc_auc_score": "ROC AUC",
                    "precision_score": "Precision",
                    "recall_score": "Recall",
                }
            )

            # save to latex
            latex_path = (
                Path("../tables_latex")
                / f"results_{dataset}_{side}_{label_name}_{resampling}_{validation_method}.tex"
            )
            with open(latex_path, "w") as f:
                f.write(grouped_data.to_latex())

            if "aggregator" in test_args.keys() and group["Aggregator"].nunique() > 2:
                group = group[
                    (group["Aggregator"] == test_args["aggregator"])
                    | (group["Aggregator"].isnull())
                ]

            unravelled_detailed_results = {
                metric: {} for metric in group["Detailed Report"].iloc[0].columns
            }
            for i, model_results in group.iterrows():
                for metric in model_results["Detailed Report"].columns:
                    cv_results = model_results["Detailed Report"][metric]
                    unravelled_detailed_results[metric][
                        (model_results["Model"], model_results["Features"])
                    ] = cv_results

            if which_test == "friedmann-nemenyi":
                friedmann_nemenyi_test(unravelled_detailed_results)
            elif which_test == "alignedfriedmann-holm":
                aligned_friedmann_holm_test(unravelled_detailed_results)
            else:
                raise ValueError(f"Unknown test: {which_test}")
        except Exception as e:
            print(
                f"Error processing group {dataset}, {side}, {label_name}, {resampling}: {e}"
            )
            continue

    return results


def get_results_path(
    results_paths: list[str] = ["../outputs/", "../outputs_adula/"],
    specific_path: str | None = None,
    only_last: bool = False,
) -> list[Path]:
    all_results = []
    for result in results_paths:
        if not os.path.exists(result):
            raise ValueError(f"Results path {result} does not exist.")
        all_results += list(Path(result).glob("*/*/*/reports.csv"))

    if specific_path is not None:
        all_results = [val for val in all_results if specific_path in str(val)]

    if only_last:
        all_results_unpacked = {}
        for val in all_results:
            # if "usilaughs-right" not in val.parts:
            #     continue
            parts = val.parts
            time_key = parts[3]
            key = tuple(el for el in parts if el != time_key)
            if key in all_results_unpacked.keys():
                if time_key > all_results_unpacked[key].parts[3]:
                    all_results_unpacked[key] = val
                else:
                    continue
            else:
                all_results_unpacked[key] = val
        all_results = sorted(list(all_results_unpacked.values()))

    return all_results
