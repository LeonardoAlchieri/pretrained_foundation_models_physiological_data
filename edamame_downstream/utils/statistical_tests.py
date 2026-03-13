"""
Non-parametric statistical tests for comparing multiple classifiers across datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from IPython.display import display
from scipy import stats
from jmetal.lab.statistical_test.functions import (
    friedman_aligned_ph_test,
    friedman_aligned_rank_test,
)


def friedmann_nemenyi_test(
    unravelled_detailed_results: dict[str, dict[tuple[str, str], list[float]]],
) -> None:
    """Run a Friedman test followed by Nemenyi post-hoc pairwise comparisons.

    Args:
        unravelled_detailed_results: Nested dict keyed by metric name, then by
            ``(model_name, features_name)`` tuples, with per-fold score lists
            as values.
    """
    for metric, results in unravelled_detailed_results.items():
        metric_df = pd.DataFrame(results)
        metric_vals = [metric_df[col].values for col in metric_df.columns]
        out = stats.friedmanchisquare(*metric_vals)
        print(f"Friedman Test {metric}: statistic={out.statistic:.3f}, pvalue={out.pvalue:.3f}")

        out = sp.posthoc_nemenyi_friedman(np.array(metric_vals).T)
        print("Nemenyi post-hoc test")
        out.index = out.columns = metric_df.columns
        display(out.sort_index(axis=0).sort_index(axis=1))


def aligned_friedmann_holm_test(
    unravelled_detailed_results: dict[str, dict[tuple[str, str], list[float]]],
) -> None:
    """Run an aligned-rank Friedman test followed by Holm post-hoc comparisons.

    Args:
        unravelled_detailed_results: Same structure as for
            :func:`friedmann_nemenyi_test`.
    """
    for metric, results in unravelled_detailed_results.items():
        metric_df = pd.DataFrame(results)
        metric_vals = metric_df.values

        rank_out = friedman_aligned_rank_test(metric_vals)
        chi2_stat = rank_out.loc["Aligned Rank stat"].iloc[0]
        p_value = rank_out.loc["p-value"].iloc[0]
        print(f"Aligned-rank Friedman χ² {metric} = {chi2_stat:.3f}, p = {p_value:.3f}")

        _, ph_out, _ = friedman_aligned_ph_test(metric_vals, apv_procedure="Holm")
        print("Holm post-hoc test")
        ph_out.index = ph_out.columns = metric_df.columns
        display(ph_out.sort_index(axis=0).sort_index(axis=1))
