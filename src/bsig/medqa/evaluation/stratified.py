"""Stratified deferral_auc.

Wraps ``bsig.core.evaluation.deferral_auc`` to produce per-stratum
metrics. For MedQA the natural stratum is ``usmle_step`` (step1 vs
step2&3 per ``MedQARawRecord.usmle_step``, propagated through to
``Outcome.secondary_labels["usmle_step"]`` by the ground-truth
extractor and onto the ground-truth DataFrame).

Useful for "does the framework's signal transfer between USMLE
step1 and step2&3?" — a stage-4 stratification analysis.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from bsig.core.evaluation import deferral_auc


def stratified_deferral_auc(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    stratum_column: str,
    score_columns: Sequence[str] = ("composite",),
    compute_ci: bool = False,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Per-stratum deferral_auc on the same scores DataFrame.

    Returns long-format DataFrame: ``stratum_value, score_column,
    roc_auc, pr_auc, n_pos, n_neg, n_dropped`` (+ CI columns if
    ``compute_ci=True``).

    Strata are read from ``ground_truth[stratum_column]``. Each unique
    value produces one set of rows in the output. Strata with no
    positives or no negatives are skipped (deferral_auc would raise);
    a warning column ``skipped: bool`` is added per row.
    """
    if stratum_column not in ground_truth.columns:
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError(
            f"stratum_column={stratum_column!r} not in ground_truth columns: "
            f"{list(ground_truth.columns)}"
        )

    rows: list[pd.DataFrame] = []
    for stratum_value, gt_subset in ground_truth.groupby(stratum_column):
        # Filter scores by trajectory_ids in this stratum
        ids_in_stratum = set(gt_subset["trajectory_id"])
        scores_subset = scores[scores["trajectory_id"].isin(ids_in_stratum)]

        # Skip strata with no class balance
        labels = gt_subset[target_column].to_numpy()
        if (labels == 1).sum() == 0 or (labels == 0).sum() == 0:
            continue

        per_stratum = deferral_auc(
            scores_subset,
            gt_subset,
            target_column,
            score_columns=score_columns,
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        per_stratum.insert(0, stratum_column, stratum_value)
        rows.append(per_stratum)

    if not rows:
        # All strata skipped — return empty DataFrame with expected columns
        return pd.DataFrame(
            {
                stratum_column: pd.Series(dtype=object),
                "score_column": pd.Series(dtype=object),
                "roc_auc": pd.Series(dtype="float64"),
                "pr_auc": pd.Series(dtype="float64"),
                "n_pos": pd.Series(dtype="int64"),
                "n_neg": pd.Series(dtype="int64"),
                "n_dropped": pd.Series(dtype="int64"),
            }
        )

    return pd.concat(rows, ignore_index=True).sort_values(
        [stratum_column, "score_column"]
    ).reset_index(drop=True)
