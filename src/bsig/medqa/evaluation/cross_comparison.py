"""Cross-LLM, cross-domain, and condition-by-condition comparison helpers.

Convenience functions wrapping ``bsig.core.evaluation.deferral_auc``
for the common analysis patterns:

- ``cross_llm_comparison``: same dataset, multiple LLMs.
- ``cross_domain_comparison``: same condition, multiple datasets.
- ``condition_comparison``: same dataset, multiple conditions.

The third one is the methods-paper-headline analysis ("does Condition
C beat B beats A on deferral-curve AUC?") and is the load-bearing
function of this module.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from bsig.core.evaluation import deferral_auc


# Default per-condition score-column mapping: A and B store their
# deferral signal directly on ConditionResult / runner-built DataFrame;
# C's deferral signal is the composite from compute_signatures.
_DEFAULT_CONDITION_SCORE_COLUMNS: Mapping[str, str] = {
    "A": "deferral_signal",
    "B": "deferral_signal",
    "C": "composite",
}


def cross_llm_comparison(
    scores_per_llm: Mapping[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    target_column: str,
    score_column: str = "composite",
    compute_ci: bool = True,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Per-LLM deferral_auc on the same dataset.

    Returns DataFrame: ``llm_name, roc_auc, pr_auc, n_pos, n_neg,
    n_dropped`` (+ CI columns if ``compute_ci=True``).
    """
    rows = []
    for llm_name, scores in scores_per_llm.items():
        per = deferral_auc(
            scores,
            ground_truth,
            target_column,
            score_columns=(score_column,),
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        per.insert(0, "llm_name", llm_name)
        rows.append(per.drop(columns=["score_column"]))
    return pd.concat(rows, ignore_index=True).sort_values("llm_name").reset_index(
        drop=True
    )


def cross_domain_comparison(
    scores_per_domain: Mapping[str, pd.DataFrame],
    ground_truths_per_domain: Mapping[str, pd.DataFrame],
    target_column: str,
    score_column: str = "composite",
    compute_ci: bool = True,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Per-domain deferral_auc with separate ground truth per domain.

    Returns DataFrame: ``domain_name, roc_auc, pr_auc, n_pos, n_neg,
    n_dropped`` (+ CI columns).

    For MedQA-vs-MMLU comparison: ``scores_per_domain = {"medqa":
    medqa_scores, "mmlu": mmlu_scores}`` and similarly for ground
    truths. Each domain has its own (scores, ground_truth) pair
    because the trajectories don't overlap.
    """
    if set(scores_per_domain) != set(ground_truths_per_domain):
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError(
            "scores_per_domain and ground_truths_per_domain must have "
            "the same keys; got "
            f"{sorted(scores_per_domain)} vs {sorted(ground_truths_per_domain)}"
        )
    rows = []
    for domain in scores_per_domain:
        per = deferral_auc(
            scores_per_domain[domain],
            ground_truths_per_domain[domain],
            target_column,
            score_columns=(score_column,),
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        per.insert(0, "domain_name", domain)
        rows.append(per.drop(columns=["score_column"]))
    return pd.concat(rows, ignore_index=True).sort_values("domain_name").reset_index(
        drop=True
    )


def condition_comparison(
    scores_per_condition: Mapping[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    target_column: str,
    score_columns_per_condition: Mapping[str, str] | None = None,
    subset_mask: pd.Series | None = None,
    compute_ci: bool = True,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Per-condition deferral_auc on the same dataset.

    THE methods-paper-headline analysis: "does Condition C beat B
    beats A on deferral-curve AUC?"

    Default score-column mapping:
    - 'A': 'deferral_signal' (constant 0.5 — the no-signal baseline)
    - 'B': 'deferral_signal' (1 - confidence)
    - 'C': 'composite' (signature composite from compute_signatures)

    Caller can override via ``score_columns_per_condition``.

    ``subset_mask`` (optional): boolean Series indexed identically to
    ``ground_truth`` that filters which rows participate in the
    comparison. Used by stage-4 analyses for the F7-relevant subset
    ("cases where Condition B is uninformative") per the eventual
    ADR-0006 gate-metric revision. None means use all data.

    Returns DataFrame: ``condition_id, score_column, roc_auc, pr_auc,
    n_pos, n_neg, n_dropped`` (+ CI columns).
    """
    score_cols = (
        dict(score_columns_per_condition)
        if score_columns_per_condition is not None
        else dict(_DEFAULT_CONDITION_SCORE_COLUMNS)
    )

    # Apply subset mask to ground truth if provided
    if subset_mask is not None:
        gt_filtered = ground_truth.loc[subset_mask].reset_index(drop=True)
    else:
        gt_filtered = ground_truth

    rows = []
    for cond_id, scores in scores_per_condition.items():
        score_col = score_cols.get(cond_id)
        if score_col is None:
            from bsig.core.evaluation import EvaluationError
            raise EvaluationError(
                f"No score_column mapping for condition {cond_id!r}; "
                f"got {sorted(score_cols)}"
            )
        # Filter scores to subset trajectory_ids if mask was applied
        if subset_mask is not None:
            ids_in_subset = set(gt_filtered["trajectory_id"])
            scores_filtered = scores[scores["trajectory_id"].isin(ids_in_subset)]
        else:
            scores_filtered = scores

        per = deferral_auc(
            scores_filtered,
            gt_filtered,
            target_column,
            score_columns=(score_col,),
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        per.insert(0, "condition_id", cond_id)
        rows.append(per)

    return pd.concat(rows, ignore_index=True).sort_values("condition_id").reset_index(
        drop=True
    )
