"""Failure-mode diagnostic table.

Per-question table for inspecting the highest-signature trajectories.
The methods-paper worked-example narrative comes from this view: "show
me the cases the framework flagged most strongly — are they actually
the wrong-answer cases?"
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import rankdata


def failure_mode_table(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    score_column: str = "composite",
    top_n: int = 50,
) -> pd.DataFrame:
    """Per-question diagnostic table sorted by score descending.

    Returns DataFrame with columns:
    - ``trajectory_id`` (str)
    - ``score`` (float)
    - ``score_percentile`` (float in [0, 1] — empirical CDF rank)
    - ``target`` (int — the ground-truth target_column value)
    - ``high_score_correct_outcome`` (bool — convenience: ``target == 1``,
      i.e., this high-score case was indeed a defer-positive)

    Sorted by ``score`` descending; first ``top_n`` rows returned.

    Useful narrative: "the top 10 highest-signature cases — are they
    mostly target=1 (genuine defer-positives)? If yes, the framework is
    catching what it should. If no, investigate why high-signature
    cases are correct-answer trajectories."
    """
    if "trajectory_id" not in scores.columns:
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError("scores must have 'trajectory_id' column")
    if score_column not in scores.columns:
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError(
            f"score_column={score_column!r} not in scores columns: "
            f"{list(scores.columns)}"
        )
    if "trajectory_id" not in ground_truth.columns:
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError("ground_truth must have 'trajectory_id' column")
    if target_column not in ground_truth.columns:
        from bsig.core.evaluation import EvaluationError
        raise EvaluationError(
            f"target_column={target_column!r} not in ground_truth columns: "
            f"{list(ground_truth.columns)}"
        )

    joined = scores.merge(
        ground_truth[["trajectory_id", target_column]],
        on="trajectory_id",
        how="inner",
    )
    if len(joined) == 0:
        return pd.DataFrame(
            {
                "trajectory_id": pd.Series(dtype=object),
                "score": pd.Series(dtype="float64"),
                "score_percentile": pd.Series(dtype="float64"),
                "target": pd.Series(dtype="int64"),
                "high_score_correct_outcome": pd.Series(dtype="bool"),
            }
        )

    score_values = joined[score_column].to_numpy()
    n = len(score_values)
    if n == 1:
        percentiles = [0.5]
    else:
        ranks = rankdata(score_values, method="average")
        percentiles = ranks / n

    out = pd.DataFrame(
        {
            "trajectory_id": joined["trajectory_id"].to_numpy(),
            "score": score_values,
            "score_percentile": percentiles,
            "target": joined[target_column].astype(int).to_numpy(),
        }
    )
    out["high_score_correct_outcome"] = out["target"] == 1
    out = out.sort_values("score", ascending=False).head(top_n).reset_index(
        drop=True
    )
    return out
