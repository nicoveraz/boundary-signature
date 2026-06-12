"""Evaluation: deferral curves, AUC, calibration, operating points.

The final downstream module of the algorithmic core. Operates on
already-computed signature scores (per-trajectory composite plus
component values from ``compute_signatures``) and ground-truth
outcomes. Produces the metrics that constitute the chest-pain gate
experiment's headline numbers (AUC, calibration, operating-point
sensitivity/specificity at clinically meaningful deferral fractions).

**Conventions**

- **Higher score = defer.** Every public function assumes higher score
  values trigger deferral to a specialist. The composite from
  ``compute_signatures`` is rank-percentile-normalized to satisfy this;
  raw components (entropy_plateau as signed slope, voi_flatness as
  mean ``|VoI|``, distance_from_trajectory as max k-NN distance) are
  also constructed so higher = boundary signal. Components with inverse
  semantics must be negated by the caller before passing here.

- **Sign-aware AUC reporting.** ``deferral_auc`` returns both the raw
  ROC-AUC (under the higher-score-defers convention) and the
  ``roc_auc_sign_aware = max(roc_auc, 1 - roc_auc)`` magnitude, plus
  ``roc_auc_direction`` ("greater" when raw AUC ≥ 0.5; "less" when
  raw AUC < 0.5). Sign-aware reporting prevents readers from being
  misled by raw AUCs near 0.5 — a raw AUC of 0.469 reads as "below
  chance" but its sign-aware magnitude is 0.531, identifying weak
  but present signal that fires in the *opposite* direction to the
  framework's higher-score-defers convention. When ``compute_ci=True``,
  bootstrap CIs are also reported in sign-aware form, oriented to
  the point estimate's direction (so the CI does NOT clip at 0.5;
  resamples that flip direction appear as sign-aware values < 0.5,
  honestly reflecting sampling uncertainty about both magnitude and
  direction). PR-AUC is reported raw — the 1-AUC symmetry is a
  ROC-AUC property; PR-AUC near base rate has a different
  no-signal interpretation that doesn't admit the same flip.

- **``target_column`` is 1 = defer-positive, 0 = defer-negative.**
  The class evaluation should detect (the rare class for chest-pain
  "needs consultation"; the incorrect-answer class for MedQA) is
  encoded as 1. Inverse-semantics labels (e.g., 1 = correct) require
  the caller to flip them. Documented per-function as well.

- **Caller responsibility for holdout.** The signature scores being
  evaluated must come from trajectories properly decoupled from the
  FAISS indices used at compute_signatures time (held-out test set or
  leave-one-out). Evaluation does not enforce this; failing to enforce
  it produces optimistic curves. See ``compute_signatures`` docstring
  and ``experiments/chest_pain_min/`` for the holdout convention.

**Public functions**

- ``deferral_curve``: coarse threshold grid (default 20 points)
  producing the data for plot rendering.
- ``deferral_auc``: exact AUC (ROC + PR), with optional bootstrap CIs.
- ``calibration_metrics``: ECE + MCE + Brier score per score column.
  Requires score values in [0, 1]; pre-normalize raw components if
  passing them.
- ``operating_points``: sensitivity/specificity/PPV/NPV at specified
  deferral fractions.
- ``component_decomposition_table``: convenience wrapper around
  ``deferral_auc`` producing per-component AUC + CIs in paper-table
  format.

**Determinism**

Bootstrap CIs use ``numpy.random.default_rng(random_seed)`` (modern,
non-global-state). Default seed 42 produces reproducible results
across runs. For development iteration, override
``n_bootstrap=500`` (rough but fast); the gate-experiment final run
uses the default 5000.

**Persistence**

Evaluation outputs are cheap to recompute (seconds, not minutes), so
``core/evaluation.py`` does not provide save/load functions. The
experiment runner is responsible for persisting outputs that need to
land in the methods paper or supplementary materials (typically as
CSVs or formatted tables).

**Low-N runs and class-imbalance failures**

The metrics defined here are mathematically undefined when ground
truth has only one class (no positives or no negatives). The
functions raise ``EvaluationError`` in that case — correct framework
behavior, since proceeding would produce meaningless numbers. But
this can fire legitimately at low N: a small smoke run where the
model happens to answer all questions correctly will have
``target == 0`` for every row of the wrong-answer ground truth.

Runner scripts should detect this case BEFORE calling evaluation
functions and exit gracefully with informative output rather than
letting the framework's exception surface as a stack trace. Pattern:

    n_pos = int((ground_truth[target_column] == 1).sum())
    n_neg = int((ground_truth[target_column] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print("WARNING: ground truth has only one class — "
              "condition_comparison cannot run. Common at small N.")
        return 0

See ``experiments/medqa_generalization/scripts/03_pipeline_validation_ollama.py``
for the canonical implementation.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ---- Errors and warnings ----


class EvaluationError(Exception):
    """Base class for evaluation errors."""


class EvaluationWarning(UserWarning):
    """Filterable warning category for degenerate-but-recoverable cases
    (e.g., dropping NaN scores). Callers can filter via
    ``warnings.simplefilter("ignore", EvaluationWarning)`` for known
    cases."""


# ---- Shared helpers ----


def _validate_ground_truth_classes(
    ground_truth: pd.DataFrame, target_column: str
) -> None:
    """Raise if ``target_column`` has no positive or no negative cases.

    Both classes must be present for AUC, calibration, and operating
    points to be defined. The empty-input case (zero rows) is handled
    upstream — this checks class balance, not row count.
    """
    if target_column not in ground_truth.columns:
        raise EvaluationError(
            f"target_column={target_column!r} not in ground_truth columns: "
            f"{list(ground_truth.columns)}"
        )
    labels = ground_truth[target_column].to_numpy()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise EvaluationError(
            f"ground_truth has no positive cases for "
            f"target_column={target_column!r}; AUC, calibration, and "
            f"operating-point metrics are undefined"
        )
    if n_neg == 0:
        raise EvaluationError(
            f"ground_truth has no negative cases for "
            f"target_column={target_column!r}; AUC, calibration, and "
            f"operating-point metrics are undefined"
        )


def _join_scores_truth(
    scores: pd.DataFrame, ground_truth: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Inner join scores and ground_truth on ``trajectory_id``; raise
    if the join is empty (no overlapping trajectory IDs).
    """
    if "trajectory_id" not in scores.columns:
        raise EvaluationError("scores must have 'trajectory_id' column")
    if "trajectory_id" not in ground_truth.columns:
        raise EvaluationError("ground_truth must have 'trajectory_id' column")
    if target_column not in ground_truth.columns:
        raise EvaluationError(
            f"target_column={target_column!r} not in ground_truth columns: "
            f"{list(ground_truth.columns)}"
        )

    n_scores = len(scores)
    n_truth = len(ground_truth)
    joined = scores.merge(
        ground_truth[["trajectory_id", target_column]],
        on="trajectory_id",
        how="inner",
    )
    if len(joined) == 0:
        raise EvaluationError(
            f"No trajectories matched between scores (n={n_scores}) and "
            f"ground_truth (n={n_truth}) on trajectory_id"
        )
    return joined


def _drop_nan_and_warn(
    df: pd.DataFrame, score_column: str
) -> tuple[pd.DataFrame, int]:
    """Drop rows with NaN in ``score_column``; warn with count and
    common causes. Returns (cleaned DataFrame, number dropped)."""
    if score_column not in df.columns:
        raise EvaluationError(
            f"score_column={score_column!r} not in scores columns: "
            f"{list(df.columns)}"
        )
    nan_mask = df[score_column].isna()
    n_dropped = int(nan_mask.sum())
    if n_dropped > 0:
        n_total = len(df)
        warnings.warn(
            f"Dropping {n_dropped} trajectories with NaN values in "
            f"score_column={score_column!r} ({n_dropped / n_total:.1%} "
            f"of input). Common causes: missing embeddings, single-state "
            f"trajectories.",
            EvaluationWarning,
            stacklevel=3,
        )
    return df.loc[~nan_mask].reset_index(drop=True), n_dropped


def _empty_deferral_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_column": pd.Series(dtype=object),
            "threshold": pd.Series(dtype=np.float64),
            "deferral_fraction": pd.Series(dtype=np.float64),
            "residual_error_rate": pd.Series(dtype=np.float64),
            "kept_count": pd.Series(dtype=np.int64),
            "deferred_count": pd.Series(dtype=np.int64),
            "total_count": pd.Series(dtype=np.int64),
            "error_rate_overall": pd.Series(dtype=np.float64),
            "n_dropped": pd.Series(dtype=np.int64),
        }
    )


# ---- deferral_curve ----


def deferral_curve(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    score_columns: Sequence[str] = ("composite",),
    n_threshold_points: int = 20,
) -> pd.DataFrame:
    """Coarse-grid deferral curve data for plotting.

    Long-format DataFrame: one row per (score_column, threshold_point).
    The grid steps through deferral fractions evenly from 0% to
    (n_threshold_points - 1) / n_threshold_points (defaults to 0%, 5%,
    ..., 95%); the 100% case (defer everything) is excluded because
    residual error is undefined for an empty kept set.

    Returned columns: ``score_column``, ``threshold``,
    ``deferral_fraction``, ``residual_error_rate``, ``kept_count``,
    ``deferred_count``, ``total_count``, ``error_rate_overall``,
    ``n_dropped``.

    Raises ``EvaluationError`` on no-positives / no-negatives in
    ``ground_truth[target_column]``, on missing required columns, or
    on empty join between scores and ground_truth.

    Empty ``scores`` returns an empty DataFrame with the correct schema
    (no error).
    """
    if len(scores) == 0:
        return _empty_deferral_curve()

    _validate_ground_truth_classes(ground_truth, target_column)
    joined = _join_scores_truth(scores, ground_truth, target_column)

    rows: list[dict[str, object]] = []
    for col in score_columns:
        df_col, n_dropped = _drop_nan_and_warn(joined, col)
        if len(df_col) == 0:
            continue

        score_vals = df_col[col].to_numpy(dtype=np.float64)
        labels = df_col[target_column].to_numpy().astype(int)
        n_total = len(score_vals)
        error_rate_overall = float(labels.mean())

        fractions = np.linspace(0.0, 1.0, n_threshold_points + 1)[:-1]
        thresholds = np.quantile(score_vals, 1.0 - fractions)

        for frac, thresh in zip(fractions, thresholds, strict=True):
            deferred_mask = score_vals >= thresh
            deferred_count = int(deferred_mask.sum())
            kept_count = n_total - deferred_count
            kept_labels = labels[~deferred_mask]
            residual_error = (
                float(kept_labels.mean())
                if kept_count > 0
                else float("nan")
            )

            rows.append(
                {
                    "score_column": col,
                    "threshold": float(thresh),
                    "deferral_fraction": float(frac),
                    "residual_error_rate": residual_error,
                    "kept_count": kept_count,
                    "deferred_count": deferred_count,
                    "total_count": n_total,
                    "error_rate_overall": error_rate_overall,
                    "n_dropped": n_dropped,
                }
            )

    if not rows:
        return _empty_deferral_curve()
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score_column", "deferral_fraction"]
    ).reset_index(drop=True)
    return df


# ---- deferral_auc ----


def deferral_auc(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    score_columns: Sequence[str] = ("composite",),
    compute_ci: bool = False,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Exact AUC (ROC and PR) per score column, with optional bootstrap
    confidence intervals. Includes sign-aware ROC-AUC reporting.

    Returned columns (without CI): ``score_column``, ``roc_auc``,
    ``roc_auc_sign_aware``, ``roc_auc_direction``, ``pr_auc``,
    ``n_pos``, ``n_neg``, ``n_dropped``.
    With ``compute_ci=True``: adds ``roc_auc_ci_low``,
    ``roc_auc_ci_high``, ``roc_auc_sign_aware_ci_low``,
    ``roc_auc_sign_aware_ci_high``, ``pr_auc_ci_low``,
    ``pr_auc_ci_high`` (95% percentile bootstrap intervals).

    Sign-aware reporting: ``roc_auc_sign_aware`` is
    ``max(roc_auc, 1 - roc_auc)``; ``roc_auc_direction`` is
    ``"greater"`` when ``roc_auc >= 0.5`` else ``"less"``. The
    sign-aware bootstrap CI is computed by reorienting all bootstrap
    AUCs to the point-estimate's direction (i.e., when direction is
    ``"less"``, the bootstrap quantity is ``1 - roc_auc`` per
    resample), then taking the 2.5/97.5 percentiles. The CI does NOT
    clip at 0.5 — resamples that flip direction relative to the
    point estimate appear as sign-aware values < 0.5, accurately
    reflecting sampling uncertainty about both magnitude and
    direction. See module docstring "Sign-aware AUC reporting" for
    the rationale.

    Default ``n_bootstrap=5000`` is for final reporting (~2-5 minutes
    on chest-pain MIMIC scale of ~25k trajectories). For development
    iteration, override ``n_bootstrap=500`` for rough-but-fast values.
    """
    _validate_ground_truth_classes(ground_truth, target_column)
    joined = _join_scores_truth(scores, ground_truth, target_column)
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, object]] = []
    for col in score_columns:
        df_col, n_dropped = _drop_nan_and_warn(joined, col)
        if len(df_col) == 0:
            row: dict[str, object] = {
                "score_column": col,
                "roc_auc": float("nan"),
                "roc_auc_sign_aware": float("nan"),
                "roc_auc_direction": "",
                "pr_auc": float("nan"),
                "n_pos": 0,
                "n_neg": 0,
                "n_dropped": n_dropped,
            }
            if compute_ci:
                row.update(
                    {
                        "roc_auc_ci_low": float("nan"),
                        "roc_auc_ci_high": float("nan"),
                        "roc_auc_sign_aware_ci_low": float("nan"),
                        "roc_auc_sign_aware_ci_high": float("nan"),
                        "pr_auc_ci_low": float("nan"),
                        "pr_auc_ci_high": float("nan"),
                    }
                )
            rows.append(row)
            continue

        scores_arr = df_col[col].to_numpy(dtype=np.float64)
        labels_arr = df_col[target_column].to_numpy().astype(int)
        n_pos = int(labels_arr.sum())
        n_neg = len(labels_arr) - n_pos

        roc_auc = float(roc_auc_score(labels_arr, scores_arr))
        pr_auc = float(average_precision_score(labels_arr, scores_arr))
        sign_aware_auc = max(roc_auc, 1.0 - roc_auc)
        direction = "greater" if roc_auc >= 0.5 else "less"

        row = {
            "score_column": col,
            "roc_auc": roc_auc,
            "roc_auc_sign_aware": sign_aware_auc,
            "roc_auc_direction": direction,
            "pr_auc": pr_auc,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_dropped": n_dropped,
        }

        if compute_ci:
            roc_aucs = np.empty(n_bootstrap)
            pr_aucs = np.empty(n_bootstrap)
            n = len(labels_arr)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                boot_labels = labels_arr[idx]
                boot_scores = scores_arr[idx]
                boot_pos = boot_labels.sum()
                if boot_pos == 0 or boot_pos == n:
                    roc_aucs[i] = np.nan
                    pr_aucs[i] = np.nan
                    continue
                roc_aucs[i] = roc_auc_score(boot_labels, boot_scores)
                pr_aucs[i] = average_precision_score(
                    boot_labels, boot_scores
                )
            # Bootstrap CI on raw AUC.
            row["roc_auc_ci_low"] = float(np.nanpercentile(roc_aucs, 2.5))
            row["roc_auc_ci_high"] = float(np.nanpercentile(roc_aucs, 97.5))
            # Sign-aware CI: reorient all bootstraps to point-estimate
            # direction. CI is NOT clipped at 0.5 — resamples that
            # flipped direction appear as values < 0.5, honestly
            # reflecting direction uncertainty.
            sign_aware_aucs = (
                roc_aucs if direction == "greater" else 1.0 - roc_aucs
            )
            row["roc_auc_sign_aware_ci_low"] = float(
                np.nanpercentile(sign_aware_aucs, 2.5)
            )
            row["roc_auc_sign_aware_ci_high"] = float(
                np.nanpercentile(sign_aware_aucs, 97.5)
            )
            row["pr_auc_ci_low"] = float(np.nanpercentile(pr_aucs, 2.5))
            row["pr_auc_ci_high"] = float(np.nanpercentile(pr_aucs, 97.5))

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("score_column").reset_index(drop=True)
    return df


# ---- calibration_metrics ----


def calibration_metrics(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    score_columns: Sequence[str] = ("composite",),
    n_bins: int = 10,
    binning: Literal["equal_width", "equal_count"] = "equal_count",
) -> pd.DataFrame:
    """ECE, MCE, and Brier score per score column.

    Requires score values in [0, 1] (inclusive, 1e-6 tolerance). The
    composite from ``compute_signatures`` is rank-percentile-normalized
    so it satisfies this; raw components do not. Pass pre-normalized
    columns or omit raw components from ``score_columns``.

    Returned columns: ``score_column``, ``ece``, ``mce``,
    ``brier_score``, ``n_dropped``.

    ``binning`` defaults to ``"equal_count"`` (each bin contains
    ``len(scores) / n_bins`` trajectories). For uniformly-distributed
    rank-percentile scores this is approximately equal to
    ``"equal_width"``, but for non-uniform scores equal_count better
    handles skew.
    """
    _validate_ground_truth_classes(ground_truth, target_column)
    joined = _join_scores_truth(scores, ground_truth, target_column)

    rows: list[dict[str, object]] = []
    for col in score_columns:
        df_col, n_dropped = _drop_nan_and_warn(joined, col)
        if len(df_col) == 0:
            continue

        scores_arr = df_col[col].to_numpy(dtype=np.float64)
        if (scores_arr < -1e-6).any() or (scores_arr > 1.0 + 1e-6).any():
            raise EvaluationError(
                f"score_column={col!r} has values outside [0, 1] "
                f"(min={scores_arr.min():.4f}, max={scores_arr.max():.4f}); "
                f"calibration_metrics requires pre-normalized scores. Use "
                f"the 'composite' column or rank-percentile-normalize raw "
                f"components before passing."
            )
        scores_clipped = np.clip(scores_arr, 0.0, 1.0)
        labels_arr = df_col[target_column].to_numpy().astype(float)

        if binning == "equal_width":
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            bin_edges = np.unique(
                np.quantile(scores_clipped, np.linspace(0.0, 1.0, n_bins + 1))
            )

        ece = 0.0
        mce = 0.0
        n_total = len(scores_clipped)
        for b in range(len(bin_edges) - 1):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == len(bin_edges) - 2:
                mask = (scores_clipped >= lo) & (scores_clipped <= hi)
            else:
                mask = (scores_clipped >= lo) & (scores_clipped < hi)
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            mean_score_b = float(scores_clipped[mask].mean())
            mean_label_b = float(labels_arr[mask].mean())
            diff = abs(mean_score_b - mean_label_b)
            ece += (n_b / n_total) * diff
            mce = max(mce, diff)

        brier = float(brier_score_loss(labels_arr, scores_clipped))

        rows.append(
            {
                "score_column": col,
                "ece": float(ece),
                "mce": float(mce),
                "brier_score": brier,
                "n_dropped": n_dropped,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(
            {
                "score_column": pd.Series(dtype=object),
                "ece": pd.Series(dtype=np.float64),
                "mce": pd.Series(dtype=np.float64),
                "brier_score": pd.Series(dtype=np.float64),
                "n_dropped": pd.Series(dtype=np.int64),
            }
        )
    return df.sort_values("score_column").reset_index(drop=True)


# ---- operating_points ----


def operating_points(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    score_column: str = "composite",
    deferral_fractions: Sequence[float] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Sensitivity / specificity / PPV / NPV at specified deferral
    fractions.

    Treats deferral as the binary classifier: ``deferred = 1`` is the
    "positive" prediction; ``target_column == 1`` is the positive
    ground-truth class (the case that should be deferred).

    Sensitivity = TP / (TP + FN) — among true-positive cases, how many
    were deferred?
    Specificity = TN / (TN + FP) — among true-negative cases, how many
    were correctly kept?
    PPV (precision among deferrals) = TP / (TP + FP) — among deferred
    cases, how many were genuinely positive?
    NPV (precision among kept) = TN / (TN + FN) — among kept cases,
    how many were genuinely negative?

    Returned columns: ``deferral_fraction``, ``threshold``,
    ``sensitivity``, ``specificity``, ``ppv``, ``npv``, ``kept_n``,
    ``deferred_n``, ``n_dropped``.
    """
    _validate_ground_truth_classes(ground_truth, target_column)
    joined = _join_scores_truth(scores, ground_truth, target_column)
    df_col, n_dropped = _drop_nan_and_warn(joined, score_column)

    if len(df_col) == 0:
        return pd.DataFrame(
            {
                "deferral_fraction": pd.Series(dtype=np.float64),
                "threshold": pd.Series(dtype=np.float64),
                "sensitivity": pd.Series(dtype=np.float64),
                "specificity": pd.Series(dtype=np.float64),
                "ppv": pd.Series(dtype=np.float64),
                "npv": pd.Series(dtype=np.float64),
                "kept_n": pd.Series(dtype=np.int64),
                "deferred_n": pd.Series(dtype=np.int64),
                "n_dropped": pd.Series(dtype=np.int64),
            }
        )

    scores_arr = df_col[score_column].to_numpy(dtype=np.float64)
    labels = df_col[target_column].to_numpy().astype(int)

    rows: list[dict[str, object]] = []
    for frac in deferral_fractions:
        if not (0.0 <= frac <= 1.0):
            raise EvaluationError(
                f"deferral_fraction must be in [0, 1], got {frac}"
            )
        threshold = float(np.quantile(scores_arr, 1.0 - frac))
        deferred_mask = scores_arr >= threshold
        kept_mask = ~deferred_mask
        kept_n = int(kept_mask.sum())
        deferred_n = int(deferred_mask.sum())

        n_pos_total = int(labels.sum())
        n_neg_total = len(labels) - n_pos_total
        tp = int((labels[deferred_mask] == 1).sum())
        fn = n_pos_total - tp
        tn = int((labels[kept_mask] == 0).sum())
        fp = n_neg_total - tn

        sensitivity = (
            tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        )
        specificity = (
            tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        )
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

        rows.append(
            {
                "deferral_fraction": float(frac),
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "kept_n": kept_n,
                "deferred_n": deferred_n,
                "n_dropped": n_dropped,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("deferral_fraction").reset_index(drop=True)
    return df


# ---- component_decomposition_table ----


def component_decomposition_table(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    target_column: str,
    compute_ci: bool = True,
    n_bootstrap: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """AUC (ROC + PR) plus 95% CIs for each signature component and
    the composite. Convenience wrapper around ``deferral_auc`` that
    fixes the four standard column names.

    Defaults ``compute_ci=True`` because the use case is paper-table
    output: AUCs without confidence intervals are misleading. Override
    to ``False`` for fast iteration.
    """
    return deferral_auc(
        scores,
        ground_truth,
        target_column,
        score_columns=(
            "entropy_plateau",
            "voi_flatness",
            "distance_from_trajectory",
            "composite",
        ),
        compute_ci=compute_ci,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
