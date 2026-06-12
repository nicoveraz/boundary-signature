"""Threshold calibration utilities for deployed deferral signals.

Mechanical operations for downstream deployment: take raw signature
scores (``mean_entropy``, composite, etc.) plus a threshold, return a
binary deferral indicator. Find the threshold that produces a specified
false-positive rate or sensitivity from a labelled validation set.
Tabulate the ROC threshold curve.

This module provides *mathematical primitives* — the framework is
agnostic to which signature operationalisation gets deployed and at
what threshold. Deployment-side decisions (what's the right
sensitivity/specificity tradeoff for clinical chest-pain triage; what
threshold reflects acceptable false-positive rate at a specific
clinical site) are not made here. Per the project's
*compartmentalization* principle: bsig provides operations; the
clinical product layer composes with them.

Distinct from ``bsig.core.evaluation.calibration_metrics``, which
measures whether a probability is well-calibrated against outcomes
(Brier / ECE / MCE). This module finds *operating thresholds* on a
signal regardless of whether the signal is itself a calibrated
probability.

Score sign convention is caller-specified via the ``direction``
parameter: ``"greater"`` means high scores defer (typical for entropy,
composite, distance — high uncertainty signals boundary); ``"less"``
means low scores defer (typical for confidence — low confidence signals
boundary). The ``apply_threshold`` and ``threshold_at_*`` functions both
respect this direction; the returned indicators / thresholds are valid
when applied in the same direction.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

Direction = Literal["greater", "less"]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Output of ``threshold_at_*`` calibration helpers.

    Carries the chosen threshold and the operating-point metrics it
    achieves on the calibration set:

    - ``threshold``: the threshold value to use with
      :func:`apply_threshold` (in the same ``direction``).
    - ``fpr``: false-positive rate at this threshold (cases incorrectly
      flagged for deferral / total negatives).
    - ``tpr``: true-positive rate (sensitivity); recall on the
      defer-positive class.
    - ``fnr``: false-negative rate (1 − tpr); cases that should have
      been deferred but weren't.
    - ``tnr``: true-negative rate (specificity); 1 − fpr.
    - ``n_pos``, ``n_neg``: positive/negative counts in the calibration
      set the threshold was computed against.
    """

    threshold: float
    fpr: float
    tpr: float
    fnr: float
    tnr: float
    n_pos: int
    n_neg: int


def apply_threshold(
    scores: Sequence[float],
    threshold: float,
    direction: Direction = "greater",
) -> np.ndarray:
    """Apply a deferral threshold to raw signature scores.

    Returns a boolean numpy array of shape ``(len(scores),)``: ``True``
    where the score crosses the threshold in the specified direction.

    Parameters
    ----------
    scores : sequence of float
        Per-trajectory raw signature scores.
    threshold : float
        The threshold value.
    direction : "greater" | "less"
        ``"greater"`` (default): score >= threshold → defer (True).
        Typical for entropy, composite, distance — high uncertainty
        signals boundary.
        ``"less"``: score <= threshold → defer (True). Typical for
        confidence — low confidence signals boundary.

    Returns
    -------
    np.ndarray of bool
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    if direction == "greater":
        return arr >= threshold
    elif direction == "less":
        return arr <= threshold
    else:
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction!r}"
        )


def threshold_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    target_fpr: float,
    direction: Direction = "greater",
) -> CalibrationResult:
    """Find the threshold producing FPR ≤ ``target_fpr`` at the highest
    achievable TPR.

    Useful for "what threshold defers at most X % of cases that don't
    actually need consultation, while catching as many true-defer cases
    as possible?"

    Parameters
    ----------
    scores : sequence of float
        Per-trajectory raw signature scores.
    labels : sequence of {0, 1}
        Per-trajectory binary labels (1 = positive class = should-be-
        deferred; 0 = negative class).
    target_fpr : float in [0, 1]
        Desired upper bound on false-positive rate.
    direction : "greater" | "less"
        Sign convention for ``scores`` per :func:`apply_threshold`.

    Returns
    -------
    :class:`CalibrationResult` with the chosen threshold and operating
    point.

    Raises
    ------
    ValueError
        If labels are degenerate (all 0 or all 1), or ``target_fpr``
        is outside ``[0, 1]``.
    """
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(
            f"target_fpr must be in [0, 1], got {target_fpr}"
        )
    fprs, tprs, thresholds, n_pos, n_neg = _roc_arrays(scores, labels, direction)
    # Pick the largest threshold-index where fpr <= target_fpr.
    # roc_curve returns thresholds in DECREASING order (when direction is
    # "greater"); fprs and tprs are non-decreasing along that order. We
    # want the highest TPR achievable subject to FPR ≤ target_fpr.
    ok = fprs <= target_fpr
    if not ok.any():
        # Even the strictest threshold has FPR > target_fpr — return that
        # threshold (the closest achievable).
        idx = int(np.argmin(fprs))
    else:
        # Among indices where fpr <= target, pick the one with highest tpr.
        idx = int(np.argmax(np.where(ok, tprs, -1.0)))
    return _result_at_index(fprs, tprs, thresholds, idx, n_pos, n_neg)


def threshold_at_sensitivity(
    scores: Sequence[float],
    labels: Sequence[int],
    target_sensitivity: float,
    direction: Direction = "greater",
) -> CalibrationResult:
    """Find the threshold producing sensitivity (TPR) ≥
    ``target_sensitivity`` at the lowest achievable FPR.

    Useful for "what threshold catches at least X % of true-defer cases,
    while flagging as few non-defer cases as possible?"

    Parameters
    ----------
    scores : sequence of float
    labels : sequence of {0, 1}
    target_sensitivity : float in [0, 1]
        Desired lower bound on TPR / sensitivity / recall on positives.
    direction : "greater" | "less"

    Returns
    -------
    :class:`CalibrationResult`

    Raises
    ------
    ValueError
        If labels are degenerate or ``target_sensitivity`` is outside
        ``[0, 1]``.
    """
    if not (0.0 <= target_sensitivity <= 1.0):
        raise ValueError(
            f"target_sensitivity must be in [0, 1], got {target_sensitivity}"
        )
    fprs, tprs, thresholds, n_pos, n_neg = _roc_arrays(scores, labels, direction)
    ok = tprs >= target_sensitivity
    if not ok.any():
        # No threshold achieves the target sensitivity — return the most
        # permissive threshold (highest TPR available).
        idx = int(np.argmax(tprs))
    else:
        # Among indices where tpr >= target, pick the one with lowest fpr.
        idx = int(np.argmin(np.where(ok, fprs, np.inf)))
    return _result_at_index(fprs, tprs, thresholds, idx, n_pos, n_neg)


def roc_threshold_table(
    scores: Sequence[float],
    labels: Sequence[int],
    direction: Direction = "greater",
) -> pd.DataFrame:
    """Tabulate the ROC threshold curve as a DataFrame.

    For each candidate threshold (drawn from the unique score values
    plus boundary thresholds), report ``threshold``, ``fpr``, ``tpr``,
    ``fnr``, ``tnr``. Useful for plotting deferral-curve tradeoffs and
    for clinical-team review of where to set the operating point.

    Parameters
    ----------
    scores : sequence of float
    labels : sequence of {0, 1}
    direction : "greater" | "less"

    Returns
    -------
    pd.DataFrame with columns ``threshold, fpr, tpr, fnr, tnr``.
    Sorted by threshold in the order ``roc_curve`` produces (decreasing
    for ``direction="greater"``; increasing for ``direction="less"``).

    Raises
    ------
    ValueError if labels are degenerate.
    """
    fprs, tprs, thresholds, _n_pos, _n_neg = _roc_arrays(
        scores, labels, direction
    )
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "fpr": fprs,
            "tpr": tprs,
            "fnr": 1.0 - tprs,
            "tnr": 1.0 - fprs,
        }
    )


# ============================================================
# Internal helpers
# ============================================================


def _roc_arrays(
    scores: Sequence[float],
    labels: Sequence[int],
    direction: Direction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Compute (fpr, tpr, thresholds) arrays from sklearn.roc_curve.

    Handles the ``direction`` parameter by negating scores when
    ``direction="less"``. Validates labels are non-degenerate.
    Returns thresholds in their original sign (the user can use them
    directly with :func:`apply_threshold` under the same direction).
    """
    score_arr = np.asarray(list(scores), dtype=np.float64)
    label_arr = np.asarray(list(labels), dtype=np.int64)
    if score_arr.shape != label_arr.shape:
        raise ValueError(
            f"scores and labels must have the same length; "
            f"got {score_arr.shape} vs {label_arr.shape}"
        )
    if label_arr.size == 0:
        raise ValueError("scores/labels must be non-empty")
    n_pos = int((label_arr == 1).sum())
    n_neg = int((label_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"labels must contain both classes for ROC computation; "
            f"got n_pos={n_pos}, n_neg={n_neg}"
        )

    if direction == "greater":
        sklearn_scores = score_arr
    elif direction == "less":
        # roc_curve assumes high score = positive; for direction="less"
        # we negate scores so the same assumption holds.
        sklearn_scores = -score_arr
    else:
        raise ValueError(
            f"direction must be 'greater' or 'less', got {direction!r}"
        )

    fprs, tprs, thresholds_internal = roc_curve(label_arr, sklearn_scores)
    if direction == "less":
        # Restore original sign: thresholds were computed against -scores;
        # flip back so the user can use them directly with apply_threshold.
        thresholds_external = -thresholds_internal
    else:
        thresholds_external = thresholds_internal
    return fprs, tprs, thresholds_external, n_pos, n_neg


def _result_at_index(
    fprs: np.ndarray,
    tprs: np.ndarray,
    thresholds: np.ndarray,
    idx: int,
    n_pos: int,
    n_neg: int,
) -> CalibrationResult:
    return CalibrationResult(
        threshold=float(thresholds[idx]),
        fpr=float(fprs[idx]),
        tpr=float(tprs[idx]),
        fnr=float(1.0 - tprs[idx]),
        tnr=float(1.0 - fprs[idx]),
        n_pos=n_pos,
        n_neg=n_neg,
    )
