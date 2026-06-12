"""Tests for bsig.core.calibration — threshold-calibration utilities for
deployed deferral signals.

These are mathematical primitives consumed by clinical-product
deployment (Eunosia Phase 1). The framework exposes the operations;
the clinical team picks thresholds via the project's
*compartmentalization* principle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bsig.core.calibration import (
    CalibrationResult,
    apply_threshold,
    roc_threshold_table,
    threshold_at_fpr,
    threshold_at_sensitivity,
)


# ---- apply_threshold ----


def test_apply_threshold_greater() -> None:
    """direction='greater': score >= threshold → True (defer)."""
    scores = [0.1, 0.5, 0.7, 0.9]
    out = apply_threshold(scores, threshold=0.6, direction="greater")
    np.testing.assert_array_equal(out, [False, False, True, True])


def test_apply_threshold_less() -> None:
    """direction='less': score <= threshold → True (defer).
    Useful for confidence (low confidence = boundary case)."""
    scores = [0.1, 0.5, 0.7, 0.9]
    out = apply_threshold(scores, threshold=0.6, direction="less")
    np.testing.assert_array_equal(out, [True, True, False, False])


def test_apply_threshold_boundary_inclusive() -> None:
    """Score == threshold counts as crossing in both directions."""
    scores = [0.5]
    assert apply_threshold(scores, 0.5, "greater")[0]
    assert apply_threshold(scores, 0.5, "less")[0]


def test_apply_threshold_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="must be 'greater' or 'less'"):
        apply_threshold([0.1], 0.5, direction="up")  # type: ignore[arg-type]


def test_apply_threshold_returns_numpy_bool_array() -> None:
    out = apply_threshold([0.1, 0.9], 0.5)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool


# ---- threshold_at_fpr ----


def _separable_scores_labels() -> tuple[list[float], list[int]]:
    """A perfectly-separable score/label set: positives have higher
    scores than negatives. Useful for testing the happy path."""
    # 10 negatives at scores 0.1-0.4, 10 positives at scores 0.6-0.9.
    neg_scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.37, 0.39, 0.4]
    pos_scores = [0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.89, 0.9]
    scores = neg_scores + pos_scores
    labels = [0] * 10 + [1] * 10
    return scores, labels


def test_threshold_at_fpr_finds_threshold_separable_data() -> None:
    """On perfectly-separable data, FPR=0 is achievable with TPR=1."""
    scores, labels = _separable_scores_labels()
    result = threshold_at_fpr(scores, labels, target_fpr=0.0)
    assert isinstance(result, CalibrationResult)
    assert result.fpr == pytest.approx(0.0, abs=1e-9)
    assert result.tpr == pytest.approx(1.0, abs=1e-9)
    assert result.n_pos == 10
    assert result.n_neg == 10
    # The threshold should sit between the highest negative (0.4) and
    # lowest positive (0.6).
    assert 0.4 < result.threshold <= 0.6


def test_threshold_at_fpr_with_overlap() -> None:
    """With overlapping classes, target_fpr=0.1 produces a threshold
    that flags at most 10% of negatives while catching most positives."""
    rng = np.random.default_rng(42)
    n_neg, n_pos = 100, 100
    neg_scores = rng.normal(loc=0.3, scale=0.1, size=n_neg)
    pos_scores = rng.normal(loc=0.6, scale=0.1, size=n_pos)
    scores = list(neg_scores) + list(pos_scores)
    labels = [0] * n_neg + [1] * n_pos
    result = threshold_at_fpr(scores, labels, target_fpr=0.1)
    assert result.fpr <= 0.1 + 1e-9
    # On normally-distributed data with this separation, TPR should be
    # well above 0.5 even at FPR=0.1.
    assert result.tpr > 0.5


def test_threshold_at_fpr_target_zero_picks_strictest() -> None:
    """target_fpr=0 with overlapping classes still returns the
    strictest threshold achieving FPR=0 (or closest thereto)."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    result = threshold_at_fpr(scores, labels, target_fpr=0.0)
    assert result.fpr == pytest.approx(0.0)
    assert result.tpr == pytest.approx(1.0)


def test_threshold_at_fpr_rejects_invalid_target() -> None:
    scores, labels = _separable_scores_labels()
    with pytest.raises(ValueError, match="target_fpr must be in"):
        threshold_at_fpr(scores, labels, target_fpr=1.5)
    with pytest.raises(ValueError, match="target_fpr must be in"):
        threshold_at_fpr(scores, labels, target_fpr=-0.1)


def test_threshold_at_fpr_rejects_degenerate_labels() -> None:
    """All-positive or all-negative labels can't produce an ROC curve."""
    with pytest.raises(ValueError, match="both classes"):
        threshold_at_fpr([0.1, 0.5, 0.9], [1, 1, 1], target_fpr=0.5)
    with pytest.raises(ValueError, match="both classes"):
        threshold_at_fpr([0.1, 0.5, 0.9], [0, 0, 0], target_fpr=0.5)


def test_threshold_at_fpr_direction_less() -> None:
    """direction='less' inverts the score sign convention. Confidence
    scores (low = boundary) work with direction='less'."""
    # B confidence: positives (deferred-needing) have LOWER confidence
    pos_conf = [0.1, 0.2, 0.3, 0.4]
    neg_conf = [0.6, 0.7, 0.8, 0.9]
    scores = pos_conf + neg_conf
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    result = threshold_at_fpr(
        scores, labels, target_fpr=0.0, direction="less"
    )
    # All positives correctly flagged; no negatives flagged
    assert result.fpr == pytest.approx(0.0)
    assert result.tpr == pytest.approx(1.0)
    # The threshold is in the gap between max positive conf (0.4) and
    # min negative conf (0.6); applied with direction='less', so
    # score ≤ threshold = defer.
    assert 0.4 <= result.threshold < 0.6


# ---- threshold_at_sensitivity ----


def test_threshold_at_sensitivity_separable() -> None:
    scores, labels = _separable_scores_labels()
    result = threshold_at_sensitivity(scores, labels, target_sensitivity=1.0)
    assert result.tpr == pytest.approx(1.0)
    assert result.fpr == pytest.approx(0.0)


def test_threshold_at_sensitivity_partial_target() -> None:
    """target_sensitivity=0.5 picks the lowest-FPR threshold catching
    at least half the positives."""
    rng = np.random.default_rng(42)
    neg_scores = rng.normal(loc=0.3, scale=0.1, size=100)
    pos_scores = rng.normal(loc=0.6, scale=0.1, size=100)
    scores = list(neg_scores) + list(pos_scores)
    labels = [0] * 100 + [1] * 100
    result = threshold_at_sensitivity(scores, labels, target_sensitivity=0.5)
    assert result.tpr >= 0.5 - 1e-9
    # On normally-distributed data, achieving 50% sensitivity should have
    # very low FPR.
    assert result.fpr < 0.1


def test_threshold_at_sensitivity_rejects_invalid_target() -> None:
    scores, labels = _separable_scores_labels()
    with pytest.raises(ValueError, match="target_sensitivity must be in"):
        threshold_at_sensitivity(scores, labels, target_sensitivity=1.5)
    with pytest.raises(ValueError, match="target_sensitivity must be in"):
        threshold_at_sensitivity(scores, labels, target_sensitivity=-0.1)


# ---- roc_threshold_table ----


def test_roc_threshold_table_columns() -> None:
    scores, labels = _separable_scores_labels()
    df = roc_threshold_table(scores, labels)
    assert list(df.columns) == ["threshold", "fpr", "tpr", "fnr", "tnr"]


def test_roc_threshold_table_fnr_plus_tpr_equals_one() -> None:
    scores, labels = _separable_scores_labels()
    df = roc_threshold_table(scores, labels)
    np.testing.assert_allclose(df["tpr"] + df["fnr"], 1.0, atol=1e-9)
    np.testing.assert_allclose(df["fpr"] + df["tnr"], 1.0, atol=1e-9)


def test_roc_threshold_table_with_separable_data_includes_perfect_point() -> None:
    """Separable data: there's a row with fpr=0 and tpr=1."""
    scores, labels = _separable_scores_labels()
    df = roc_threshold_table(scores, labels)
    perfect_rows = df[(df["fpr"] == 0.0) & (df["tpr"] == 1.0)]
    assert len(perfect_rows) >= 1


def test_roc_threshold_table_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_threshold_table([0.1, 0.5, 0.9], [1, 1, 1])


# ---- end-to-end deployment workflow ----


def test_deployment_workflow_calibrate_then_apply() -> None:
    """The full clinical-deployment workflow:
    1. Score a calibration set with ground truth.
    2. Find the threshold producing target FPR (clinical tolerance).
    3. Apply the threshold to NEW unlabelled scores.
    """
    # Step 1: calibration set
    rng = np.random.default_rng(42)
    cal_neg = rng.normal(loc=0.3, scale=0.1, size=100)
    cal_pos = rng.normal(loc=0.6, scale=0.1, size=50)
    cal_scores = list(cal_neg) + list(cal_pos)
    cal_labels = [0] * 100 + [1] * 50

    # Step 2: pick threshold at 10% FPR (clinical tolerance)
    cal = threshold_at_fpr(cal_scores, cal_labels, target_fpr=0.1)
    assert cal.fpr <= 0.1 + 1e-9
    threshold = cal.threshold

    # Step 3: deploy — new scores arrive without labels, apply the threshold
    new_scores = [0.2, 0.4, 0.5, 0.7, 0.8]
    deferral_indicator = apply_threshold(new_scores, threshold)
    # Should not raise; should be bool array of correct length
    assert len(deferral_indicator) == 5
    assert deferral_indicator.dtype == bool


def test_deployment_workflow_rate_consistency() -> None:
    """Applying the chosen threshold to the calibration set itself
    should reproduce the reported FPR/TPR (within sampling-noise; here
    on the same data, exactly)."""
    scores, labels = _separable_scores_labels()
    cal = threshold_at_fpr(scores, labels, target_fpr=0.0)
    indicator = apply_threshold(scores, cal.threshold)
    label_arr = np.asarray(labels)
    flagged_neg = int(((indicator) & (label_arr == 0)).sum())
    flagged_pos = int(((indicator) & (label_arr == 1)).sum())
    measured_fpr = flagged_neg / cal.n_neg
    measured_tpr = flagged_pos / cal.n_pos
    assert measured_fpr == pytest.approx(cal.fpr, abs=1e-9)
    assert measured_tpr == pytest.approx(cal.tpr, abs=1e-9)
