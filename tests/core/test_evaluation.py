"""Tests for evaluation: deferral curves, AUC, calibration, operating points."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bsig.core.evaluation import (
    EvaluationError,
    EvaluationWarning,
    calibration_metrics,
    component_decomposition_table,
    deferral_auc,
    deferral_curve,
    operating_points,
)


# ---- Fixtures ----


def _scores(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "entropy_plateau": rng.normal(0, 1, n).astype(np.float32),
            "voi_flatness": rng.uniform(0, 2, n).astype(np.float32),
            "distance_from_trajectory": rng.uniform(0, 1, n).astype(np.float32),
            "composite": rng.uniform(0, 1, n).astype(np.float32),
        }
    )


def _truth(n: int = 100, p_pos: float = 0.3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, p_pos, n).astype(int),
        }
    )


def _correlated(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High composite -> high probability of label=1. Tests that AUC > 0.5."""
    rng = np.random.default_rng(seed)
    composite = rng.uniform(0, 1, n).astype(np.float32)
    labels = (composite > 0.5).astype(int)
    flip = rng.binomial(1, 0.1, n).astype(bool)
    labels = np.where(flip, 1 - labels, labels)
    scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "entropy_plateau": rng.normal(0, 1, n).astype(np.float32),
            "voi_flatness": rng.uniform(0, 2, n).astype(np.float32),
            "distance_from_trajectory": rng.uniform(0, 1, n).astype(np.float32),
            "composite": composite,
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": labels,
        }
    )
    return scores, truth


# ---- Validation: shared helpers via public API ----


def test_raises_on_no_positives() -> None:
    scores = _scores(20)
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(20)],
            "needs_consultation": np.zeros(20, dtype=int),
        }
    )
    with pytest.raises(EvaluationError, match="no positive"):
        deferral_auc(scores, truth, "needs_consultation")


def test_raises_on_no_negatives() -> None:
    scores = _scores(20)
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(20)],
            "needs_consultation": np.ones(20, dtype=int),
        }
    )
    with pytest.raises(EvaluationError, match="no negative"):
        calibration_metrics(scores, truth, "needs_consultation")


def test_raises_on_missing_target_column() -> None:
    scores = _scores(20)
    truth = _truth(20)
    with pytest.raises(EvaluationError, match="not in ground_truth columns"):
        deferral_auc(scores, truth, "nonexistent_column")


def test_raises_on_empty_join() -> None:
    scores = _scores(10)
    truth = _truth(10).assign(trajectory_id=lambda d: d["trajectory_id"] + "_xyz")
    with pytest.raises(EvaluationError, match="No trajectories matched"):
        deferral_auc(scores, truth, "needs_consultation")


def test_inner_join_handles_partial_overlap() -> None:
    """Scores has more trajectories than truth; inner join keeps only
    overlapping rows."""
    scores = _scores(20)
    truth = _truth(10)
    df = deferral_auc(scores, truth, "needs_consultation")
    assert len(df) == 1
    assert df.iloc[0]["n_pos"] + df.iloc[0]["n_neg"] == 10


# ---- NaN handling ----


def test_nan_scores_dropped_with_warning() -> None:
    scores = _scores(20)
    scores.loc[0:4, "composite"] = np.nan
    truth = _truth(20, p_pos=0.5, seed=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EvaluationWarning)
        df = deferral_auc(scores, truth, "needs_consultation")
    assert any(
        isinstance(w.message, EvaluationWarning) and "5" in str(w.message)
        for w in caught
    )
    assert df.iloc[0]["n_dropped"] == 5


def test_n_dropped_reported_per_column() -> None:
    """Different score columns may have different NaN counts."""
    scores = _scores(20)
    scores.loc[0:2, "composite"] = np.nan
    scores.loc[0:5, "voi_flatness"] = np.nan
    truth = _truth(20, p_pos=0.5, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EvaluationWarning)
        df = deferral_auc(
            scores, truth, "needs_consultation",
            score_columns=("composite", "voi_flatness"),
        )
    df = df.set_index("score_column")
    assert df.loc["composite", "n_dropped"] == 3
    assert df.loc["voi_flatness", "n_dropped"] == 6


# ---- deferral_curve ----


def test_deferral_curve_schema() -> None:
    scores, truth = _correlated(50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = deferral_curve(scores, truth, "needs_consultation")
    expected_cols = [
        "score_column", "threshold", "deferral_fraction",
        "residual_error_rate", "kept_count", "deferred_count",
        "total_count", "error_rate_overall", "n_dropped",
    ]
    assert list(df.columns) == expected_cols


def test_deferral_curve_n_threshold_points() -> None:
    scores, truth = _correlated(50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = deferral_curve(
            scores, truth, "needs_consultation", n_threshold_points=10
        )
    assert len(df) == 10


def test_deferral_curve_residual_decreases_with_correlation() -> None:
    """When composite correlates with label, deferring high scores should
    drop the residual error rate."""
    scores, truth = _correlated(500)
    df = deferral_curve(
        scores, truth, "needs_consultation", n_threshold_points=10
    )
    overall = df.iloc[0]["error_rate_overall"]
    high_deferral = df[df["deferral_fraction"] >= 0.4]
    assert (high_deferral["residual_error_rate"] < overall).all()


def test_deferral_curve_total_count_consistency() -> None:
    """total_count == kept_count + deferred_count."""
    scores, truth = _correlated(50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = deferral_curve(scores, truth, "needs_consultation")
    assert (df["total_count"] == df["kept_count"] + df["deferred_count"]).all()


def test_deferral_curve_empty_scores_returns_empty_df() -> None:
    """Empty input is a valid degenerate case (no error)."""
    empty = pd.DataFrame(
        {
            "trajectory_id": pd.Series(dtype=object),
            "composite": pd.Series(dtype=np.float32),
        }
    )
    truth = _truth(10)
    df = deferral_curve(empty, truth, "needs_consultation")
    assert len(df) == 0


def test_deferral_curve_multiple_score_columns() -> None:
    scores, truth = _correlated(100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = deferral_curve(
            scores, truth, "needs_consultation",
            score_columns=("composite", "distance_from_trajectory"),
            n_threshold_points=10,
        )
    assert set(df["score_column"].unique()) == {
        "composite", "distance_from_trajectory"
    }


# ---- deferral_auc ----


def test_deferral_auc_schema_without_ci() -> None:
    scores, truth = _correlated(50)
    df = deferral_auc(scores, truth, "needs_consultation")
    assert list(df.columns) == [
        "score_column", "roc_auc", "roc_auc_sign_aware",
        "roc_auc_direction", "pr_auc", "n_pos", "n_neg", "n_dropped",
    ]


def test_deferral_auc_correlated_scores_yield_high_auc() -> None:
    scores, truth = _correlated(500)
    df = deferral_auc(scores, truth, "needs_consultation")
    composite_auc = df[df["score_column"] == "composite"].iloc[0]["roc_auc"]
    assert composite_auc > 0.85


def test_deferral_auc_random_scores_near_half() -> None:
    """Uncorrelated scores should give AUC near 0.5."""
    scores = _scores(500, seed=10)
    truth = _truth(500, p_pos=0.5, seed=11)
    df = deferral_auc(scores, truth, "needs_consultation")
    composite_auc = df[df["score_column"] == "composite"].iloc[0]["roc_auc"]
    assert 0.4 < composite_auc < 0.6


def test_deferral_auc_with_ci_adds_columns() -> None:
    scores, truth = _correlated(100)
    df = deferral_auc(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200, random_seed=42,
    )
    for col in (
        "roc_auc_ci_low", "roc_auc_ci_high",
        "roc_auc_sign_aware_ci_low", "roc_auc_sign_aware_ci_high",
        "pr_auc_ci_low", "pr_auc_ci_high",
    ):
        assert col in df.columns


# ---- sign-aware AUC reporting ----


def test_sign_aware_auc_greater_direction_when_auc_above_half() -> None:
    """When raw AUC > 0.5, sign-aware == raw AUC and direction is 'greater'."""
    scores, truth = _correlated(500)
    df = deferral_auc(scores, truth, "needs_consultation")
    row = df[df["score_column"] == "composite"].iloc[0]
    assert row["roc_auc"] > 0.5
    assert row["roc_auc_sign_aware"] == pytest.approx(row["roc_auc"], abs=1e-12)
    assert row["roc_auc_direction"] == "greater"


def test_sign_aware_auc_less_direction_when_score_inverted() -> None:
    """Inverting the score sign produces raw AUC < 0.5; sign-aware
    reflects 1 - raw_auc and reports direction='less'. The sign-aware
    magnitude equals the magnitude of the un-inverted score."""
    scores, truth = _correlated(500)
    raw_df = deferral_auc(scores, truth, "needs_consultation")
    raw_row = raw_df[raw_df["score_column"] == "composite"].iloc[0]

    inverted = scores.copy()
    inverted["composite"] = -inverted["composite"]
    inv_df = deferral_auc(
        inverted, truth, "needs_consultation"
    )
    inv_row = inv_df[inv_df["score_column"] == "composite"].iloc[0]

    assert inv_row["roc_auc"] < 0.5
    assert inv_row["roc_auc_direction"] == "less"
    assert inv_row["roc_auc_sign_aware"] == pytest.approx(
        1.0 - inv_row["roc_auc"], abs=1e-12
    )
    # Sign-aware magnitude is invariant to score-sign flipping.
    assert inv_row["roc_auc_sign_aware"] == pytest.approx(
        raw_row["roc_auc_sign_aware"], abs=1e-12
    )


def test_sign_aware_auc_at_half_picks_greater() -> None:
    """Exact AUC=0.5 ties are reported as direction='greater'."""
    # Construct labels and scores that yield AUC exactly 0.5: equal
    # number of pos/neg with the same score for all.
    scores = pd.DataFrame({
        "trajectory_id": [f"t{i}" for i in range(10)],
        "composite": [0.5] * 10,
    })
    truth = pd.DataFrame({
        "trajectory_id": [f"t{i}" for i in range(10)],
        "needs_consultation": [0, 1] * 5,
    })
    df = deferral_auc(scores, truth, "needs_consultation")
    row = df.iloc[0]
    assert row["roc_auc"] == pytest.approx(0.5, abs=1e-9)
    assert row["roc_auc_sign_aware"] == pytest.approx(0.5, abs=1e-9)
    assert row["roc_auc_direction"] == "greater"


def test_sign_aware_ci_reorients_to_point_direction() -> None:
    """When point AUC < 0.5 (direction='less'), the sign-aware CI is
    the bootstrap percentiles of (1 - raw_auc) per resample. Verify by
    constructing an inverted-score case and checking that sign-aware CI
    matches the un-inverted score's raw CI."""
    scores, truth = _correlated(300)
    raw_df = deferral_auc(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200, random_seed=42,
    )
    raw_row = raw_df[raw_df["score_column"] == "composite"].iloc[0]

    inverted = scores.copy()
    inverted["composite"] = -inverted["composite"]
    inv_df = deferral_auc(
        inverted, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200, random_seed=42,
    )
    inv_row = inv_df[inv_df["score_column"] == "composite"].iloc[0]

    # Bootstrapping uses the same seed; the per-resample resamplings
    # are identical. Inverted CI on raw is 1 - reversed bounds.
    assert inv_row["roc_auc_ci_low"] == pytest.approx(
        1.0 - raw_row["roc_auc_ci_high"], abs=1e-9
    )
    assert inv_row["roc_auc_ci_high"] == pytest.approx(
        1.0 - raw_row["roc_auc_ci_low"], abs=1e-9
    )
    # Sign-aware CI on the inverted scores reorients back to the
    # original direction, so it matches the raw CI of the un-inverted
    # score.
    assert inv_row["roc_auc_sign_aware_ci_low"] == pytest.approx(
        raw_row["roc_auc_ci_low"], abs=1e-9
    )
    assert inv_row["roc_auc_sign_aware_ci_high"] == pytest.approx(
        raw_row["roc_auc_ci_high"], abs=1e-9
    )


def test_deferral_auc_ci_brackets_point_estimate() -> None:
    scores, truth = _correlated(200)
    df = deferral_auc(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=500, random_seed=42,
    )
    row = df[df["score_column"] == "composite"].iloc[0]
    assert row["roc_auc_ci_low"] <= row["roc_auc"] <= row["roc_auc_ci_high"]
    assert row["pr_auc_ci_low"] <= row["pr_auc"] <= row["pr_auc_ci_high"]


def test_deferral_auc_seed_reproduces_ci() -> None:
    scores, truth = _correlated(100)
    df1 = deferral_auc(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200, random_seed=42,
    )
    df2 = deferral_auc(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200, random_seed=42,
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_deferral_auc_n_pos_n_neg_correct() -> None:
    scores, truth = _correlated(200)
    expected_pos = int(truth["needs_consultation"].sum())
    expected_neg = len(truth) - expected_pos
    df = deferral_auc(scores, truth, "needs_consultation")
    row = df.iloc[0]
    assert row["n_pos"] == expected_pos
    assert row["n_neg"] == expected_neg


# ---- calibration_metrics ----


def test_calibration_requires_scores_in_unit_interval() -> None:
    """Raw entropy_plateau (signed slope) is not in [0, 1] -> raise."""
    scores = _scores(50)
    truth = _truth(50, p_pos=0.5, seed=4)
    with pytest.raises(EvaluationError, match="outside \\[0, 1\\]"):
        calibration_metrics(
            scores, truth, "needs_consultation",
            score_columns=("entropy_plateau",),
        )


def test_calibration_composite_works() -> None:
    """Composite is rank-percentile-normalized to [0, 1] -> works."""
    scores, truth = _correlated(100)
    df = calibration_metrics(
        scores, truth, "needs_consultation",
        score_columns=("composite",),
    )
    assert list(df.columns) == [
        "score_column", "ece", "mce", "brier_score", "n_dropped"
    ]
    assert 0.0 <= df.iloc[0]["ece"] <= 1.0
    assert 0.0 <= df.iloc[0]["mce"] <= 1.0
    assert 0.0 <= df.iloc[0]["brier_score"] <= 1.0


def test_calibration_perfect_scores_have_low_ece() -> None:
    """Perfectly calibrated scores: composite = label probability exactly."""
    n = 500
    rng = np.random.default_rng(42)
    composite = rng.uniform(0, 1, n).astype(np.float32)
    labels = (rng.uniform(0, 1, n) < composite).astype(int)
    scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "composite": composite,
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": labels,
        }
    )
    df = calibration_metrics(scores, truth, "needs_consultation")
    assert df.iloc[0]["ece"] < 0.1


def test_calibration_equal_count_vs_equal_width() -> None:
    """Both bin types produce valid metrics on uniform composite."""
    scores, truth = _correlated(200)
    df_ec = calibration_metrics(
        scores, truth, "needs_consultation", binning="equal_count"
    )
    df_ew = calibration_metrics(
        scores, truth, "needs_consultation", binning="equal_width"
    )
    assert len(df_ec) == 1
    assert len(df_ew) == 1


# ---- operating_points ----


def test_operating_points_schema() -> None:
    scores, truth = _correlated(100)
    df = operating_points(scores, truth, "needs_consultation")
    assert list(df.columns) == [
        "deferral_fraction", "threshold", "sensitivity", "specificity",
        "ppv", "npv", "kept_n", "deferred_n", "n_dropped",
    ]


def test_operating_points_default_three_fractions() -> None:
    scores, truth = _correlated(100)
    df = operating_points(scores, truth, "needs_consultation")
    assert len(df) == 3
    assert list(df["deferral_fraction"]) == [0.10, 0.20, 0.30]


def test_operating_points_custom_fractions() -> None:
    scores, truth = _correlated(100)
    df = operating_points(
        scores, truth, "needs_consultation",
        deferral_fractions=(0.05, 0.5, 0.95),
    )
    assert list(df["deferral_fraction"]) == [0.05, 0.5, 0.95]


def test_operating_points_kept_plus_deferred_equals_total() -> None:
    scores, truth = _correlated(100)
    df = operating_points(scores, truth, "needs_consultation")
    assert (df["kept_n"] + df["deferred_n"] == 100).all()


def test_operating_points_rejects_invalid_fraction() -> None:
    scores, truth = _correlated(50)
    with pytest.raises(EvaluationError, match="must be in \\[0, 1\\]"):
        operating_points(
            scores, truth, "needs_consultation",
            deferral_fractions=(0.5, 1.5),
        )


def test_operating_points_correlated_scores_better_than_random() -> None:
    """Sensitivity at deferral fraction f should exceed f when scores
    correlate with labels (random scoring would yield sensitivity = f)."""
    scores, truth = _correlated(500)
    df = operating_points(
        scores, truth, "needs_consultation",
        deferral_fractions=(0.3,),
    )
    sensitivity_at_30 = df.iloc[0]["sensitivity"]
    assert sensitivity_at_30 > 0.3


# ---- component_decomposition_table ----


def test_component_decomposition_returns_four_rows() -> None:
    scores, truth = _correlated(100)
    df = component_decomposition_table(
        scores, truth, "needs_consultation",
        compute_ci=True, n_bootstrap=200,
    )
    assert len(df) == 4
    assert set(df["score_column"]) == {
        "entropy_plateau", "voi_flatness",
        "distance_from_trajectory", "composite",
    }


def test_component_decomposition_includes_ci_by_default() -> None:
    scores, truth = _correlated(80)
    df = component_decomposition_table(
        scores, truth, "needs_consultation", n_bootstrap=100
    )
    assert "roc_auc_ci_low" in df.columns
    assert "roc_auc_ci_high" in df.columns


def test_component_decomposition_no_ci_when_disabled() -> None:
    scores, truth = _correlated(80)
    df = component_decomposition_table(
        scores, truth, "needs_consultation", compute_ci=False
    )
    assert "roc_auc_ci_low" not in df.columns


# ---- Determinism / sort order ----


def test_deferral_auc_output_sorted_by_score_column() -> None:
    scores, truth = _correlated(50)
    df = deferral_auc(
        scores, truth, "needs_consultation",
        score_columns=("voi_flatness", "composite", "entropy_plateau"),
    )
    assert list(df["score_column"]) == sorted(df["score_column"])
