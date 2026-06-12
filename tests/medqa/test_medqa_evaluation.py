"""Tests for MedQA-specific evaluation extensions."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from bsig.core.evaluation import EvaluationError, EvaluationWarning
from bsig.medqa import (
    condition_comparison,
    cross_domain_comparison,
    cross_llm_comparison,
    failure_mode_table,
    stratified_deferral_auc,
)


# ---- Fixtures ----


def _scores(n: int = 60, score_col: str = "composite", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            score_col: rng.uniform(0, 1, n).astype(np.float32),
        }
    )


def _scores_with_correlation(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High composite -> high probability of label=1. AUC > 0.5 expected."""
    rng = np.random.default_rng(seed)
    composite = rng.uniform(0, 1, n).astype(np.float32)
    labels = (composite > 0.5).astype(int)
    flip = rng.binomial(1, 0.1, n).astype(bool)
    labels = np.where(flip, 1 - labels, labels)
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
    return scores, truth


def _truth_with_strata(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, 0.5, n).astype(int),
            "usmle_step": ["step1" if i % 2 == 0 else "step2&3" for i in range(n)],
        }
    )


# ---- stratified_deferral_auc ----


def test_stratified_returns_one_row_per_stratum_per_score_column() -> None:
    scores = _scores(n=60)
    truth = _truth_with_strata(n=60)
    df = stratified_deferral_auc(
        scores, truth, "needs_consultation", stratum_column="usmle_step"
    )
    assert "usmle_step" in df.columns
    assert set(df["usmle_step"]) == {"step1", "step2&3"}
    assert len(df) == 2  # one row per stratum


def test_stratified_skips_strata_without_class_balance() -> None:
    """A stratum with all-positives or all-negatives is skipped."""
    truth = pd.DataFrame(
        {
            "trajectory_id": ["t1", "t2", "t3", "t4"],
            "needs_consultation": [1, 1, 1, 0],
            "usmle_step": ["step1", "step1", "step2&3", "step2&3"],
        }
    )
    scores = pd.DataFrame(
        {
            "trajectory_id": ["t1", "t2", "t3", "t4"],
            "composite": [0.9, 0.8, 0.5, 0.2],
        }
    )
    df = stratified_deferral_auc(
        scores, truth, "needs_consultation", stratum_column="usmle_step"
    )
    # step1 has all positives — skipped. step2&3 has both — kept.
    assert set(df["usmle_step"]) == {"step2&3"}


def test_stratified_raises_on_missing_stratum_column() -> None:
    scores = _scores(20)
    truth = _truth_with_strata(20).drop(columns=["usmle_step"])
    with pytest.raises(EvaluationError, match="stratum_column"):
        stratified_deferral_auc(
            scores, truth, "needs_consultation", stratum_column="usmle_step"
        )


def test_stratified_returns_empty_when_all_strata_skipped() -> None:
    truth = pd.DataFrame(
        {
            "trajectory_id": ["t1", "t2"],
            "needs_consultation": [1, 1],
            "usmle_step": ["step1", "step1"],
        }
    )
    scores = pd.DataFrame(
        {"trajectory_id": ["t1", "t2"], "composite": [0.5, 0.7]}
    )
    df = stratified_deferral_auc(
        scores, truth, "needs_consultation", stratum_column="usmle_step"
    )
    assert len(df) == 0
    assert "usmle_step" in df.columns


# ---- cross_llm_comparison ----


def test_cross_llm_one_row_per_llm() -> None:
    scores_qwen, truth = _scores_with_correlation(60)
    scores_llama, _ = _scores_with_correlation(60, seed=42)
    df = cross_llm_comparison(
        {"qwen2.5:7b": scores_qwen, "llama3.1:8b": scores_llama},
        truth,
        "needs_consultation",
        compute_ci=False,
    )
    assert set(df["llm_name"]) == {"qwen2.5:7b", "llama3.1:8b"}
    assert "roc_auc" in df.columns
    assert "pr_auc" in df.columns


def test_cross_llm_with_ci_includes_ci_columns() -> None:
    scores_a, truth = _scores_with_correlation(60)
    scores_b, _ = _scores_with_correlation(60, seed=42)
    df = cross_llm_comparison(
        {"a": scores_a, "b": scores_b},
        truth,
        "needs_consultation",
        compute_ci=True,
        n_bootstrap=200,
    )
    for col in ("roc_auc_ci_low", "roc_auc_ci_high"):
        assert col in df.columns


# ---- cross_domain_comparison ----


def test_cross_domain_takes_separate_truths() -> None:
    medqa_scores, medqa_truth = _scores_with_correlation(60)
    mmlu_scores, mmlu_truth = _scores_with_correlation(60, seed=42)
    df = cross_domain_comparison(
        scores_per_domain={"medqa": medqa_scores, "mmlu": mmlu_scores},
        ground_truths_per_domain={"medqa": medqa_truth, "mmlu": mmlu_truth},
        target_column="needs_consultation",
        compute_ci=False,
    )
    assert set(df["domain_name"]) == {"medqa", "mmlu"}


def test_cross_domain_rejects_mismatched_keys() -> None:
    medqa_scores, medqa_truth = _scores_with_correlation(60)
    with pytest.raises(EvaluationError, match="same keys"):
        cross_domain_comparison(
            scores_per_domain={"medqa": medqa_scores},
            ground_truths_per_domain={"mmlu": medqa_truth},
            target_column="needs_consultation",
            compute_ci=False,
        )


# ---- condition_comparison ----


def test_condition_comparison_default_score_columns() -> None:
    """Default mapping: A and B use deferral_signal, C uses composite."""
    rng = np.random.default_rng(0)
    n = 60
    a_scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "deferral_signal": np.full(n, 0.5),
        }
    )
    b_scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "deferral_signal": rng.uniform(0, 1, n),
        }
    )
    c_scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "composite": rng.uniform(0, 1, n),
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, 0.5, n).astype(int),
        }
    )
    df = condition_comparison(
        {"A": a_scores, "B": b_scores, "C": c_scores},
        truth,
        "needs_consultation",
        compute_ci=False,
    )
    assert set(df["condition_id"]) == {"A", "B", "C"}


def test_condition_comparison_subset_mask_filters() -> None:
    """subset_mask filters to a subset of trajectories."""
    rng = np.random.default_rng(0)
    n = 60
    c_scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "composite": rng.uniform(0, 1, n),
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, 0.5, n).astype(int),
        }
    )
    # Subset: even-indexed trajectories only
    mask = pd.Series([i % 2 == 0 for i in range(n)])
    df = condition_comparison(
        {"C": c_scores},
        truth,
        "needs_consultation",
        score_columns_per_condition={"C": "composite"},
        subset_mask=mask,
        compute_ci=False,
    )
    # Subset has 30 trajectories
    assert df.iloc[0]["n_pos"] + df.iloc[0]["n_neg"] == 30


def test_condition_comparison_constant_signal_yields_half_auc() -> None:
    """Condition A's flat-line baseline produces AUC = 0.5."""
    n = 60
    rng = np.random.default_rng(0)
    a_scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "deferral_signal": np.full(n, 0.5),
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, 0.5, n).astype(int),
        }
    )
    df = condition_comparison(
        {"A": a_scores}, truth, "needs_consultation", compute_ci=False
    )
    assert df.iloc[0]["roc_auc"] == pytest.approx(0.5)


def test_condition_comparison_rejects_unknown_condition() -> None:
    """Condition not in score_columns_per_condition raises."""
    rng = np.random.default_rng(0)
    n = 30
    scores = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "composite": rng.uniform(0, 1, n),
        }
    )
    truth = pd.DataFrame(
        {
            "trajectory_id": [f"t{i}" for i in range(n)],
            "needs_consultation": rng.binomial(1, 0.5, n).astype(int),
        }
    )
    with pytest.raises(EvaluationError, match="No score_column mapping"):
        condition_comparison(
            {"NonexistentConditionX": scores},
            truth,
            "needs_consultation",
            compute_ci=False,
        )


# ---- failure_mode_table ----


def test_failure_mode_table_sorted_by_score_descending() -> None:
    scores, truth = _scores_with_correlation(20)
    table = failure_mode_table(
        scores, truth, "needs_consultation", top_n=10
    )
    assert len(table) == 10
    scores_descending = table["score"].to_numpy()
    assert (scores_descending[:-1] >= scores_descending[1:]).all()


def test_failure_mode_table_includes_correctness_flag() -> None:
    scores, truth = _scores_with_correlation(20)
    table = failure_mode_table(scores, truth, "needs_consultation")
    assert "high_score_correct_outcome" in table.columns
    assert table.dtypes["high_score_correct_outcome"] == bool


def test_failure_mode_table_score_percentile_in_unit_interval() -> None:
    scores, truth = _scores_with_correlation(20)
    table = failure_mode_table(scores, truth, "needs_consultation", top_n=20)
    assert (table["score_percentile"] >= 0.0).all()
    assert (table["score_percentile"] <= 1.0).all()


def test_failure_mode_table_empty_input_returns_empty() -> None:
    scores = pd.DataFrame(
        {"trajectory_id": pd.Series(dtype=object), "composite": pd.Series(dtype="float64")}
    )
    truth = pd.DataFrame(
        {"trajectory_id": pd.Series(dtype=object), "needs_consultation": pd.Series(dtype=int)}
    )
    table = failure_mode_table(scores, truth, "needs_consultation")
    assert len(table) == 0


# ---- build_faiss_indices_from_visits ----


def test_build_faiss_from_visits_groups_by_timestep() -> None:
    pytest.importorskip("faiss")
    from bsig.core.persistence import build_faiss_indices_from_visits

    visits = pd.DataFrame(
        {
            "visit_seq": [0, 1, 2, 3, 4, 5],
            "trajectory_id": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "timestep": [0, 1, 2, 0, 1, 2],
            "node_id": [f"n{i}" for i in range(6)],
            "embedding": [np.array([float(i), 0.0], dtype=np.float32) for i in range(6)],
        }
    )
    indices, manifest = build_faiss_indices_from_visits(
        visits, dimension=2, embedding_model="test"
    )
    assert set(indices) == {0, 1, 2}
    for idx in indices.values():
        assert idx.ntotal == 2  # 2 visits per timestep
    assert manifest["embedding_model"] == "test"
    assert manifest["dimension"] == 2
    assert manifest["index_type"] == "IndexFlatIP"


def test_build_faiss_from_empty_visits_returns_empty() -> None:
    pytest.importorskip("faiss")
    from bsig.core.persistence import build_faiss_indices_from_visits

    empty = pd.DataFrame(
        {
            "visit_seq": pd.Series(dtype="int64"),
            "trajectory_id": pd.Series(dtype=object),
            "timestep": pd.Series(dtype="int32"),
            "node_id": pd.Series(dtype=object),
            "embedding": pd.Series(dtype=object),
        }
    )
    indices, manifest = build_faiss_indices_from_visits(
        empty, dimension=2, embedding_model="test"
    )
    assert indices == {}
    assert manifest["dimension"] == 2
