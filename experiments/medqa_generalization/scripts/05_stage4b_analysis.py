#!/usr/bin/env python
"""Stage-4b analysis: pre-registered predictions + exploratory analyses.

Consumes artifacts produced by ``04_pipeline_validation_llama_cpp.py``
when run with ``--benchmark mmlu --mmlu-subject <subject>``. Evaluates
the seven pre-registered sub-predictions (P1, P2, P3, P4, P5a, P5b,
P6) and the five exploratory analyses (E1, E2, E3, E4, E5) from
``docs/decisions/stage_4b_mmlu_cross_benchmark_pre_design_notes.md``.

Outputs:
- ``predictions_outcomes.csv``: per-prediction pass/fail with thresholds
- ``per_component_auc.csv``: sign-aware AUC + bootstrap CI per scorer
- ``composite_sweep.csv``: AUC for orig-3, corrected-3, mass-flipped,
  all-flipped, mean_H_only constructions
- ``mass_capture_shape.json``: stratified stats, MW/KS, bottom-decile lift
- ``b_vs_c_complementarity.csv``: tertile contingency + lift
- ``entropy_summaries.csv``: AUC for mean, final, max, prior, step-k
- Plots (PNG): histogram, ECDF, per-position ribbons, AUC bars
- ``results.json``: structured aggregate of all numerical findings

Usage:
    python 05_stage4b_analysis.py \\
        --artifact-dir ~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law \\
        --output-dir ~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law-analysis \\
        --smoke-mean-entropy 0.72 \\
        --n-bootstrap 5000

The smoke-mean-entropy default of 0.72 is the smoke point estimate on
professional_law from the post-smoke analysis. Override for other
subjects when running stage-4c.

Console output prints the predictions table and key numerical findings;
the structured artifacts are for the writeup script's consumption.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score


# ============================================================
# Constants from the pre-design notes
# ============================================================

SUBJECT_DEFAULT = "professional_law"

# Pre-registered thresholds (do not modify post-hoc)
P1_AUC_THRESHOLD = 0.60
P1_CI_LOWER_THRESHOLD = 0.55
P2_DELTA_THRESHOLD = 0.05  # mean_H ≥ corrected-3 + 0.05
P3_CONSISTENCY_BAND = 0.10  # |full - smoke| ≤ 0.10
P4_DISABLED_CI_UPPER = 0.55  # distance/slope CI low must be < this
P4_ENABLED_CI_LOWER = 0.55  # mean_H/final_H CI low must be > this
P5A_DELTA_MEDIAN_THRESHOLD = 0.02
P5A_COHEN_D_THRESHOLD = 0.15
P5B_LIFT_LOW = 5.0  # percentage points
P5B_LIFT_HIGH = 15.0
P6_FAILURE_RATE_THRESHOLD = 0.05

# Bonferroni: 7 sub-predictions × 1 confirmatory subject = 7 tests
BONFERRONI_ALPHA = 0.05 / 7

# Plotting
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================
# Data structures
# ============================================================


@dataclass
class AucResult:
    raw: float
    sign_aware: float
    direction: str
    ci_low: float
    ci_high: float
    n_pos: int
    n_neg: int


@dataclass
class PredictionOutcome:
    name: str
    description: str
    threshold: str
    measured: str
    held: bool
    p_value: float | None = None
    bonferroni_significant: bool | None = None


# ============================================================
# Helpers
# ============================================================


def auc_with_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> AucResult:
    """Sign-aware AUC + percentile bootstrap CI, oriented to point estimate."""
    if len(set(y)) < 2:
        return AucResult(
            raw=float("nan"),
            sign_aware=float("nan"),
            direction="?",
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_pos=int((y == 1).sum()),
            n_neg=int((y == 0).sum()),
        )
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            boots[i] = np.nan
            continue
        boots[i] = roc_auc_score(y[idx], scores[idx])
    point = roc_auc_score(y, scores)
    sa = max(point, 1 - point)
    direction = "greater" if point >= 0.5 else "less"
    sa_boots = boots if direction == "greater" else 1 - boots
    return AucResult(
        raw=float(point),
        sign_aware=float(sa),
        direction=direction,
        ci_low=float(np.nanpercentile(sa_boots, 2.5)),
        ci_high=float(np.nanpercentile(sa_boots, 97.5)),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
    )


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def rank_pct(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.5])
    return rankdata(values, method="average") / n


def shannon_entropy_bits(distribution: dict[str, float]) -> float:
    if not distribution:
        return 0.0
    ps = [p for p in distribution.values() if p > 0]
    return -sum(p * math.log2(p) for p in ps) if ps else 0.0


# ============================================================
# Data loading
# ============================================================


def load_full_data(artifact_dir: Path) -> dict[str, Any]:
    """Load all artifacts needed for stage-4b analysis."""
    sigs_path = artifact_dir / "condition_C_artifact" / "signature_scores.csv"
    if not sigs_path.exists():
        raise FileNotFoundError(
            f"signature_scores.csv not found at {sigs_path} — "
            f"check that the run completed."
        )
    sigs = pd.read_csv(sigs_path)
    if "mean_entropy" not in sigs.columns:
        print(
            "WARNING: mean_entropy column missing from signature_scores.csv "
            "(pre-ae3bde1 artifact?). Computing from cached states.",
            file=sys.stderr,
        )

    pr_path = artifact_dir / "partial_results.json"
    pr = json.loads(pr_path.read_text())
    c_results = pd.DataFrame(pr["C"]).rename(columns={"question_id": "trajectory_id"})

    gt_path = artifact_dir / "condition_c_cached" / "trajectories.parquet"
    gt = pd.read_parquet(gt_path)[["trajectory_id", "primary_label"]].rename(
        columns={"primary_label": "correct_letter"}
    )

    states_path = artifact_dir / "condition_c_cached" / "states.parquet"
    states = pd.read_parquet(states_path)

    # Merge into one dataframe
    df = sigs.merge(gt, on="trajectory_id").merge(
        c_results[["trajectory_id", "predicted_answer", "success"]],
        on="trajectory_id",
    )
    df["wrong"] = (df["predicted_answer"] != df["correct_letter"]).astype(int)

    # Per-position series
    pos_series_mc = {}
    pos_series_h = {}
    for tid in df["trajectory_id"]:
        ts = states[states["trajectory_id"] == tid].sort_values("position")
        pos_series_mc[tid] = ts["mass_capture"].to_numpy(dtype=np.float64)
        # Per-position entropy from hypothesis_distribution_json
        h_per = []
        for hd_json in ts["hypothesis_distribution_json"].tolist():
            if hd_json:
                hd = json.loads(hd_json)
                h_per.append(shannon_entropy_bits(hd))
            else:
                h_per.append(float("nan"))
        pos_series_h[tid] = np.array(h_per)
    df["mc_positions"] = df["trajectory_id"].map(pos_series_mc)
    df["h_positions"] = df["trajectory_id"].map(pos_series_h)
    df["n_positions"] = df["mc_positions"].apply(len)

    # Compute final_entropy from last position
    df["final_entropy"] = df["h_positions"].apply(
        lambda arr: arr[-1] if len(arr) > 0 else float("nan")
    )
    # Compute initial (prior) entropy from position 0
    df["initial_entropy"] = df["h_positions"].apply(
        lambda arr: arr[0] if len(arr) > 0 else float("nan")
    )
    # Compute max_entropy across positions
    df["max_entropy"] = df["h_positions"].apply(
        lambda arr: float(np.nanmax(arr)) if len(arr) > 0 else float("nan")
    )

    # Backfill mean_entropy if absent (pre-ae3bde1 artifacts: stage-4a N=1273)
    if "mean_entropy" not in df.columns:
        df["mean_entropy"] = df["h_positions"].apply(
            lambda arr: float(np.nanmean(arr)) if len(arr) > 0 else float("nan")
        )

    # Condition B confidence (verbalised)
    if "B" in pr:
        b_results = pd.DataFrame(pr["B"]).rename(columns={"question_id": "trajectory_id"})
        b_df = b_results[["trajectory_id", "deferral_signal", "success"]].rename(
            columns={"deferral_signal": "b_confidence", "success": "b_success"}
        )
        df = df.merge(b_df, on="trajectory_id", how="left")

    return {
        "df": df,
        "partial_results": pr,
        "states": states,
        "n_total_partial": len(c_results),
        "n_success_c": int(c_results["success"].sum()) if "success" in c_results.columns else len(c_results),
    }


# ============================================================
# Pre-registered confirmatory predictions
# ============================================================


def evaluate_predictions(
    data: dict[str, Any],
    smoke_mean_entropy: float,
    n_boot: int,
) -> tuple[list[PredictionOutcome], dict[str, AucResult]]:
    df = data["df"]
    y = df["wrong"].to_numpy()
    outcomes: list[PredictionOutcome] = []

    # Per-component AUCs (used by P1, P4)
    component_aucs = {}
    for comp in [
        "mean_entropy",
        "final_entropy",
        "initial_entropy",
        "max_entropy",
        "entropy_plateau",
        "voi_flatness",
        "distance_from_trajectory",
        "mass_capture_mean",
        "mass_capture_min",
        "composite",
    ]:
        if comp in df.columns:
            scores = df[comp].to_numpy(dtype=np.float64)
            mask = ~np.isnan(scores)
            if mask.sum() == 0:
                continue
            component_aucs[comp] = auc_with_ci(y[mask], scores[mask], n_boot=n_boot)

    # ---- P1: mean_entropy primary ----
    me = component_aucs.get("mean_entropy")
    if me is not None:
        p1_held = (
            me.sign_aware > P1_AUC_THRESHOLD
            and me.ci_low > P1_CI_LOWER_THRESHOLD
            and me.direction == "greater"
        )
        outcomes.append(PredictionOutcome(
            name="P1",
            description="mean_entropy primary deferral signal",
            threshold=(
                f"sign-aware AUC > {P1_AUC_THRESHOLD}, "
                f"CI low > {P1_CI_LOWER_THRESHOLD}, direction=greater"
            ),
            measured=(
                f"sign-aware AUC = {me.sign_aware:.3f} "
                f"[{me.ci_low:.3f}, {me.ci_high:.3f}], dir={me.direction}"
            ),
            held=p1_held,
        ))

    # ---- P2: corrected-3 composite as exploratory ----
    if all(c in df.columns for c in ["distance_from_trajectory", "final_entropy", "entropy_plateau"]):
        rp_dist = rank_pct(df["distance_from_trajectory"].to_numpy())
        rp_finH = rank_pct(df["final_entropy"].to_numpy())
        rp_inv_slope = rank_pct(-df["entropy_plateau"].to_numpy())
        corrected3 = (rp_dist + rp_finH + rp_inv_slope) / 3
        c3_auc = auc_with_ci(y, corrected3, n_boot=n_boot)
        if me is not None:
            delta = me.sign_aware - c3_auc.sign_aware
            p2_held = delta >= P2_DELTA_THRESHOLD
            outcomes.append(PredictionOutcome(
                name="P2",
                description="corrected-3 composite as exploratory baseline (mean_H dominates)",
                threshold=f"mean_entropy_AUC ≥ corrected-3_AUC + {P2_DELTA_THRESHOLD}",
                measured=(
                    f"mean_H = {me.sign_aware:.3f}, corrected-3 = {c3_auc.sign_aware:.3f}, "
                    f"Δ = {delta:+.3f}"
                ),
                held=p2_held,
            ))
        component_aucs["corrected_3"] = c3_auc

    # ---- P3: smoke-to-full-N consistency ----
    if me is not None:
        delta_smoke = abs(me.sign_aware - smoke_mean_entropy)
        p3_held = delta_smoke <= P3_CONSISTENCY_BAND
        outcomes.append(PredictionOutcome(
            name="P3",
            description="smoke-to-full-N consistency on mean_entropy",
            threshold=(
                f"|full-N − smoke| ≤ {P3_CONSISTENCY_BAND}; "
                f"acceptable band: [{smoke_mean_entropy - P3_CONSISTENCY_BAND:.2f}, "
                f"{smoke_mean_entropy + P3_CONSISTENCY_BAND:.2f}]"
            ),
            measured=(
                f"smoke = {smoke_mean_entropy:.3f}, full-N = {me.sign_aware:.3f}, "
                f"Δ = {delta_smoke:.3f}"
            ),
            held=p3_held,
        ))

    # ---- P4: per-component CI behavior ----
    p4_disabled_ok = []
    p4_enabled_ok = []
    p4_lines = []
    for comp in ["distance_from_trajectory", "entropy_plateau"]:
        if comp in component_aucs:
            r = component_aucs[comp]
            ok = r.ci_low < P4_DISABLED_CI_UPPER
            p4_disabled_ok.append(ok)
            p4_lines.append(
                f"{comp}: CI low = {r.ci_low:.3f} "
                f"({'<' if ok else '≥'} {P4_DISABLED_CI_UPPER})"
            )
    for comp in ["mean_entropy", "final_entropy"]:
        if comp in component_aucs:
            r = component_aucs[comp]
            ok = r.ci_low > P4_ENABLED_CI_LOWER
            p4_enabled_ok.append(ok)
            p4_lines.append(
                f"{comp}: CI low = {r.ci_low:.3f} "
                f"({'>' if ok else '≤'} {P4_ENABLED_CI_LOWER})"
            )
    p4_held = all(p4_disabled_ok) and all(p4_enabled_ok)
    outcomes.append(PredictionOutcome(
        name="P4",
        description="per-component CI behavior (disabled vs enabled)",
        threshold=(
            f"distance/slope CI low < {P4_DISABLED_CI_UPPER}; "
            f"mean_H/final_H CI low > {P4_ENABLED_CI_LOWER}"
        ),
        measured="; ".join(p4_lines),
        held=p4_held,
    ))

    # ---- P5a: mass-capture central-tendency null ----
    correct_df = df[df.wrong == 0]
    wrong_df = df[df.wrong == 1]
    if len(correct_df) > 0 and len(wrong_df) > 0:
        cmm = correct_df["mass_capture_mean"].to_numpy()
        wmm = wrong_df["mass_capture_mean"].to_numpy()
        delta_median = np.median(wmm) - np.median(cmm)
        d = cohen_d(cmm, wmm)
        try:
            mw = mannwhitneyu(cmm, wmm, alternative="two-sided")
            mw_p = float(mw.pvalue)
        except ValueError:
            mw_p = float("nan")
        p5a_held = abs(delta_median) < P5A_DELTA_MEDIAN_THRESHOLD and abs(d) < P5A_COHEN_D_THRESHOLD
        outcomes.append(PredictionOutcome(
            name="P5a",
            description="mass-capture central-tendency null replicates",
            threshold=(
                f"|Δ_median| < {P5A_DELTA_MEDIAN_THRESHOLD} "
                f"AND |Cohen's d| < {P5A_COHEN_D_THRESHOLD}"
            ),
            measured=(
                f"Δ_median = {delta_median:+.4f}, Cohen's d = {d:+.3f}, MW p = {mw_p:.3f}"
            ),
            held=p5a_held,
            p_value=mw_p,
            bonferroni_significant=(mw_p < BONFERRONI_ALPHA) if not math.isnan(mw_p) else None,
        ))
    else:
        outcomes.append(PredictionOutcome(
            name="P5a",
            description="mass-capture central-tendency null replicates",
            threshold="(degenerate: no correct or no wrong subgroup)",
            measured="N/A",
            held=False,
        ))

    # ---- P5b: bottom-decile lift ----
    threshold_decile = float(np.percentile(df["mass_capture_mean"], 10))
    in_bottom = df["mass_capture_mean"] <= threshold_decile
    n_bottom = int(in_bottom.sum())
    n_wrong_bottom = int(((df.wrong == 1) & in_bottom).sum())
    base_wrong_rate = float(df.wrong.mean())
    decile_wrong_rate = n_wrong_bottom / n_bottom if n_bottom else 0.0
    lift_pp = (decile_wrong_rate - base_wrong_rate) * 100
    # Wilson lower bound for one-sided 95% CI on lift
    if n_bottom > 0:
        from scipy.stats import binomtest
        try:
            ci = binomtest(n_wrong_bottom, n_bottom).proportion_ci(
                confidence_level=0.90, method="wilson"
            )
            lift_ci_low_pp = (ci.low - base_wrong_rate) * 100
        except Exception:
            lift_ci_low_pp = float("nan")
    else:
        lift_ci_low_pp = float("nan")
    p5b_held = (
        P5B_LIFT_LOW <= lift_pp <= P5B_LIFT_HIGH
        and lift_ci_low_pp > 0.0
    )
    outcomes.append(PredictionOutcome(
        name="P5b",
        description="mass-capture bottom-decile concentration replicates",
        threshold=(
            f"bottom-decile lift in [{P5B_LIFT_LOW}pp, {P5B_LIFT_HIGH}pp] "
            f"AND one-sided 95% CI low > 0"
        ),
        measured=(
            f"lift = {lift_pp:+.2f}pp ({n_wrong_bottom}/{n_bottom} wrong vs "
            f"base {base_wrong_rate*100:.1f}%); CI low = {lift_ci_low_pp:+.2f}pp"
        ),
        held=p5b_held,
    ))

    # ---- P6: failure rate ----
    pr = data["partial_results"]
    n_total = len(pr["C"])
    n_failed = sum(1 for r in pr["C"] if not r.get("success", True))
    failure_rate = n_failed / n_total if n_total else 0.0
    p6_held = failure_rate < P6_FAILURE_RATE_THRESHOLD
    outcomes.append(PredictionOutcome(
        name="P6",
        description="measurement protocol robust on professional_law",
        threshold=f"failure rate < {P6_FAILURE_RATE_THRESHOLD * 100}%",
        measured=f"{n_failed}/{n_total} = {failure_rate * 100:.2f}%",
        held=p6_held,
    ))

    return outcomes, component_aucs


# ============================================================
# Exploratory analyses
# ============================================================


def composite_sweep(df: pd.DataFrame, n_boot: int) -> dict[str, AucResult]:
    """E1 — replicate the composite-construction sweep on full-N data."""
    y = df["wrong"].to_numpy()

    rp_dist = rank_pct(df["distance_from_trajectory"].to_numpy())
    rp_finH = rank_pct(df["final_entropy"].to_numpy())
    rp_inv_slope = rank_pct(-df["entropy_plateau"].to_numpy())
    rp_slope = rank_pct(df["entropy_plateau"].to_numpy())
    rp_inv_dist = rank_pct(1.0 - df["distance_from_trajectory"].to_numpy())
    rp_inv_finH = rank_pct(1.0 - df["final_entropy"].to_numpy())
    rp_inv_mc_mean = rank_pct(1.0 - df["mass_capture_mean"].to_numpy())
    rp_inv_mc_min = rank_pct(1.0 - df["mass_capture_min"].to_numpy())

    composites = {
        "orig-3": df["composite"].to_numpy() if "composite" in df.columns else None,
        "corrected-3": (rp_dist + rp_finH + rp_inv_slope) / 3,
        "mass-flipped":
            (rp_dist + rp_finH + rp_inv_slope + rp_inv_mc_mean + rp_inv_mc_min) / 5,
        "all-flipped": (rp_inv_dist + rp_inv_finH + rp_slope) / 3,
        "mean_H_only": df["mean_entropy"].to_numpy() if "mean_entropy" in df.columns else None,
    }
    out = {}
    for name, scores in composites.items():
        if scores is None:
            continue
        out[name] = auc_with_ci(y, scores, n_boot=n_boot)
    return out


def b_vs_c_complementarity(df: pd.DataFrame) -> dict[str, Any]:
    """E2 — Condition B (top-tertile confidence) × Condition C (top-tertile mean_entropy).

    Test: in the cell where B reports high confidence AND C reports high
    mean_entropy, is wrong-rate lift over base rate ≥ 5pp with one-sided
    95% CI low > 0?
    """
    if "b_confidence" not in df.columns or "mean_entropy" not in df.columns:
        return {"available": False, "reason": "missing b_confidence or mean_entropy"}

    df = df.dropna(subset=["b_confidence", "mean_entropy"]).copy()
    if len(df) == 0:
        return {"available": False, "reason": "no rows with both signals"}

    # Tertile cutoffs
    b_high_threshold = float(np.percentile(df["b_confidence"], 100 - 100 / 3))
    me_high_threshold = float(np.percentile(df["mean_entropy"], 100 - 100 / 3))

    df["b_high"] = df["b_confidence"] >= b_high_threshold
    df["me_high"] = df["mean_entropy"] >= me_high_threshold

    base_wrong_rate = float(df.wrong.mean())
    cell = df[df.b_high & df.me_high]
    n_cell = len(cell)
    n_wrong_cell = int(cell.wrong.sum())
    cell_wrong_rate = n_wrong_cell / n_cell if n_cell else 0.0
    lift_pp = (cell_wrong_rate - base_wrong_rate) * 100

    if n_cell > 0:
        from scipy.stats import binomtest
        try:
            ci = binomtest(n_wrong_cell, n_cell).proportion_ci(
                confidence_level=0.90, method="wilson"
            )
            lift_ci_low_pp = (ci.low - base_wrong_rate) * 100
        except Exception:
            lift_ci_low_pp = float("nan")
    else:
        lift_ci_low_pp = float("nan")

    return {
        "available": True,
        "b_high_threshold": b_high_threshold,
        "me_high_threshold": me_high_threshold,
        "base_wrong_rate_pp": base_wrong_rate * 100,
        "cell_size": n_cell,
        "cell_wrong_count": n_wrong_cell,
        "cell_wrong_rate_pp": cell_wrong_rate * 100,
        "lift_pp": lift_pp,
        "lift_ci_low_pp": lift_ci_low_pp,
        "lift_holds": lift_pp >= 5.0 and lift_ci_low_pp > 0.0,
    }


def alternative_entropy_summaries(df: pd.DataFrame, n_boot: int) -> dict[str, AucResult]:
    """E3 — AUC for mean_entropy, final_entropy, initial_entropy, max_entropy."""
    y = df["wrong"].to_numpy()
    out = {}
    for col in ["mean_entropy", "final_entropy", "initial_entropy", "max_entropy"]:
        if col in df.columns:
            scores = df[col].to_numpy(dtype=np.float64)
            mask = ~np.isnan(scores)
            if mask.sum() == 0:
                continue
            out[col] = auc_with_ci(y[mask], scores[mask], n_boot=n_boot)
    return out


def mass_capture_shape(df: pd.DataFrame) -> dict[str, Any]:
    """E4 — full mass-capture shape characterization (anchors P5a/P5b)."""
    correct = df[df.wrong == 0]
    wrong = df[df.wrong == 1]
    out = {}
    for col in ["mass_capture_mean", "mass_capture_min"]:
        if col not in df.columns:
            continue
        c = correct[col].to_numpy()
        w = wrong[col].to_numpy()
        if len(c) == 0 or len(w) == 0:
            out[col] = {"available": False}
            continue
        try:
            mw = mannwhitneyu(c, w, alternative="two-sided")
            ks = ks_2samp(c, w)
            mw_p = float(mw.pvalue)
            ks_p = float(ks.pvalue)
            ks_stat = float(ks.statistic)
        except ValueError:
            mw_p = ks_p = ks_stat = float("nan")
        out[col] = {
            "n_correct": len(c),
            "n_wrong": len(w),
            "correct_mean": float(c.mean()),
            "correct_median": float(np.median(c)),
            "correct_std": float(c.std()),
            "wrong_mean": float(w.mean()),
            "wrong_median": float(np.median(w)),
            "wrong_std": float(w.std()),
            "delta_mean": float(w.mean() - c.mean()),
            "delta_median": float(np.median(w) - np.median(c)),
            "cohen_d": cohen_d(c, w),
            "mw_p": mw_p,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
        }
    return out


def per_position_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """E5 — per-position median + IQR of entropy and mass_capture, by group."""
    correct = df[df.wrong == 0]
    wrong = df[df.wrong == 1]
    max_pos = int(df["n_positions"].max()) if len(df) else 0
    out = {"max_positions": max_pos, "by_position": []}
    for p in range(max_pos):
        c_h, w_h, c_mc, w_mc = [], [], [], []
        for arr in correct["h_positions"].tolist():
            if p < len(arr):
                c_h.append(arr[p])
        for arr in wrong["h_positions"].tolist():
            if p < len(arr):
                w_h.append(arr[p])
        for arr in correct["mc_positions"].tolist():
            if p < len(arr):
                c_mc.append(arr[p])
        for arr in wrong["mc_positions"].tolist():
            if p < len(arr):
                w_mc.append(arr[p])
        if len(c_h) < 10 or len(w_h) < 10:
            continue
        c_h = np.array(c_h); w_h = np.array(w_h)
        c_mc = np.array(c_mc); w_mc = np.array(w_mc)
        out["by_position"].append({
            "position": p,
            "n_correct": len(c_h),
            "n_wrong": len(w_h),
            "h_correct_median": float(np.median(c_h)),
            "h_wrong_median": float(np.median(w_h)),
            "h_delta_median": float(np.median(w_h) - np.median(c_h)),
            "mc_correct_median": float(np.median(c_mc)),
            "mc_wrong_median": float(np.median(w_mc)),
            "mc_delta_median": float(np.median(w_mc) - np.median(c_mc)),
        })
    return out


# ============================================================
# Plots
# ============================================================


def write_plots(df: pd.DataFrame, output_dir: Path, subject: str) -> None:
    correct = df[df.wrong == 0]
    wrong = df[df.wrong == 1]

    # Plot A: mass_capture histograms + ECDFs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j, col in enumerate(["mass_capture_mean", "mass_capture_min"]):
        bins = np.linspace(0.0, 1.0, 51)
        axes[0, j].hist(correct[col], bins=bins, alpha=0.6,
                        label=f"correct (n={len(correct)})",
                        color="#2a9d8f", density=True)
        axes[0, j].hist(wrong[col], bins=bins, alpha=0.6,
                        label=f"wrong (n={len(wrong)})",
                        color="#e76f51", density=True)
        axes[0, j].set_xlabel(col); axes[0, j].set_ylabel("density")
        axes[0, j].legend(); axes[0, j].grid(alpha=0.3)
        c_sorted = np.sort(correct[col].to_numpy())
        w_sorted = np.sort(wrong[col].to_numpy())
        axes[1, j].step(c_sorted, np.arange(1, len(c_sorted) + 1) / len(c_sorted),
                        where="post", label="correct", color="#2a9d8f", linewidth=2)
        axes[1, j].step(w_sorted, np.arange(1, len(w_sorted) + 1) / len(w_sorted),
                        where="post", label="wrong", color="#e76f51", linewidth=2)
        axes[1, j].set_xlabel(col); axes[1, j].set_ylabel("ECDF")
        axes[1, j].legend(); axes[1, j].grid(alpha=0.3)
    plt.suptitle(f"{subject}: mass-capture by correct/wrong", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_mass_capture_shape.png", dpi=120)
    plt.close()

    # Plot B: per-position entropy + mass_capture ribbons
    max_pos = int(df["n_positions"].max())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, ylabel in [
        (axes[0], "h_positions", "entropy (bits)"),
        (axes[1], "mc_positions", "mass_capture"),
    ]:
        for label, g, color in [
            ("correct", correct, "#2a9d8f"),
            ("wrong", wrong, "#e76f51"),
        ]:
            mat = np.full((len(g), max_pos), np.nan)
            for i, arr in enumerate(g[key].tolist()):
                mat[i, :len(arr)] = arr
            n_per_pos = (~np.isnan(mat)).sum(axis=0)
            valid = n_per_pos >= max(10, int(0.05 * len(g)))
            pos = np.arange(max_pos)
            median = np.nanmedian(mat, axis=0)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            ax.plot(pos[valid], median[valid], "-o",
                    label=f"{label} median", color=color, linewidth=2)
            ax.fill_between(pos[valid], q25[valid], q75[valid],
                            alpha=0.25, color=color)
        ax.set_xlabel("position (0=prior)"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f"{subject}: per-position trajectories by correct/wrong", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_per_position_trajectories.png", dpi=120)
    plt.close()


# ============================================================
# Output formatting
# ============================================================


def write_predictions_table(
    outcomes: list[PredictionOutcome],
    output_dir: Path,
) -> None:
    rows = [asdict(o) for o in outcomes]
    pd.DataFrame(rows).to_csv(output_dir / "predictions_outcomes.csv", index=False)


def write_per_component_auc(
    aucs: dict[str, AucResult],
    output_dir: Path,
) -> None:
    rows = []
    for name, r in aucs.items():
        rows.append({
            "scorer": name,
            "raw_auc": r.raw,
            "sign_aware_auc": r.sign_aware,
            "direction": r.direction,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "n_pos": r.n_pos,
            "n_neg": r.n_neg,
        })
    pd.DataFrame(rows).to_csv(output_dir / "per_component_auc.csv", index=False)


def write_composite_sweep(
    sweep: dict[str, AucResult],
    output_dir: Path,
) -> None:
    rows = []
    for name, r in sweep.items():
        rows.append({
            "construction": name,
            "sign_aware_auc": r.sign_aware,
            "direction": r.direction,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
        })
    pd.DataFrame(rows).to_csv(output_dir / "composite_sweep.csv", index=False)


def write_results_json(
    outcomes: list[PredictionOutcome],
    component_aucs: dict[str, AucResult],
    sweep: dict[str, AucResult],
    bvc: dict[str, Any],
    entropy_summaries: dict[str, AucResult],
    mc_shape: dict[str, Any],
    per_pos: dict[str, Any],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {
        "metadata": metadata,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "predictions": [asdict(o) for o in outcomes],
        "per_component_auc": {
            k: asdict(v) for k, v in component_aucs.items()
        },
        "composite_sweep": {k: asdict(v) for k, v in sweep.items()},
        "b_vs_c_complementarity": bvc,
        "entropy_summaries": {
            k: asdict(v) for k, v in entropy_summaries.items()
        },
        "mass_capture_shape": mc_shape,
        "per_position_diagnostics": per_pos,
    }

    def _coerce(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} not serialisable")

    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, default=_coerce)
    )


def print_console_summary(
    outcomes: list[PredictionOutcome],
    component_aucs: dict[str, AucResult],
    sweep: dict[str, AucResult],
    bvc: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    print()
    print("=" * 90)
    print(f"  Stage-4b analysis: {metadata['subject']}, N={metadata['n_total']}")
    print(f"  Accuracy: {metadata['accuracy']:.3f}, "
          f"n_wrong: {metadata['n_wrong']}, n_correct: {metadata['n_correct']}")
    print("=" * 90)

    print("\nPRE-REGISTERED PREDICTIONS")
    print("-" * 90)
    for o in outcomes:
        status = "HOLDS" if o.held else "FAILS"
        symbol = "+" if o.held else "-"
        print(f"  {symbol} {o.name}  [{status}]  {o.description}")
        print(f"     threshold: {o.threshold}")
        print(f"     measured:  {o.measured}")

    held_count = sum(1 for o in outcomes if o.held)
    print(f"\n  Summary: {held_count}/{len(outcomes)} predictions hold")
    print(f"  Bonferroni-corrected α = {BONFERRONI_ALPHA:.4f} (per-test)")

    print("\nPER-COMPONENT SIGN-AWARE AUC (95% bootstrap CI)")
    print("-" * 90)
    print(f"  {'scorer':32s} {'AUC':>6s} {'CI':>16s} {'dir':>8s}")
    for name, r in sorted(component_aucs.items(),
                          key=lambda x: -x[1].sign_aware
                          if not math.isnan(x[1].sign_aware) else 0):
        if math.isnan(r.sign_aware):
            continue
        print(f"  {name:32s} {r.sign_aware:6.3f} "
              f"[{r.ci_low:5.3f}, {r.ci_high:5.3f}] {r.direction:>8s}")

    print("\nCOMPOSITE-CONSTRUCTION SWEEP (E1)")
    print("-" * 90)
    print(f"  {'construction':24s} {'AUC':>6s} {'CI':>16s} {'dir':>8s}")
    for name, r in sweep.items():
        print(f"  {name:24s} {r.sign_aware:6.3f} "
              f"[{r.ci_low:5.3f}, {r.ci_high:5.3f}] {r.direction:>8s}")

    print("\nB-VS-C COMPLEMENTARITY (E2)")
    print("-" * 90)
    if bvc.get("available", False):
        print(f"  Top-tertile B-confidence × top-tertile mean_entropy cell:")
        print(f"    cell size:         {bvc['cell_size']}")
        print(f"    cell wrong-rate:   {bvc['cell_wrong_rate_pp']:.1f}%")
        print(f"    base wrong-rate:   {bvc['base_wrong_rate_pp']:.1f}%")
        print(f"    lift over base:    {bvc['lift_pp']:+.2f}pp "
              f"(CI low: {bvc['lift_ci_low_pp']:+.2f}pp)")
        print(f"    holds (≥+5pp, CI>0): "
              f"{'YES' if bvc.get('lift_holds', False) else 'NO'}")
    else:
        print(f"  Not available: {bvc.get('reason', 'unknown')}")
    print()


# ============================================================
# CLI
# ============================================================


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else "",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path.home() / "work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subject", default=SUBJECT_DEFAULT)
    parser.add_argument(
        "--smoke-mean-entropy",
        type=float,
        default=0.72,
        help=(
            "Smoke point estimate of mean_entropy AUC for P3 consistency check. "
            "Default 0.72 is the smoke result on professional_law."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Bootstrap samples for AUC CIs (default 5000 — methods-paper level)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        args.artifact_dir.parent / f"{args.artifact_dir.name}-analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading artifacts from: {args.artifact_dir}")
    print(f"Writing analysis to:    {output_dir}")
    data = load_full_data(args.artifact_dir)
    df = data["df"]
    metadata = {
        "subject": args.subject,
        "artifact_dir": str(args.artifact_dir),
        "n_total": len(df),
        "n_wrong": int(df.wrong.sum()),
        "n_correct": int((df.wrong == 0).sum()),
        "accuracy": float(1 - df.wrong.mean()),
        "smoke_mean_entropy": args.smoke_mean_entropy,
        "n_bootstrap": args.n_bootstrap,
    }

    outcomes, component_aucs = evaluate_predictions(
        data, args.smoke_mean_entropy, args.n_bootstrap
    )
    sweep = composite_sweep(df, args.n_bootstrap)
    bvc = b_vs_c_complementarity(df)
    entropy_summaries = alternative_entropy_summaries(df, args.n_bootstrap)
    mc_shape = mass_capture_shape(df)
    per_pos = per_position_diagnostics(df)

    write_predictions_table(outcomes, output_dir)
    write_per_component_auc(component_aucs, output_dir)
    write_composite_sweep(sweep, output_dir)
    write_plots(df, output_dir, args.subject)
    write_results_json(
        outcomes, component_aucs, sweep, bvc, entropy_summaries,
        mc_shape, per_pos, metadata, output_dir,
    )
    print_console_summary(outcomes, component_aucs, sweep, bvc, metadata)
    print(f"Analysis artifacts written to: {output_dir}")


if __name__ == "__main__":
    cli()
