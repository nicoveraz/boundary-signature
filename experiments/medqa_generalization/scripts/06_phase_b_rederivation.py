#!/usr/bin/env python
"""Phase-B uncertainty-signal re-derivation on cached schema-v3 trajectories.

Per ADR-0009 + the *measurement vs computation* methodology principle
(M10), the new uncertainty scorers (p_max, entropy_full, top_k_mass,
gap_top2, gap_top1_topK) are pure derivations from cached top-K
logprobs. This script re-derives them on cached stage-4b
professional_law N=1534 trajectories without re-running inference,
and tests the pre-registered E1/E2/E3 exploratory predictions
from ADR-0009.

**Schema-v3 limitation**: stage-4a MedQA N=1273 was collected
*before* schema-v3 (top-K logprobs preservation) went in, so the
cached states do not carry ``top_k_logprobs_json``. This script
runs on stage-4b professional_law (and the stage-4b smoke
artifacts where schema-v3 was active). Cross-domain extension to
MedQA requires re-running stage-4a inference under schema-v3,
which is a separate workstream (~16h on M1 Pro).

**Pre-registered E1/E2/E3 thresholds** (ADR-0009 §"pre-registered
exploratory predictions"):

- **E1**: ``mean_p_max`` AUC ∈ [0.55, 0.75]. Outside the range
  (above 0.75 = "candidate complementary"; below 0.55 = "framework
  signal is specifically entropy-based, not peak-sharpness-based").
- **E2**: ``mean_top_k_mass`` (top-10) AUC underperforms
  ``mean_entropy`` by ≤ 0.05 (Δ ≥ -0.05) AND Spearman r ≥ 0.5 with
  ``mean_entropy``. If correlation is low, top-K mass is measuring
  something different and warrants follow-up.
- **E3**: ``mean_gap_top2`` AUC ≥ 0.60 AND bottom-decile of
  ``mean_gap_top2`` shows ≥ +5pp wrong-rate lift over base rate.

All AUCs sign-aware (max(roc_auc, 1-roc_auc)) with bootstrap CIs at
n=2000 (smaller than the headline 5000 because exploratory).

Usage:

    python 06_phase_b_rederivation.py \\
        --artifact-dir ~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law \\
        --output-dir ~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law-phaseB

Output:
- ``phase_b_predictions.csv``: E1/E2/E3 outcomes
- ``phase_b_per_scorer_auc.csv``: AUC + bootstrap CI for each new scorer
- ``phase_b_correlations.csv``: Spearman ρ matrix among new scorers + mean_entropy
- ``phase_b_results.json``: structured aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# Pre-registered thresholds (ADR-0009)
# ============================================================

E1_AUC_LOW = 0.55
E1_AUC_HIGH = 0.75
E2_DELTA_MIN = -0.05  # mean_top_k_mass AUC ≥ mean_entropy − 0.05
E2_SPEARMAN_MIN = 0.5
E3_AUC_THRESHOLD = 0.60
E3_BOTTOM_DECILE_LIFT_MIN = 5.0  # percentage points


# ============================================================
# Phase B scorers — re-imported from bsig.core.signature
# ============================================================
# We use the source-of-truth implementations to avoid drift.

from bsig.core.signature import (  # noqa: E402
    entropy_full_from_top_k,
    gap_top1_top_k_from_top_k,
    gap_top2_from_top_k,
    p_max_from_top_k,
    top_k_mass_from_top_k,
)


# ============================================================
# Data structures
# ============================================================


@dataclass
class ScorerAuc:
    name: str
    point: float
    sign_aware: float
    direction: str
    ci_low: float
    ci_high: float


@dataclass
class PredictionOutcome:
    name: str
    description: str
    threshold: str
    measured: str
    held: bool


# ============================================================
# Helpers
# ============================================================


def auc_with_ci(
    y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> ScorerAuc:
    if len(set(y)) < 2:
        return ScorerAuc(
            name="?", point=float("nan"), sign_aware=float("nan"),
            direction="?", ci_low=float("nan"), ci_high=float("nan"),
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
    return ScorerAuc(
        name="?",
        point=float(point),
        sign_aware=float(sa),
        direction=direction,
        ci_low=float(np.nanpercentile(sa_boots, 2.5)),
        ci_high=float(np.nanpercentile(sa_boots, 97.5)),
    )


def parse_top_k_json(json_str: str | None) -> Mapping[str, float]:
    if not json_str:
        return {}
    return json.loads(json_str)


def shannon_entropy_bits(distribution: Mapping[str, float]) -> float:
    if not distribution:
        return 0.0
    ps = [p for p in distribution.values() if p > 0]
    return -sum(p * math.log2(p) for p in ps) if ps else 0.0


# ============================================================
# Main analysis
# ============================================================


def analyse(artifact_dir: Path, output_dir: Path, n_boot: int) -> None:
    print(f"Loading artifacts from: {artifact_dir}")
    print(f"Writing analysis to:    {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pr = json.loads((artifact_dir / "partial_results.json").read_text())
    sigs = pd.read_csv(artifact_dir / "condition_C_artifact" / "signature_scores.csv")
    gt = pd.read_parquet(artifact_dir / "condition_c_cached" / "trajectories.parquet")[
        ["trajectory_id", "primary_label"]
    ].rename(columns={"primary_label": "correct_letter"})
    states = pd.read_parquet(artifact_dir / "condition_c_cached" / "states.parquet")

    if "top_k_logprobs_json" not in states.columns:
        print(
            "ERROR: cached states do not carry top_k_logprobs_json. "
            "This artifact predates schema-v3 (ADR-0008 follow-up). "
            "Re-running inference under schema-v3 is required for "
            "Phase B re-derivation.",
            file=sys.stderr,
        )
        sys.exit(1)

    c_results = pd.DataFrame(pr["C"]).rename(
        columns={"question_id": "trajectory_id"}
    )
    df = sigs.merge(gt, on="trajectory_id").merge(
        c_results[["trajectory_id", "predicted_answer"]], on="trajectory_id"
    )
    df["wrong"] = (df["predicted_answer"] != df["correct_letter"]).astype(int)
    print(
        f"N={len(df)}, accuracy={1 - df.wrong.mean():.3f}, "
        f"n_wrong={int(df.wrong.sum())}, n_correct={int((df.wrong == 0).sum())}"
    )

    # ---- Per-trajectory aggregations of Phase-B scorers ----
    states_by_id: dict[str, pd.DataFrame] = {
        tid: states[states["trajectory_id"] == tid].sort_values("position")
        for tid in df["trajectory_id"]
    }

    new_columns = {
        "mean_p_max": [],
        "min_p_max": [],
        "mean_entropy_full": [],
        "mean_top_k_mass": [],
        "min_top_k_mass": [],
        "mean_gap_top2": [],
        "min_gap_top2": [],
        "mean_gap_top1_top10": [],
    }

    for tid in df["trajectory_id"]:
        ts = states_by_id[tid]
        per_p_max: list[float] = []
        per_h_full: list[float] = []
        per_top_k_mass: list[float] = []
        per_gap_top2: list[float] = []
        per_gap_top1_top10: list[float] = []
        for tk_json in ts["top_k_logprobs_json"].tolist():
            tk = parse_top_k_json(tk_json)
            if not tk:
                continue
            per_p_max.append(p_max_from_top_k(tk))
            per_h_full.append(entropy_full_from_top_k(tk))
            per_top_k_mass.append(top_k_mass_from_top_k(tk, k=10))
            per_gap_top2.append(gap_top2_from_top_k(tk))
            per_gap_top1_top10.append(gap_top1_top_k_from_top_k(tk, k=10))

        def safe_agg(values: list[float], how: str, default: float = float("nan")) -> float:
            if not values:
                return default
            if how == "mean":
                return float(np.mean(values))
            if how == "min":
                return float(min(values))
            raise ValueError(how)

        new_columns["mean_p_max"].append(safe_agg(per_p_max, "mean"))
        new_columns["min_p_max"].append(safe_agg(per_p_max, "min"))
        new_columns["mean_entropy_full"].append(safe_agg(per_h_full, "mean"))
        new_columns["mean_top_k_mass"].append(safe_agg(per_top_k_mass, "mean"))
        new_columns["min_top_k_mass"].append(safe_agg(per_top_k_mass, "min"))
        new_columns["mean_gap_top2"].append(safe_agg(per_gap_top2, "mean"))
        new_columns["min_gap_top2"].append(safe_agg(per_gap_top2, "min"))
        new_columns["mean_gap_top1_top10"].append(
            safe_agg(per_gap_top1_top10, "mean")
        )

    for col, values in new_columns.items():
        df[col] = values

    # ---- AUCs ----
    y = df["wrong"].to_numpy()
    scorer_aucs: dict[str, ScorerAuc] = {}
    for col in [
        "mean_entropy",  # baseline (existing)
        "mean_p_max",
        "min_p_max",
        "mean_entropy_full",
        "mean_top_k_mass",
        "min_top_k_mass",
        "mean_gap_top2",
        "min_gap_top2",
        "mean_gap_top1_top10",
    ]:
        if col not in df.columns:
            print(f"  skipping {col} (column missing)")
            continue
        scores = df[col].to_numpy(dtype=np.float64)
        mask = ~np.isnan(scores)
        if mask.sum() < 10:
            print(f"  skipping {col} (insufficient non-NaN values)")
            continue
        result = auc_with_ci(y[mask], scores[mask], n_boot=n_boot)
        result.name = col
        scorer_aucs[col] = result

    # ---- Spearman correlations among new scorers + mean_entropy ----
    corr_columns = [
        "mean_entropy",
        "mean_p_max",
        "mean_entropy_full",
        "mean_top_k_mass",
        "mean_gap_top2",
        "mean_gap_top1_top10",
    ]
    available = [c for c in corr_columns if c in df.columns]
    sub = df[available].dropna()
    if len(sub) >= 10:
        corr_matrix = sub.corr(method="spearman")
    else:
        corr_matrix = pd.DataFrame()

    # ---- Pre-registered E1/E2/E3 ----
    outcomes: list[PredictionOutcome] = []

    # E1: mean_p_max AUC ∈ [0.55, 0.75]
    if "mean_p_max" in scorer_aucs:
        r = scorer_aucs["mean_p_max"]
        held = E1_AUC_LOW <= r.sign_aware <= E1_AUC_HIGH
        outcomes.append(PredictionOutcome(
            name="E1",
            description="mean_p_max AUC in [0.55, 0.75]",
            threshold=f"sign-aware AUC ∈ [{E1_AUC_LOW}, {E1_AUC_HIGH}]",
            measured=f"sign-aware AUC = {r.sign_aware:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}], dir={r.direction}",
            held=held,
        ))

    # E2: mean_top_k_mass underperforms mean_entropy by ≤ 0.05 AND Spearman ≥ 0.5
    if "mean_top_k_mass" in scorer_aucs and "mean_entropy" in scorer_aucs:
        delta = scorer_aucs["mean_top_k_mass"].sign_aware - scorer_aucs["mean_entropy"].sign_aware
        if not corr_matrix.empty and "mean_top_k_mass" in corr_matrix and "mean_entropy" in corr_matrix:
            spearman_r = float(corr_matrix.loc["mean_top_k_mass", "mean_entropy"])
        else:
            spearman_r = float("nan")
        held = delta >= E2_DELTA_MIN and spearman_r >= E2_SPEARMAN_MIN
        outcomes.append(PredictionOutcome(
            name="E2",
            description="mean_top_k_mass underperforms mean_H by ≤0.05 AND Spearman ≥0.5",
            threshold=f"Δ ≥ {E2_DELTA_MIN}, Spearman ρ ≥ {E2_SPEARMAN_MIN}",
            measured=f"Δ_AUC = {delta:+.3f} (top_k_mass {scorer_aucs['mean_top_k_mass'].sign_aware:.3f} vs mean_H {scorer_aucs['mean_entropy'].sign_aware:.3f}); Spearman ρ = {spearman_r:.3f}",
            held=held,
        ))

    # E3: mean_gap_top2 AUC ≥ 0.60 AND bottom-decile lift ≥ +5pp
    if "mean_gap_top2" in scorer_aucs:
        gap_auc = scorer_aucs["mean_gap_top2"].sign_aware
        # Bottom-decile lift on raw mean_gap_top2 (ascending: bottom = smallest values = tightest competition)
        scores = df["mean_gap_top2"].to_numpy()
        mask = ~np.isnan(scores)
        threshold_decile = float(np.percentile(scores[mask], 10))
        in_bottom = mask & (scores <= threshold_decile)
        n_bottom = int(in_bottom.sum())
        n_wrong_bottom = int(((df.wrong == 1) & in_bottom).sum())
        base_wrong_rate = float(df.wrong.mean())
        decile_wrong_rate = n_wrong_bottom / n_bottom if n_bottom else 0.0
        lift_pp = (decile_wrong_rate - base_wrong_rate) * 100
        held = gap_auc >= E3_AUC_THRESHOLD and lift_pp >= E3_BOTTOM_DECILE_LIFT_MIN
        outcomes.append(PredictionOutcome(
            name="E3",
            description="mean_gap_top2 AUC ≥ 0.60 AND bottom-decile lift ≥ +5pp",
            threshold=f"sign-aware AUC ≥ {E3_AUC_THRESHOLD}, lift ≥ +{E3_BOTTOM_DECILE_LIFT_MIN}pp",
            measured=f"AUC = {gap_auc:.3f}; bottom-decile lift = {lift_pp:+.2f}pp ({n_wrong_bottom}/{n_bottom} wrong vs base {base_wrong_rate * 100:.1f}%)",
            held=held,
        ))

    # ---- Output ----
    pd.DataFrame([asdict(o) for o in outcomes]).to_csv(
        output_dir / "phase_b_predictions.csv", index=False
    )

    auc_rows = [
        {
            "scorer": r.name,
            "point_auc": r.point,
            "sign_aware_auc": r.sign_aware,
            "direction": r.direction,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
        }
        for r in scorer_aucs.values()
    ]
    pd.DataFrame(auc_rows).to_csv(
        output_dir / "phase_b_per_scorer_auc.csv", index=False
    )

    if not corr_matrix.empty:
        corr_matrix.to_csv(output_dir / "phase_b_correlations.csv")

    payload = {
        "artifact_dir": str(artifact_dir),
        "n_total": int(len(df)),
        "n_wrong": int(df.wrong.sum()),
        "n_correct": int((df.wrong == 0).sum()),
        "accuracy": float(1 - df.wrong.mean()),
        "n_bootstrap": n_boot,
        "predictions": [asdict(o) for o in outcomes],
        "per_scorer_auc": {k: asdict(v) for k, v in scorer_aucs.items()},
        "spearman_correlations": (
            corr_matrix.to_dict() if not corr_matrix.empty else {}
        ),
    }
    (output_dir / "phase_b_results.json").write_text(
        json.dumps(payload, indent=2)
    )

    # ---- Console summary ----
    print()
    print("=" * 90)
    print("  Phase-B re-derivation results")
    print("=" * 90)

    print("\nPer-scorer sign-aware AUC (95% bootstrap CI):")
    for r in sorted(
        scorer_aucs.values(),
        key=lambda x: -x.sign_aware if not math.isnan(x.sign_aware) else 0,
    ):
        print(
            f"  {r.name:30s} {r.sign_aware:.3f} "
            f"[{r.ci_low:.3f}, {r.ci_high:.3f}] {r.direction:>8s}"
        )

    if not corr_matrix.empty:
        print("\nSpearman ρ among scorers:")
        print(corr_matrix.round(3).to_string())

    print("\nPre-registered E1/E2/E3:")
    for o in outcomes:
        symbol = "+" if o.held else "-"
        status = "HOLDS" if o.held else "FAILS"
        print(f"  {symbol} {o.name} [{status}] {o.description}")
        print(f"     threshold: {o.threshold}")
        print(f"     measured:  {o.measured}")
    print()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path.home() / "work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.artifact_dir.parent / f"{args.artifact_dir.name}-phaseB"
    )
    analyse(args.artifact_dir, output_dir, args.n_bootstrap)


if __name__ == "__main__":
    cli()
