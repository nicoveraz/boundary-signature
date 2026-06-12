"""2D serial×parallel INTERACTION probe (cache-only, zero new inference).

Pre-registration: docs/decisions/prereg_2d_interaction_probe.md

Tests whether the 2D interaction term (mean_entropy : js_div) beats the
additive sum (mean_entropy + js_div) for predicting MLX-4bit wrong answers
on MedQA N=1273 — the go/no-go gate before funding a per-step dual-codec run.

Serial axis  (AU-analog): mean_entropy over reasoning steps, MLX trajectory.
Parallel axis (EU-analog): js_div, MLX-4bit vs GGUF-Q4_K_M terminal disagreement.
Label: y_wrong = 1 - (MLX argmax == gold).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 42
B_BOOT = 2000
ARTIFACTS = Path("/Users/nicoveraz/work/eunosia/artifacts")
STATES = ARTIFACTS / "medqa-stage-4a-n1273/condition_c_cached/states.parquet"
CROSSQ = ARTIFACTS / "medqa-cross-quant/cache.jsonl"


def shannon_bits(dist: dict[str, float]) -> float:
    ps = np.array([dist.get(k, 0.0) for k in ("A", "B", "C", "D")], dtype=float)
    s = ps.sum()
    if s <= 0:
        return float("nan")
    ps = ps / s
    return float(-sum(p * math.log2(p) for p in ps if p > 0))


def serial_features() -> pd.DataFrame:
    """Per-trajectory mean_entropy, entropy_slope, terminal_entropy (MLX)."""
    t = pq.read_table(STATES, columns=["trajectory_id", "timestep",
                                        "hypothesis_distribution_json"]).to_pandas()
    t = t[t["hypothesis_distribution_json"].notna()].copy()
    t["ent"] = t["hypothesis_distribution_json"].apply(
        lambda s: shannon_bits(json.loads(s)))
    t = t[t["ent"].notna()]
    rows = []
    for qid, g in t.groupby("trajectory_id"):
        g = g.sort_values("timestep")
        ent = g["ent"].to_numpy()
        ts = g["timestep"].to_numpy(dtype=float)
        slope = float(np.polyfit(ts, ent, 1)[0]) if len(ent) >= 2 else 0.0
        rows.append({"question_id": qid, "mean_entropy": float(ent.mean()),
                     "entropy_slope": slope, "terminal_entropy": float(ent[-1]),
                     "n_steps": len(ent)})
    return pd.DataFrame(rows)


def parallel_features() -> pd.DataFrame:
    recs = [json.loads(line) for line in CROSSQ.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(recs)
    return df[["question_id", "js_div", "disagree", "correct"]].copy()


def cv_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=1000)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


def paired_boot(y: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray) -> tuple:
    """Bootstrap CI of AUC(a) - AUC(b) by resampling questions."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == n:
            continue
        diffs.append(roc_auc_score(ys, oof_a[idx]) - roc_auc_score(ys, oof_b[idx]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, center - half, center + half)


def main() -> None:
    ser = serial_features()
    par = parallel_features()
    df = ser.merge(par, on="question_id", how="inner")
    print(f"Joined N = {len(df)} (serial {len(ser)}, parallel {len(par)})")

    df["y_wrong"] = 1 - df["correct"].astype(int)
    y = df["y_wrong"].to_numpy()
    print(f"wrong rate = {y.mean():.3f}  ({y.sum()}/{len(y)})")

    # Standardize the two base features; build interaction on standardized.
    Z = StandardScaler().fit_transform(df[["mean_entropy", "js_div"]].to_numpy())
    me, js = Z[:, 0], Z[:, 1]
    inter = me * js

    X0 = me.reshape(-1, 1)                      # M0 serial
    X1 = np.column_stack([me, js])              # M1 additive
    X2 = np.column_stack([me, js, inter])       # M2 interaction

    oof0, oof1, oof2 = cv_oof(X0, y), cv_oof(X1, y), cv_oof(X2, y)
    auc0 = roc_auc_score(y, oof0)
    auc1 = roc_auc_score(y, oof1)
    auc2 = roc_auc_score(y, oof2)

    print("\n=== CV AUC (5-fold, out-of-fold) ===")
    print(f"M0 mean_entropy           : {auc0:.4f}")
    print(f"M1 + js_div (additive)    : {auc1:.4f}")
    print(f"M2 + interaction (2D)     : {auc2:.4f}")

    lo_p, hi_p = paired_boot(y, oof2, oof1)
    lo_s, hi_s = paired_boot(y, oof1, oof0)
    print("\n=== PRIMARY: interaction over additive (threshold +0.02, CI excl 0) ===")
    print(f"ΔAUC(M2-M1) = {auc2 - auc1:+.4f}  95% CI [{lo_p:+.4f}, {hi_p:+.4f}]")
    signal = (auc2 - auc1) >= 0.02 and lo_p > 0
    print(f"VERDICT: {'SIGNAL — fund per-step run' if signal else 'NULL — recombination, stop'}")
    print("\n=== SECONDARY: additive over serial ===")
    print(f"ΔAUC(M1-M0) = {auc1 - auc0:+.4f}  95% CI [{lo_s:+.4f}, {hi_s:+.4f}]")

    print("\n=== DESCRIPTIVE: conflict-quadrant contingency ===")
    conf = df["mean_entropy"] < df["mean_entropy"].median()
    disag = df["js_div"] > df["js_div"].quantile(0.75)
    for clabel, cmask in [("confident", conf), ("uncertain", ~conf)]:
        for dlabel, dmask in [("disagree", disag), ("agree", ~disag)]:
            m = cmask & dmask
            k, n = int(df.loc[m, "y_wrong"].sum()), int(m.sum())
            p, lo, hi = wilson(k, n)
            print(f"  {clabel:9s} & {dlabel:8s}: wrong {p:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  (n={n})")

    print("\n=== REDUNDANCY: Spearman(js_div, mean_entropy) prior [0.3,0.7] ===")
    rho, _ = spearmanr(df["js_div"], df["mean_entropy"])
    print(f"rho = {rho:.3f}")

    out = ARTIFACTS / "medqa-2d-interaction-probe"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "joined_features.parquet")
    summary = {
        "n": int(len(df)), "wrong_rate": float(y.mean()),
        "auc_M0_serial": float(auc0), "auc_M1_additive": float(auc1),
        "auc_M2_interaction": float(auc2),
        "primary_delta_M2_M1": float(auc2 - auc1), "primary_ci": [lo_p, hi_p],
        "secondary_delta_M1_M0": float(auc1 - auc0), "secondary_ci": [lo_s, hi_s],
        "primary_verdict": "SIGNAL" if signal else "NULL",
        "spearman_js_meanent": float(rho), "seed": SEED, "n_boot": B_BOOT,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}/summary.json")


if __name__ == "__main__":
    main()
