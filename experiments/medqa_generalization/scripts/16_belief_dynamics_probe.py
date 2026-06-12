"""Belief-dynamics exploratory probe (cache-only, HYPOTHESIS-GENERATING).

NOT pre-registered, NOT confirmatory. Explores whether trajectory-DYNAMICS
features (movement of the answer distribution across reasoning steps) add
COMPLEMENTARY signal over the mean_entropy level baseline (AUC 0.686) on
MedQA-Qwen N=1273. Anything promising goes to a prereg + held-out validation.

Literature (2026-06): the dynamics CONSTRUCT is converged prior art (EDIS
2602.01288, JSD-volatility 2602.02863 @ AUC 0.66-0.74, Certaindex/ACR for
answer-stability-as-efficiency). All operate on a DIFFERENT distribution
object; this probe tests the cheap constrained-decoding answer-distribution
version. Realistic prior: comparable-not-superior, additive-at-best.

Features (all from cached per-step A/B/C/D hypothesis_distribution + mass_capture):
  mean_entropy        level baseline (sanity: should reproduce ~0.686)
  volatility_mean/max step-to-step JS divergence between consecutive dists
  flip_rate           argmax changes across steps / (n_steps-1)
  mono_viol_rate      fraction of steps where entropy INCREASES (Zhao anchor)
  margin_min/mean     top1-top2 of the answer distribution per step
  offmass_mean/slope  1 - mass_capture (mass outside A/B/C/D) level + trend
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED, B_BOOT = 42, 2000
ART = Path("/Users/nicoveraz/work/eunosia/artifacts")
STATES = ART / "medqa-stage-4a-n1273/condition_c_cached/states.parquet"
CROSSQ = ART / "medqa-cross-quant/cache.jsonl"
LETTERS = ("A", "B", "C", "D")


def ent_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def feats_for(g: pd.DataFrame) -> dict:
    g = g.sort_values("timestep")
    dists, masses = [], []
    for s, mc in zip(g["hypothesis_distribution_json"], g["mass_capture"]):
        d = json.loads(s)
        v = np.array([d.get(k, 0.0) for k in LETTERS], dtype=float)
        tot = v.sum()
        dists.append(v / tot if tot > 0 else np.full(4, 0.25))
        masses.append(float(mc) if mc is not None and not pd.isna(mc) else np.nan)
    dists = np.array(dists)
    ents = np.array([ent_bits(d) for d in dists])
    argmaxes = dists.argmax(1)
    n = len(dists)

    # volatility: JS divergence (bits) between consecutive answer distributions
    js = [float(jensenshannon(dists[i], dists[i + 1], base=2) ** 2)
          for i in range(n - 1)] or [0.0]
    # flips
    flips = int((np.diff(argmaxes) != 0).sum())
    # monotonicity violations: entropy goes UP step-to-step
    mono_viol = int((np.diff(ents) > 1e-9).sum())
    # margin top1-top2 of answer distribution
    sd = np.sort(dists, 1)
    margins = sd[:, -1] - sd[:, -2]
    # off-answer mass
    offmass = 1.0 - np.array(masses)
    off_valid = offmass[~np.isnan(offmass)]
    off_slope = (float(np.polyfit(np.arange(len(off_valid)), off_valid, 1)[0])
                 if len(off_valid) >= 2 else 0.0)

    return {
        "mean_entropy": float(ents.mean()),
        "volatility_mean": float(np.mean(js)),
        "volatility_max": float(np.max(js)),
        "flip_rate": flips / max(n - 1, 1),
        "mono_viol_rate": mono_viol / max(n - 1, 1),
        "margin_min": float(margins.min()),
        "margin_mean": float(margins.mean()),
        "offmass_mean": float(np.nanmean(offmass)) if len(off_valid) else 0.0,
        "offmass_slope": off_slope,
        "pred": LETTERS[argmaxes[-1]],
    }


def boot_auc_ci(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y, s)
    base = max(base, 1 - base)
    n = len(y)
    vals = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        a = roc_auc_score(y[idx], s[idx])
        vals.append(max(a, 1 - a))
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cv_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv,
                             method="predict_proba")[:, 1]


def paired_incr(y, oof_full, oof_base) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    d = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        d.append(roc_auc_score(y[idx], oof_full[idx]) - roc_auc_score(y[idx], oof_base[idx]))
    return (roc_auc_score(y, oof_full) - roc_auc_score(y, oof_base),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main() -> None:
    t = pq.read_table(STATES, columns=["trajectory_id", "timestep",
                                        "hypothesis_distribution_json", "mass_capture"]).to_pandas()
    t = t[t["hypothesis_distribution_json"].notna()]
    rows = {qid: feats_for(g) for qid, g in t.groupby("trajectory_id")}
    df = pd.DataFrame(rows).T.reset_index(names="question_id")

    gold = {json.loads(l)["question_id"]: json.loads(l)["gold"]
            for l in CROSSQ.read_text().splitlines() if l.strip()}
    df["gold"] = df["question_id"].map(gold)
    df = df[df["gold"].notna()].copy()
    df["y_wrong"] = (df["pred"] != df["gold"]).astype(int)
    y = df["y_wrong"].to_numpy()
    print(f"N={len(df)}  wrong_rate={y.mean():.3f}")

    feat_cols = ["mean_entropy", "volatility_mean", "volatility_max", "flip_rate",
                 "mono_viol_rate", "margin_min", "margin_mean", "offmass_mean", "offmass_slope"]
    for c in feat_cols:
        df[c] = df[c].astype(float)

    me = StandardScaler().fit_transform(df[["mean_entropy"]].to_numpy())
    oof_base = cv_auc(me, y)
    auc_base = roc_auc_score(y, oof_base)
    print(f"\nBaseline mean_entropy CV-AUC = {auc_base:.4f}  (paper: 0.686)\n")

    print(f"{'feature':16s} {'standalone(sign)':>20s} {'incr over mean_ent':>26s}")
    results = {}
    for c in feat_cols:
        s = df[c].to_numpy()
        a, lo, hi = boot_auc_ci(y, s)
        if c == "mean_entropy":
            print(f"{c:16s} {a:.3f} [{lo:.3f},{hi:.3f}]   (baseline)")
            results[c] = {"standalone": a, "ci": [lo, hi]}
            continue
        X = StandardScaler().fit_transform(df[["mean_entropy", c]].to_numpy())
        oof = cv_auc(X, y)
        di, dlo, dhi = paired_incr(y, oof, oof_base)
        flag = "*" if dlo > 0 else " "
        print(f"{c:16s} {a:.3f} [{lo:.3f},{hi:.3f}]   "
              f"+{di:+.4f} [{dlo:+.4f},{dhi:+.4f}] {flag}")
        results[c] = {"standalone": a, "ci": [lo, hi],
                      "incr": di, "incr_ci": [dlo, dhi], "incr_sig": dlo > 0}

    # all-dynamics composite over mean_entropy
    dyn = [c for c in feat_cols if c != "mean_entropy"]
    Xall = StandardScaler().fit_transform(df[["mean_entropy"] + dyn].to_numpy())
    oof_all = cv_auc(Xall, y)
    di, dlo, dhi = paired_incr(y, oof_all, oof_base)
    print(f"\nALL dynamics + mean_entropy: CV-AUC {roc_auc_score(y, oof_all):.4f}  "
          f"incr +{di:+.4f} [{dlo:+.4f},{dhi:+.4f}] {'*' if dlo > 0 else ''}")

    out = ART / "medqa-belief-dynamics-probe"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "features.parquet")
    (out / "summary.json").write_text(json.dumps(
        {"n": len(df), "auc_baseline": auc_base, "features": results,
         "all_dyn_incr": di, "all_dyn_incr_ci": [dlo, dhi]}, indent=2, default=float))
    print(f"\nWrote {out}/summary.json")
    print("\n* = incremental CI excludes 0 (HYPOTHESIS for prereg, not a result)")


if __name__ == "__main__":
    main()
