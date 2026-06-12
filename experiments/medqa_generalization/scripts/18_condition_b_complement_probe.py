"""Condition-B verbalized-confidence complementarity probe (cache-only).

Real-time-compatible candidate: verbalized self-confidence is one extra
generation (single forward, no 2nd model) — fits the Eunosia constraint.
Question: does it ADD incremental signal over mean_entropy (0.686)?
Standalone B is weak (condition_comparison.csv: AUC 0.541) but it is
ORTHOGONAL (verbalized vs token-distribution), so it could complement.

Cross-run caveat: B-confidence is from B's CoT, mean_entropy from C's CoT;
same question, different generation — an approximation of "both signals on
one pass". Target = C terminal prediction wrong (deployment-relevant).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED, B_BOOT = 42, 2000
ART = Path("/Users/nicoveraz/work/eunosia/artifacts")
CDIR = ART / "medqa-stage-4a-n1273"
LETTERS = ("A", "B", "C", "D")


def ent_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cv_auc(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv,
                             method="predict_proba")[:, 1]


def paired_incr(y, full, base):
    rng = np.random.default_rng(SEED); n = len(y); d = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        d.append(roc_auc_score(y[idx], full[idx]) - roc_auc_score(y[idx], base[idx]))
    return (roc_auc_score(y, full) - roc_auc_score(y, base),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def boot_auc(y, s):
    rng = np.random.default_rng(SEED); n = len(y); v = []
    base = roc_auc_score(y, s); base = max(base, 1 - base)
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        a = roc_auc_score(y[idx], s[idx]); v.append(max(a, 1 - a))
    if not v:
        return base, float("nan"), float("nan")
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    # Condition B: verbalized confidence + B prediction
    bs = pq.read_table(CDIR / "condition_b_cached/states.parquet",
                       columns=["trajectory_id", "metadata_json"]).to_pandas()
    bt = pq.read_table(CDIR / "condition_b_cached/trajectories.parquet",
                       columns=["trajectory_id", "primary_label"]).to_pandas()
    bs["b_conf"] = bs["metadata_json"].apply(lambda s: json.loads(s).get("confidence"))
    b = bs[["trajectory_id", "b_conf"]].merge(
        bt.rename(columns={"primary_label": "b_pred"}), on="trajectory_id")

    # Condition C: mean_entropy + terminal prediction
    cs = pq.read_table(CDIR / "condition_c_cached/states.parquet",
                       columns=["trajectory_id", "timestep", "hypothesis_distribution_json"]).to_pandas()
    cs = cs[cs["hypothesis_distribution_json"].notna()]
    rows = []
    for qid, g in cs.groupby("trajectory_id"):
        g = g.sort_values("timestep")
        dd = []
        for s in g["hypothesis_distribution_json"]:
            dct = json.loads(s); v = np.array([dct.get(k, 0.0) for k in LETTERS], float)
            tot = v.sum(); dd.append(v / tot if tot > 0 else np.full(4, .25))
        dd = np.array(dd)
        rows.append({"trajectory_id": qid, "mean_entropy": float(np.mean([ent_bits(x) for x in dd])),
                     "c_pred": LETTERS[int(dd[-1].argmax())]})
    c = pd.DataFrame(rows)

    gold = {json.loads(l)["question_id"]: json.loads(l)["gold"]
            for l in (ART / "medqa-cross-quant/cache.jsonl").read_text().splitlines() if l.strip()}
    df = b.merge(c, on="trajectory_id")
    df["gold"] = df["trajectory_id"].map(gold)
    df = df[df["gold"].notna() & df["b_conf"].notna()].copy()
    df["b_conf"] = df["b_conf"].astype(float)
    df["y_wrong_c"] = (df["c_pred"] != df["gold"]).astype(int)
    df["y_wrong_b"] = (df["b_pred"] != df["gold"]).astype(int)
    print(f"N={len(df)}")
    print(f"b_conf: nunique={df['b_conf'].nunique()} "
          f"range=[{df['b_conf'].min():.2f},{df['b_conf'].max():.2f}] mean={df['b_conf'].mean():.3f}")

    print(f"B accuracy (b_pred==gold): {(df['b_pred'] == df['gold']).mean():.3f}  "
          f"C accuracy: {(df['c_pred'] == df['gold']).mean():.3f}")
    yc = df["y_wrong_c"].to_numpy()
    a, lo, hi = boot_auc(yc, df["b_conf"].to_numpy())
    print(f"\nB verbalized conf standalone (vs C-wrong): AUC {a:.3f} [{lo:.3f},{hi:.3f}]")
    me = StandardScaler().fit_transform(df[["mean_entropy"]].to_numpy())
    oof_base = cv_auc(me, yc)
    print(f"mean_entropy baseline (vs C-wrong): CV-AUC {roc_auc_score(yc, oof_base):.4f}")
    X = StandardScaler().fit_transform(df[["mean_entropy", "b_conf"]].to_numpy())
    di, dlo, dhi = paired_incr(yc, cv_auc(X, yc), oof_base)
    print(f"\nINCREMENTAL mean_entropy + b_conf over mean_entropy: "
          f"+{di:+.4f} [{dlo:+.4f},{dhi:+.4f}] {'*SIGNAL' if dlo > 0 else 'NULL'}")

    out = ART / "medqa-condition-b-probe"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"n": int(len(df)), "b_conf_nunique": int(df["b_conf"].nunique()),
         "b_conf_mean": float(df["b_conf"].mean()),
         "b_standalone_auc": float(a), "incr_over_mean_entropy": float(di),
         "incr_ci": [float(dlo), float(dhi)], "incr_sig": bool(dlo > 0)}, indent=2))
    print(f"\nWrote {out}/summary.json")


if __name__ == "__main__":
    main()
