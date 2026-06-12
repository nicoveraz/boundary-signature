#!/usr/bin/env python
"""Interpretive diagnostics on the Phase-1 cross-quant caches (no new inference).

Adjudicates the apparent MODEL-DEPENDENCE of E_quant_3's incremental value
(Qwen redundant +0.011 / Spearman 0.73 vs Llama +0.028 / Spearman 0.64),
using the two full-N=1273 cross-quant caches. Three moves:

  1. Cross-model DIFFERENCE CIs (paired bootstrap — same questions/order, so
     paired by question_id): is ΔSpearman / Δincremental distinguishable from
     0? If the difference CIs include 0, "model-dependence" is over-read.
  2. Entropy/JSD dynamic range per model: if Qwen's mean_entropy is wider/more
     discriminative at baseline, JSD has less to add -> clean calibration
     mechanism for the difference.
  3. Conversion-pipeline divergence on AGREE & CORRECT cases (model certain +
     right, no quantization-induced answer uncertainty): if Llama's codecs
     diverge more there, the conversion-lineage contribution to disagreement
     is empirically supported (tests the report's own caveat).

Usage::  python 13_cross_model_diagnostics.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _load(p: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(ln) for ln in p.read_text().splitlines() if ln.strip())


def _spear(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def _oof(x: np.ndarray, feats: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_predict(clf, pd.DataFrame(feats), y, cv=5, method="predict_proba")[:, 1]


def _ci(v: list[float]) -> tuple[float, float]:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main(a: argparse.Namespace) -> int:
    q = _load(Path(a.qwen)).set_index("question_id")
    lm = _load(Path(a.llama)).set_index("question_id")
    shared = q.index.intersection(lm.index)
    q, lm = q.loc[shared], lm.loc[shared]
    n = len(shared)
    print(f"paired on {n} shared question_ids\n")

    jq, eq, cq, dq = (q.js_div.to_numpy(), q.mlx_entropy.to_numpy(),
                      q.correct.to_numpy(), q.disagree.to_numpy())
    jl, el, cl, dl = (lm.js_div.to_numpy(), lm.mlx_entropy.to_numpy(),
                      lm.correct.to_numpy(), lm.disagree.to_numpy())
    yq, yl = 1 - cq, 1 - cl

    # ---- point estimates ----
    rho_q, rho_l = _spear(jq, eq), _spear(jl, el)
    ofb_q = _oof(eq, {"e": eq}, yq)
    off_q = _oof(eq, {"e": eq, "j": jq}, yq)
    ofb_l = _oof(el, {"e": el}, yl)
    off_l = _oof(el, {"e": el, "j": jl}, yl)
    inc_q = roc_auc_score(yq, off_q) - roc_auc_score(yq, ofb_q)
    inc_l = roc_auc_score(yl, off_l) - roc_auc_score(yl, ofb_l)

    # ---- MOVE 1: paired bootstrap on the cross-model differences ----
    rng = np.random.default_rng(0)
    d_rho_b, d_inc_b = [], []
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, n)
        d_rho_b.append(_spear(jq[idx], eq[idx]) - _spear(jl[idx], el[idx]))
        if yq[idx].sum() in (0, n) or yl[idx].sum() in (0, n):
            continue
        iq = roc_auc_score(yq[idx], off_q[idx]) - roc_auc_score(yq[idx], ofb_q[idx])
        il = roc_auc_score(yl[idx], off_l[idx]) - roc_auc_score(yl[idx], ofb_l[idx])
        d_inc_b.append(iq - il)
    dr_lo, dr_hi = _ci(d_rho_b)
    di_lo, di_hi = _ci(d_inc_b)

    print("=" * 72)
    print("  MOVE 1 — cross-model DIFFERENCE CIs (paired bootstrap, same Qs)")
    print("=" * 72)
    print(f"  Spearman(JSD,mean_entropy):  Qwen {rho_q:+.3f}  Llama {rho_l:+.3f}")
    print(f"    Δ(Qwen-Llama) = {rho_q - rho_l:+.3f}  95% CI [{dr_lo:+.3f}, {dr_hi:+.3f}]  "
          f"{'EXCLUDES 0' if dr_lo > 0 or dr_hi < 0 else 'INCLUDES 0'}")
    print(f"  incremental AUC over mean_entropy:  Qwen {inc_q:+.4f}  Llama {inc_l:+.4f}")
    print(f"    Δ(Qwen-Llama) = {inc_q - inc_l:+.4f}  95% CI [{di_lo:+.4f}, {di_hi:+.4f}]  "
          f"{'EXCLUDES 0' if di_lo > 0 or di_hi < 0 else 'INCLUDES 0'}")

    # ---- MOVE 2: dynamic range ----
    def stats(x: np.ndarray) -> str:
        p = np.percentile(x, [10, 25, 50, 75, 90])
        return (f"mean {x.mean():.3f}  std {x.std():.3f}  "
                f"IQR [{p[1]:.3f},{p[3]:.3f}]  p10/p90 [{p[0]:.3f},{p[4]:.3f}]")
    print("\n" + "=" * 72)
    print("  MOVE 2 — dynamic range (does Qwen's mean_entropy spread more?)")
    print("=" * 72)
    print(f"  mean_entropy Qwen : {stats(eq)}")
    print(f"  mean_entropy Llama: {stats(el)}")
    print(f"  JSD          Qwen : {stats(jq)}")
    print(f"  JSD          Llama: {stats(jl)}")

    # ---- MOVE 3: conversion divergence on agree & correct ----
    def subset_js(j: np.ndarray, d: np.ndarray, c: np.ndarray) -> tuple[float, float, int]:
        mask = (d == 0) & (c == 1)
        s = j[mask]
        return (float(s.mean()), float(np.median(s)), int(mask.sum()))
    qm, qmd, qn = subset_js(jq, dq, cq)
    lmn, lmd, ln_ = subset_js(jl, dl, cl)
    print("\n" + "=" * 72)
    print("  MOVE 3 — codec JSD on AGREE & CORRECT (certain+right; tests caveat)")
    print("=" * 72)
    print(f"  Qwen : mean JSD {qm:.4f}  median {qmd:.4f}  (n={qn})")
    print(f"  Llama: mean JSD {lmn:.4f}  median {lmd:.4f}  (n={ln_})")
    print(f"  ratio Llama/Qwen mean JSD on agree+correct: {lmn / qm:.2f}x")
    print("\nReading: Move1 says whether the difference is real; Move2 whether a")
    print("calibration/spread mechanism explains it; Move3 whether baseline")
    print("conversion divergence (not codec-on-hard-cases) drives Llama's larger JSD.")
    return 0


def cli() -> None:
    base = Path.home() / "work" / "eunosia" / "artifacts"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--qwen", default=str(base / "medqa-cross-quant" / "cache.jsonl"))
    p.add_argument("--llama", default=str(base / "medqa-cross-quant-llama" / "cache.jsonl"))
    p.add_argument("--n-boot", type=int, default=5000)
    raise SystemExit(main(p.parse_args()))


if __name__ == "__main__":
    cli()
