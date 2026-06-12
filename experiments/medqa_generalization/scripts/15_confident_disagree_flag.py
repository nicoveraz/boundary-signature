"""Confident-but-disagreeing triage flag — cross-model held-out validation.

Pre-registration: docs/decisions/prereg_confident_disagree_triage_flag.md

Parameter-free, label-free flag:
  confident = terminal_entropy < within-model median
  disagree  = js_div > within-model 75th percentile
  flag      = confident AND disagree

Discovery (contaminated): MedQA-Qwen N=1273.
Held-out test: MedQA-Llama N=1273 (different base model).
Primary: on Llama, lift = wrong(flag) - wrong(confident&agree) >= 0.15 AND
flag-cell Wilson CI lower bound > Llama base wrong-rate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ARTIFACTS = Path("/Users/nicoveraz/work/eunosia/artifacts")
CACHES = {
    "Qwen (discovery)": ARTIFACTS / "medqa-cross-quant/cache.jsonl",
    "Llama (held-out)": ARTIFACTS / "medqa-cross-quant-llama/cache.jsonl",
}


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p, z = k / n, 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, center - half, center + half)


def load(path: Path) -> pd.DataFrame:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df["y_wrong"] = 1 - df["correct"].astype(int)
    return df


def analyze(name: str, df: pd.DataFrame) -> dict:
    base_k, base_n = int(df["y_wrong"].sum()), len(df)
    base_p, base_lo, base_hi = wilson(base_k, base_n)
    conf = df["mlx_entropy"] < df["mlx_entropy"].median()
    disag = df["js_div"] > df["js_div"].quantile(0.75)

    print(f"\n=== {name}  (N={base_n}, base wrong-rate {base_p:.3f} "
          f"[{base_lo:.3f},{base_hi:.3f}]) ===")
    cells = {}
    for cl, cm in [("confident", conf), ("uncertain", ~conf)]:
        for dl, dm in [("disagree", disag), ("agree", ~disag)]:
            m = cm & dm
            k, n = int(df.loc[m, "y_wrong"].sum()), int(m.sum())
            p, lo, hi = wilson(k, n)
            cells[f"{cl}&{dl}"] = {"wrong": p, "lo": lo, "hi": hi, "n": n}
            print(f"  {cl:9s} & {dl:8s}: wrong {p:.3f} [{lo:.3f},{hi:.3f}]  (n={n})")

    flag = cells["confident&disagree"]
    ref = cells["confident&agree"]
    lift = flag["wrong"] - ref["wrong"]
    validates = lift >= 0.15 and flag["lo"] > base_p
    print(f"  -> flag lift (flag - confident&agree) = {lift:+.3f}")
    print(f"  -> flag-cell CI low {flag['lo']:.3f} vs base {base_p:.3f}: "
          f"{'EXCLUDES' if flag['lo'] > base_p else 'overlaps'}")
    return {"name": name, "base_rate": base_p, "cells": cells,
            "lift": lift, "validates_criteria": bool(validates)}


def main() -> None:
    results = {}
    for name, path in CACHES.items():
        results[name] = analyze(name, load(path))

    held = results["Llama (held-out)"]
    print("\n=== PRIMARY VERDICT (held-out Llama) ===")
    print(f"lift = {held['lift']:+.3f} (threshold +0.15), "
          f"flag-cell CI excludes base = "
          f"{held['cells']['confident&disagree']['lo'] > held['base_rate']}")
    verdict = "VALIDATES — model-portable triage flag" if held["validates_criteria"] \
        else "FAILS TO REPLICATE — Qwen/medical-specific"
    print(f"VERDICT: {verdict}")

    out = ARTIFACTS / "medqa-confident-disagree-flag"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {**results, "primary_verdict": verdict}, indent=2))
    print(f"\nWrote {out}/summary.json")


if __name__ == "__main__":
    main()
