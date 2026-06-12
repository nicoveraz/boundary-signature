"""Generate docs/figures/entropy_trajectory.png from cached measurements.

The "thermometer" figure: median per-step answer-entropy of eventually-correct
vs eventually-wrong trajectories on MedQA-USMLE (N=1273). Reproduces paper §5.5
(the boundary signal is present at the prior and narrows as reasoning resolves).
Reads only the embedding-stripped cached measurements shipped in measurements/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
STATES = ROOT / "measurements/medqa-stage-4a-n1273/condition_c_cached/states.parquet"
CROSSQ = ROOT / "measurements/medqa-cross-quant/cache.jsonl"
OUT = ROOT / "docs/figures/entropy_trajectory.png"
LETTERS = ("A", "B", "C", "D")
MAXPOS = 5


def ent_bits(d: dict) -> float:
    v = np.array([d.get(k, 0.0) for k in LETTERS], float)
    s = v.sum()
    if s <= 0:
        return float("nan")
    v = v / s
    return float(-sum(p * math.log2(p) for p in v if p > 0))


def main() -> None:
    gold = {json.loads(l)["question_id"]: json.loads(l)["gold"]
            for l in CROSSQ.read_text().splitlines() if l.strip()}

    t = pq.read_table(STATES, columns=["trajectory_id", "timestep",
                                       "hypothesis_distribution_json"]).to_pandas()
    t = t[t["hypothesis_distribution_json"].notna()].copy()
    t["ent"] = t["hypothesis_distribution_json"].apply(lambda s: ent_bits(json.loads(s)))

    # correctness = terminal argmax vs gold
    correct = {}
    for qid, g in t.groupby("trajectory_id"):
        g = g.sort_values("timestep")
        last = json.loads(g.iloc[-1]["hypothesis_distribution_json"])
        pred = max(last, key=lambda k: last.get(k, 0.0))
        if qid in gold:
            correct[qid] = (pred == gold[qid])

    pos = list(range(MAXPOS + 1))
    med_c, med_w = [], []
    for p in pos:
        rows = t[t["timestep"] == p]
        ec = [e for q, e in zip(rows["trajectory_id"], rows["ent"])
              if correct.get(q) is True and not math.isnan(e)]
        ew = [e for q, e in zip(rows["trajectory_id"], rows["ent"])
              if correct.get(q) is False and not math.isnan(e)]
        med_c.append(np.median(ec) if ec else np.nan)
        med_w.append(np.median(ew) if ew else np.nan)

    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(pos, med_w, "-o", color="#c0392b", lw=2.4, ms=7, label="eventually wrong")
    ax.plot(pos, med_c, "-o", color="#2471a3", lw=2.4, ms=7, label="eventually correct")
    ax.fill_between(pos, med_c, med_w, color="#c0392b", alpha=0.08)
    ax.set_xlabel("reasoning step (0 = prior, before any reasoning text)")
    ax.set_ylabel("median answer entropy (bits)")
    ax.set_title("The boundary signal is present at the prior\nMedQA-USMLE, N=1273 · Qwen2.5-7B-Instruct (4-bit)")
    ax.set_xticks(pos)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate(f"+{(med_w[0]-med_c[0]):.2f} bits gap at step 0",
                xy=(0, (med_w[0]+med_c[0])/2), xytext=(0.8, (med_w[0]+med_c[0])/2 + 0.1),
                fontsize=10, color="#7b241c")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("step:   " + "  ".join(f"{p}" for p in pos))
    print("wrong:  " + "  ".join(f"{m:.2f}" for m in med_w))
    print("correct:" + "  ".join(f"{m:.2f}" for m in med_c))


if __name__ == "__main__":
    main()
