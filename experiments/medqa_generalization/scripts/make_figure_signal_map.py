"""Generate docs/figures/signal_map.png — the negative-results map.

Incremental AUC over `mean_entropy` for every cheap signal tested (MedQA
N=1273). Only cross-quantization disagreement clears zero; everything else
straddles it. Values trace to the probe artifacts under measurements/:
  - cross-quant additive + 2D interaction : medqa-2d-interaction-probe/summary.json
  - trajectory dynamics (composite)       : medqa-belief-dynamics-probe/summary.json
  - verbalised confidence                 : medqa-condition-b-probe/summary.json
  - logit-noise perturbation (sigma=2)    : medqa-realtime-perturbation-probe/summary.json
  - richer distribution (composite)       : medqa-richdist-capture/run.log
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/figures/signal_map.png"

# (label, delta_auc, ci_low, ci_high)
SIGNALS = [
    ("Cross-quantization disagreement", 0.031, 0.017, 0.045),
    ("Trajectory dynamics\n(volatility · flips · monotonicity · margin)", 0.005, -0.008, 0.019),
    ("Logit-noise perturbation (one model)", 0.004, -0.001, 0.009),
    ("2D serial × parallel interaction", 0.001, -0.003, 0.005),
    ("Verbalised confidence", -0.001, -0.003, 0.001),
    ("Richer distribution\n(varentropy · full-vocab entropy · EPR)", -0.016, -0.035, 0.000),
]


def main() -> None:
    plt.rcParams.update({"font.size": 11.5, "figure.dpi": 150})
    labels = [s[0] for s in SIGNALS]
    vals = np.array([s[1] for s in SIGNALS])
    lo = np.array([s[2] for s in SIGNALS])
    hi = np.array([s[3] for s in SIGNALS])
    y = np.arange(len(SIGNALS))[::-1]  # first signal on top
    err = np.vstack([vals - lo, hi - vals])

    # cross-quant (CI excludes 0) highlighted; the rest grey
    colors = ["#2471a3" if (lo[i] > 0) else "#95a5a6" for i in range(len(SIGNALS))]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.barh(y, vals, color=colors, height=0.6, zorder=2)
    ax.errorbar(vals, y, xerr=err, fmt="none", ecolor="#34495e",
                elinewidth=1.4, capsize=4, zorder=3)
    ax.axvline(0, color="#2c3e50", lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ AUC over mean_entropy   (0 = no gain; 95% CI)")
    ax.set_title("Can anything cheap beat mean entropy?\nMedQA-USMLE N=1273 · single model, single forward pass, 4-bit")
    ax.set_xlim(-0.05, 0.06)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    # annotate the only winner
    top_y = y[0]
    ax.annotate("the only additive gain —\nbut needs a 2nd model,\nand label-sensitive",
                xy=(0.031, top_y), xytext=(0.040, top_y - 0.9),
                fontsize=9.5, color="#1a5276",
                arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
