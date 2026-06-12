"""Generate docs/figures/protocol.png — schematic of the measurement protocol.

A left-to-right pipeline: question -> CoT generation -> per-step constrained
measurement of the answer distribution -> trajectory of distributions ->
per-trajectory scorer (mean entropy) -> deferral signal. Pure matplotlib,
no data dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/figures/protocol.png"

BOXES = [
    ("Question\n+ options A/B/C/D", "#eaf2f8"),
    ("Chain-of-thought\ngeneration", "#eaf2f8"),
    ("Per-step measurement\nread P(A,B,C,D) at each\nreasoning-step boundary", "#d4e6f1"),
    ("Trajectory of\ndistributions\n(step 0 … T)", "#d4e6f1"),
    ("Per-trajectory scorer\nmean entropy\n(+ plateau, distance, …)", "#d1f2eb"),
    ("Deferral signal\ndefer if high", "#fadbd8"),
]


def main() -> None:
    plt.rcParams.update({"font.size": 10.5, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    n = len(BOXES)
    w, h, gap = 1.55, 1.25, 0.42
    x = 0.0
    centers = []
    for label, color in BOXES:
        box = FancyBboxPatch((x, -h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#34495e", facecolor=color, zorder=2)
        ax.add_patch(box)
        ax.text(x + w / 2, 0, label, ha="center", va="center", zorder=3)
        centers.append(x + w / 2)
        x += w + gap

    for i in range(n - 1):
        a = FancyArrowPatch((centers[i] + w / 2, 0), (centers[i + 1] - w / 2, 0),
                            arrowstyle="-|>", mutation_scale=16, lw=1.4,
                            color="#34495e", zorder=1)
        ax.add_patch(a)

    # mini answer-distribution insets above measurement -> trajectory, getting peakier
    rng = np.linspace(0, 1, 4)
    dists = [[0.30, 0.28, 0.24, 0.18], [0.55, 0.25, 0.12, 0.08], [0.86, 0.08, 0.04, 0.02]]
    xs0 = centers[2] - w / 2
    span = (centers[3] + w / 2) - xs0
    for j, d in enumerate(dists):
        bx = xs0 + span * (0.12 + 0.38 * j)
        for k, p in enumerate(d):
            ax.add_patch(plt.Rectangle((bx + k * 0.075, 1.05), 0.06, p * 0.7,
                                       color="#2471a3", zorder=3))
        ax.text(bx + 0.15, 1.02, f"t={j}", ha="center", va="top", fontsize=8, color="#555")
    ax.text((xs0 + span / 2), 1.92, "answer distribution sharpens across steps",
            ha="center", fontsize=9, style="italic", color="#1a5276")

    ax.set_xlim(-0.3, x)
    ax.set_ylim(-1.1, 2.2)
    ax.axis("off")
    ax.set_title("The measurement protocol", fontsize=13, pad=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
