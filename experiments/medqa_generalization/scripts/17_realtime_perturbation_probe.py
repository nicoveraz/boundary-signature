"""Real-time logit-perturbation probe (cache-only, HYPOTHESIS-GENERATING).

Question: can a CHEAP single-model perturbation reproduce the cross-quant
disagreement signal's additive +3pp gain, at ~1x cost (deployable real-time
in Eunosia on one M4 Pro, single model)?

Cheapest possible perturbation = additive Gaussian noise on the answer LOGITS
the model already produces (free; derivable from the cached terminal A/B/C/D
distribution). R noised copies -> mean pairwise JS = "perturbation disagreement".

STRONG PRIOR: NULL. Noise on one model's final logits carries NO independent
information (unlike a genuinely different 2nd quantization) — it is a
deterministic smear of the same distribution, so perturb-disagreement is
expected to be a monotone function of entropy => redundant with mean_entropy
=> NO incremental gain. A null here is INFORMATIVE: it rules out the
literally-free real-time path and points to costlier independent-information
perturbations (activation/input, ~2x forward). A positive would be a big win.

Reports, per noise sigma: corr(perturb_disagree, cross_quant_js) [does noise
mimic a real 2nd model?], corr(perturb_disagree, mean_entropy) [redundancy
threat], standalone AUC, incremental AUC over mean_entropy.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED, R, B_BOOT = 42, 16, 2000
SIGMAS = [0.5, 1.0, 2.0]
ART = Path("/Users/nicoveraz/work/eunosia/artifacts")
STATES = ART / "medqa-stage-4a-n1273/condition_c_cached/states.parquet"
CROSSQ = ART / "medqa-cross-quant/cache.jsonl"
LETTERS = ("A", "B", "C", "D")
_rng = np.random.default_rng(SEED)
_PAIRS = list(combinations(range(R), 2))


def ent_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def perturb_disagree(p: np.ndarray, sigma: float) -> float:
    """Mean pairwise JS over R logit-noised copies of answer distribution p."""
    logit = np.log(np.clip(p, 1e-9, None))
    noised = logit[None, :] + _rng.normal(0, sigma, size=(R, 4))
    e = np.exp(noised - noised.max(1, keepdims=True))
    dists = e / e.sum(1, keepdims=True)
    js = [jensenshannon(dists[i], dists[j], base=2) ** 2 for i, j in _PAIRS]
    return float(np.mean(js))


def cv_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv,
                             method="predict_proba")[:, 1]


def paired_incr(y, oof_full, oof_base):
    rng = np.random.default_rng(SEED)
    n = len(y); d = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        d.append(roc_auc_score(y[idx], oof_full[idx]) - roc_auc_score(y[idx], oof_base[idx]))
    return (roc_auc_score(y, oof_full) - roc_auc_score(y, oof_base),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def main() -> None:
    t = pq.read_table(STATES, columns=["trajectory_id", "timestep",
                                        "hypothesis_distribution_json"]).to_pandas()
    t = t[t["hypothesis_distribution_json"].notna()]
    rows = []
    for qid, g in t.groupby("trajectory_id"):
        g = g.sort_values("timestep")
        dists = []
        for s in g["hypothesis_distribution_json"]:
            d = json.loads(s)
            v = np.array([d.get(k, 0.0) for k in LETTERS], float)
            tot = v.sum()
            dists.append(v / tot if tot > 0 else np.full(4, 0.25))
        dists = np.array(dists)
        ents = [ent_bits(d) for d in dists]
        rows.append({"question_id": qid, "term_dist": dists[-1],
                     "mean_entropy": float(np.mean(ents)),
                     "pred": LETTERS[int(dists[-1].argmax())]})
    df = pd.DataFrame(rows)

    cq = {json.loads(l)["question_id"]: json.loads(l)
          for l in CROSSQ.read_text().splitlines() if l.strip()}
    df["gold"] = df["question_id"].map(lambda q: cq.get(q, {}).get("gold"))
    df["cross_quant_js"] = df["question_id"].map(lambda q: cq.get(q, {}).get("js_div"))
    df = df[df["gold"].notna()].copy()
    df["y_wrong"] = (df["pred"] != df["gold"]).astype(int)
    y = df["y_wrong"].to_numpy()
    print(f"N={len(df)}  wrong_rate={y.mean():.3f}")

    me = StandardScaler().fit_transform(df[["mean_entropy"]].to_numpy())
    oof_base = cv_auc(me, y)
    print(f"baseline mean_entropy CV-AUC = {roc_auc_score(y, oof_base):.4f}")
    # reference: real cross-quant js_div incremental (the +3pp to beat)
    Xcq = StandardScaler().fit_transform(df[["mean_entropy", "cross_quant_js"]].to_numpy())
    di, dlo, dhi = paired_incr(y, cv_auc(Xcq, y), oof_base)
    print(f"REFERENCE real cross-quant js_div incr = +{di:+.4f} [{dlo:+.4f},{dhi:+.4f}]\n")

    out = {"n": int(len(df)), "sigmas": {}}
    print(f"{'sigma':>6s} {'corr(perturb,xquant)':>20s} {'corr(perturb,ent)':>18s} "
          f"{'standalone':>11s} {'incr over mean_ent':>22s}")
    for sigma in SIGMAS:
        pd_sig = df["term_dist"].apply(lambda p: perturb_disagree(p, sigma)).to_numpy()
        r_xq = spearmanr(pd_sig, df["cross_quant_js"]).correlation
        r_ent = spearmanr(pd_sig, df["mean_entropy"]).correlation
        a = roc_auc_score(y, pd_sig); a = max(a, 1 - a)
        X = StandardScaler().fit_transform(np.column_stack([me[:, 0], pd_sig]))
        di, dlo, dhi = paired_incr(y, cv_auc(X, y), oof_base)
        flag = "*" if dlo > 0 else " "
        print(f"{sigma:6.1f} {r_xq:>20.3f} {r_ent:>18.3f} {a:>11.3f} "
              f"+{di:+.4f} [{dlo:+.4f},{dhi:+.4f}] {flag}")
        out["sigmas"][sigma] = {"corr_xquant": float(r_xq), "corr_entropy": float(r_ent),
                                "standalone_auc": float(a), "incr": float(di),
                                "incr_ci": [float(dlo), float(dhi)], "incr_sig": bool(dlo > 0)}

    outd = ART / "medqa-realtime-perturbation-probe"
    outd.mkdir(exist_ok=True)
    (outd / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {outd}/summary.json")
    print("* = incremental CI excludes 0 (would refute the null prior)")


if __name__ == "__main__":
    main()
