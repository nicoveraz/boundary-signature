"""Richer-distribution feature capture (NEW INFERENCE, MLX, schema-v3 top-K).

Pre-registration: docs/decisions/prereg_richdist_features.md

Captures the FULL top-K next-token distribution per Condition-C measurement
position (the object the N=1273 v2 cache discarded), to test whether
varentropy / full-vocab entropy / EPR-rate complement mean_entropy.

Reuses ConditionC's exact prompt construction + Decomposer so the
mean_entropy baseline matches the paper protocol; captures raw
TokenProbabilityResults (with top_k_logprobs) instead of building the
trajectory/graph.

Usage:
    python 19_richdist_capture.py --smoke            # N=3 sanity
    python 19_richdist_capture.py --n 200            # the go/no-go run
    python 19_richdist_capture.py --analyze-only     # re-run analysis on JSONL
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path

import numpy as np

OUT = Path("/Users/nicoveraz/work/eunosia/artifacts/medqa-richdist-capture")
JSONL = OUT / "per_question.jsonl"
LETTERS = ("A", "B", "C", "D")


def ent_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


def varentropy_nats(top_k_logprobs: dict) -> float:
    """Variance of surprisal over the renormalised top-K distribution."""
    lp = np.array(list(top_k_logprobs.values()), dtype=float)
    if lp.size == 0:
        return 0.0
    p = np.exp(lp - lp.max())
    p = p / p.sum()
    info = -np.log(np.clip(p, 1e-12, None))
    H = float((p * info).sum())
    return float((p * info**2).sum() - H**2)


def entfull_nats(top_k_logprobs: dict) -> float:
    lp = np.array(list(top_k_logprobs.values()), dtype=float)
    if lp.size == 0:
        return 0.0
    p = np.exp(lp - lp.max()); p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_capture(n: int, smoke: bool) -> None:
    from bsig.medqa import (Decomposer, MCQActionCanonicalizer,
                            MCQStateCanonicalizer, MedQAQuestionLoader)
    from bsig.medqa.conditions.condition_c import ConditionC
    from bsig.reference.llm_mlx import MLXLLMAdapter

    class _StubEmb:
        def embed(self, text): return np.zeros(8)
        def embed_batch(self, texts): return np.zeros((len(texts), 8))
        def get_metadata(self): return {}
        @property
        def dimension(self): return 8

    stub = _StubEmb()
    llm = MLXLLMAdapter(logprobs_top_k=64)
    cc = ConditionC(
        llm=llm,
        state_canonicalizer=MCQStateCanonicalizer(stub),
        action_canonicalizer=MCQActionCanonicalizer(stub),
        embedder=stub,
        decomposer=Decomposer(),
    )

    OUT.mkdir(exist_ok=True)
    done = set()
    if JSONL.exists() and not smoke:
        done = {json.loads(l)["question_id"]
                for l in JSONL.read_text().splitlines() if l.strip()}
    print(f"already done: {len(done)}")

    loader = MedQAQuestionLoader(split="test")
    records = list(islice(loader.iter_records(), n))
    mode = "a" if (JSONL.exists() and not smoke) else "w"
    sink = JSONL if not smoke else OUT / "smoke.jsonl"
    fh = open(sink, mode)
    ok = fail = 0
    for i, rec in enumerate(records):
        if rec.question_id in done:
            continue
        try:
            prompt = cc._format_initial_prompt(rec)
            raw = llm.generate(prompt)
            decomp = cc._decomposer.decompose(raw)
            if len(decomp.reasoning_steps) == 0:
                fail += 1
                continue
            prompts = cc._build_measurement_prompts(rec, list(decomp.reasoning_steps))
            meas = llm.get_token_probabilities_batch(prompts, sorted(rec.choices.keys()))
            ent4, entf, vent = [], [], []
            for m in meas:
                d = np.array([m.distribution.get(k, 0.0) for k in LETTERS], float)
                s = d.sum(); d = d / s if s > 0 else np.full(4, .25)
                ent4.append(ent_bits(d))
                tk = dict(m.top_k_logprobs)
                entf.append(entfull_nats(tk))
                vent.append(varentropy_nats(tk))
            term = meas[-1].distribution
            pred = max(term, key=lambda k: term[k])
            row = {"question_id": rec.question_id, "gold": rec.answer_letter,
                   "pred": pred, "n_steps": len(meas),
                   "ent4": ent4, "entfull": entf, "varent": vent}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            ok += 1
            if (ok % 10) == 0:
                print(f"  {ok} done ({i+1}/{len(records)})")
        except Exception as exc:  # noqa: BLE001 — fail loud per question, keep going
            print(f"  FAIL {rec.question_id}: {type(exc).__name__}: {exc}")
            fail += 1
    fh.close()
    print(f"capture done: ok={ok} fail={fail} -> {sink}")
    if smoke:
        for l in sink.read_text().splitlines():
            r = json.loads(l)
            print(f"  {r['question_id']} steps={r['n_steps']} pred={r['pred']} "
                  f"gold={r['gold']} ent4_mean={np.mean(r['ent4']):.3f} "
                  f"entfull_mean={np.mean(r['entfull']):.3f} varent_mean={np.mean(r['varent']):.3f}")


def analyze() -> None:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    df = []
    for r in rows:
        ef = np.array(r["entfull"]); ts = np.arange(len(ef))
        df.append({
            "y_wrong": int(r["pred"] != r["gold"]),
            "mean_entropy": float(np.mean(r["ent4"])),
            "entropy_full": float(np.mean(r["entfull"])),
            "varentropy": float(np.mean(r["varent"])),
            "entfull_slope": float(np.polyfit(ts, ef, 1)[0]) if len(ef) >= 2 else 0.0,
        })
    import pandas as pd
    d = pd.DataFrame(df)
    y = d["y_wrong"].to_numpy()
    print(f"\nN={len(d)} wrong_rate={y.mean():.3f}")

    def cv(X):
        cvk = StratifiedKFold(5, shuffle=True, random_state=42)
        return cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cvk,
                                 method="predict_proba")[:, 1]

    def boot_incr(full, base):
        rng = np.random.default_rng(42); n = len(y); dd = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            if y[idx].sum() in (0, n):
                continue
            dd.append(roc_auc_score(y[idx], full[idx]) - roc_auc_score(y[idx], base[idx]))
        return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))

    me = StandardScaler().fit_transform(d[["mean_entropy"]].to_numpy())
    oof_base = cv(me); auc_base = roc_auc_score(y, oof_base)
    sa = max(auc_base, 1 - auc_base)
    print(f"SANITY mean_entropy sign-aware AUC = {sa:.4f}  (expect [0.62,0.74])")
    if not (0.62 <= sa <= 0.74):
        print("  !! SANITY FAIL — lean replication diverged from paper protocol")

    print(f"\n{'feature':16s} {'standalone':>11s} {'incr over mean_ent':>24s} {'spearman':>9s}")
    for c in ["entropy_full", "varentropy", "entfull_slope"]:
        s = d[c].to_numpy()
        a = roc_auc_score(y, s); a = max(a, 1 - a)
        X = StandardScaler().fit_transform(d[["mean_entropy", c]].to_numpy())
        oof = cv(X); di = roc_auc_score(y, oof) - auc_base
        lo, hi = boot_incr(oof, oof_base)
        rho = spearmanr(s, d["mean_entropy"]).correlation
        flag = "*SIGNAL" if (di >= 0.02 and lo > 0) else "null"
        print(f"{c:16s} {a:>11.3f} {di:>+8.4f} [{lo:+.4f},{hi:+.4f}] {rho:>9.3f} {flag}")

    # all three together
    X = StandardScaler().fit_transform(
        d[["mean_entropy", "entropy_full", "varentropy", "entfull_slope"]].to_numpy())
    oof = cv(X); di = roc_auc_score(y, oof) - auc_base; lo, hi = boot_incr(oof, oof_base)
    print(f"\nALL three + mean_entropy: incr {di:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'*SIGNAL' if (di >= 0.02 and lo > 0) else 'null'}")
    (OUT / "analysis_done.txt").write_text(f"sanity_auc={sa:.4f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only:
        analyze(); return
    run_capture(3 if a.smoke else a.n, a.smoke)
    if not a.smoke:
        analyze()


if __name__ == "__main__":
    main()
