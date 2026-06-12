#!/usr/bin/env python
"""Cross-quantization disagreement (E_quant_3) vs correctness on MedQA.

Tests the framework's one calibrated-novel signal — same-model cross-codec
disagreement as a deployment-cheap perturbation UQ axis — on a CLEAN
correctness target, the same structure as the semantic-entropy comparator.

Two codecs of Qwen2.5-7B-Instruct: MLX-4bit (in-process) and GGUF-Q4_K_M
(llama-server). Each queried with get_token_probabilities over the answer
letters at the IDENTICAL direct-answer prompt (1 constrained forward each).

On single-step MCQ, argmax disagreement is boolean per question (degenerate
2-point ROC), so the rankable standalone signal is the CONTINUOUS codec
divergence: Jensen-Shannon (primary) + L1 (secondary) between the two
renormalised letter distributions. The boolean disagreement is reported as a
contingency, not as the AUROC signal.

Honest prior (see pre-reg): stage-6 found cross-quant Spearman +0.507 with
mean_entropy (moderately redundant, GT-independent) — redundancy is the modal
expectation; the live question is incremental AUC ≥0.02 DESPITE it.

correct = MLX-4bit argmax vs gold (the deployed prediction). Resumable cache.
Pre-reg: docs/decisions/prereg_cross_quant_comparator.md.

Usage::
    # start GGUF server first:
    #   llama-server -m <Qwen2.5-7B-Instruct-Q4_K_M.gguf> --port 8085 -c 4096
    python 12_cross_quant_comparator.py --n-questions 150
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bsig.medqa.trajectory_sources.medqa import MedQAQuestionLoader

_PROMPT = "{q}\n{choices}\nThe single best answer is:"


def _entropy_bits(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _completed(cache: Path) -> set[str]:
    if not cache.exists():
        return set()
    return {json.loads(ln)["question_id"]
            for ln in cache.read_text().splitlines() if ln.strip()}


def main(args: argparse.Namespace) -> int:
    from bsig.reference.llm_llama_cpp import LlamaCppLLMAdapter
    from bsig.reference.llm_mlx import MLXLLMAdapter

    mlx = MLXLLMAdapter() if not args.mlx_model else MLXLLMAdapter(model=args.mlx_model)
    gguf = LlamaCppLLMAdapter(model=args.gguf_model, host=args.gguf_host)

    loader = MedQAQuestionLoader(split=args.split)
    records = []
    for r in loader.iter_records():
        records.append(r)
        if len(records) >= args.n_questions:
            break

    cache = Path(args.cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = _completed(cache)
    todo = [r for r in records if r.question_id not in done]
    print(f"[run] {len(done)} cached; {len(todo)} to do (N={args.n_questions})",
          flush=True)

    with cache.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            letters = sorted(rec.choices.keys())
            choices = "\n".join(f"{k}. {rec.choices[k]}" for k in letters)
            prompt = _PROMPT.format(q=rec.question, choices=choices)
            mlx_dist = mlx.get_token_probabilities(prompt, letters).distribution
            gguf_dist = gguf.get_token_probabilities(prompt, letters).distribution
            p = np.array([mlx_dist.get(k, 0.0) for k in letters], dtype=float)
            q = np.array([gguf_dist.get(k, 0.0) for k in letters], dtype=float)
            p = p / p.sum() if p.sum() > 0 else p
            q = q / q.sum() if q.sum() > 0 else q
            mlx_argmax = letters[int(p.argmax())]
            gguf_argmax = letters[int(q.argmax())]
            js = float(jensenshannon(p, q, base=2))          # JS distance
            js_div = 0.0 if math.isnan(js) else js ** 2       # JS divergence
            l1 = float(np.abs(p - q).sum())
            fh.write(json.dumps({
                "question_id": rec.question_id,
                "gold": rec.answer_letter,
                "mlx_argmax": mlx_argmax,
                "gguf_argmax": gguf_argmax,
                "correct": int(mlx_argmax == rec.answer_letter),
                "disagree": int(mlx_argmax != gguf_argmax),
                "mlx_entropy": _entropy_bits(p),
                "gguf_entropy": _entropy_bits(q),
                "js_div": js_div,
                "l1": l1,
                "mlx_dist": {k: float(v) for k, v in zip(letters, p, strict=True)},
                "gguf_dist": {k: float(v) for k, v in zip(letters, q, strict=True)},
            }) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[run] {i}/{len(todo)}", flush=True)

    _evaluate(cache, Path(args.semantic_cache))
    return 0


def _boot_ci(vals: list[float]) -> tuple[float, float]:
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _sa(y: np.ndarray, score: np.ndarray) -> tuple[float, str]:
    a = roc_auc_score(y, score)
    return max(a, 1 - a), ("greater" if a >= 0.5 else "less")


def _oriented(y: np.ndarray, score: np.ndarray, direction: str) -> float:
    a = roc_auc_score(y, score)
    return a if direction == "greater" else 1.0 - a


def _cv_oof(features: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_predict(clf, features, y, cv=5, method="predict_proba")[:, 1]


def _spearman_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    boot = [float(pd.Series(a[idx]).corr(pd.Series(b[idx]), method="spearman"))
            for idx in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    lo, hi = _boot_ci(boot)
    return rho, lo, hi


def _evaluate(cache: Path, semantic_cache: Path, n_boot: int = 5000, seed: int = 0) -> None:
    df = pd.DataFrame(
        json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()
    )
    # join semantic_entropy by question_id (if available)
    if semantic_cache.exists():
        sem = {json.loads(ln)["question_id"]: json.loads(ln).get("semantic_entropy")
               for ln in semantic_cache.read_text().splitlines() if ln.strip()}
        df["semantic_entropy"] = df["question_id"].map(sem)

    y_wrong = 1 - df["correct"].to_numpy()
    js = df["js_div"].to_numpy()
    l1 = df["l1"].to_numpy()
    ent = df["mlx_entropy"].to_numpy()
    n, n_wrong = len(df), int(y_wrong.sum())
    rng = np.random.default_rng(seed)

    js_auc, js_dir = _sa(y_wrong, js)
    l1_auc, l1_dir = _sa(y_wrong, l1)
    ent_auc, ent_dir = _sa(y_wrong, ent)

    # bootstrap CI for the standalone JSD AUC (oriented to point direction)
    js_b = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_wrong[idx]
        if yb.sum() in (0, n):
            continue
        js_b.append(_oriented(yb, js[idx], js_dir))
    js_lo, js_hi = _boot_ci(js_b)

    # incremental AUC: mean_entropy vs mean_entropy + JSD (5-fold CV OOF)
    can_cv = min(n_wrong, n - n_wrong) >= 5
    if can_cv:
        oof_base = _cv_oof(pd.DataFrame({"ent": ent}), y_wrong)
        oof_full = _cv_oof(pd.DataFrame({"ent": ent, "js": js}), y_wrong)
        base_auc = roc_auc_score(y_wrong, oof_base)
        full_auc = roc_auc_score(y_wrong, oof_full)
        inc_b = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb = y_wrong[idx]
            if yb.sum() in (0, n):
                continue
            inc_b.append(roc_auc_score(yb, oof_full[idx]) - roc_auc_score(yb, oof_base[idx]))
        inc_lo, inc_hi = _boot_ci(inc_b)
        p_inc_pos = float(np.mean(np.array(inc_b) > 0))
    else:
        base_auc = full_auc = inc_lo = inc_hi = p_inc_pos = float("nan")

    # agreement
    rho_ent, re_lo, re_hi = _spearman_ci(js, ent, n_boot, rng)
    if "semantic_entropy" in df and df["semantic_entropy"].notna().any():
        m = df["semantic_entropy"].notna().to_numpy()
        rho_sem, rs_lo, rs_hi = _spearman_ci(js[m], df["semantic_entropy"].to_numpy()[m],
                                             n_boot, rng)
        sem_n = int(m.sum())
    else:
        rho_sem = rs_lo = rs_hi = float("nan")
        sem_n = 0

    # boolean disagreement contingency
    dis = df["disagree"].to_numpy().astype(bool)
    wr_dis = y_wrong[dis].mean() if dis.any() else float("nan")
    wr_agr = y_wrong[~dis].mean() if (~dis).any() else float("nan")

    print("=" * 72)
    print("  Cross-quantization disagreement (E_quant_3) vs MedQA correctness")
    print("=" * 72)
    print(f"N={n}  accuracy {(n - n_wrong) / n * 100:.1f}%  (wrong={n_wrong})  "
          f"[{n_boot} bootstrap]  MLX-4bit vs GGUF-Q4_K_M")
    print("\n(1) STANDALONE (sign-aware AUC vs wrong-prediction)  threshold>=0.65")
    print(f"  codec JSD            {js_auc:.4f}  ({js_dir})  95% CI [{js_lo:.4f}, {js_hi:.4f}]")
    print(f"  codec L1             {l1_auc:.4f}  ({l1_dir})")
    print(f"  mean_entropy (ref)   {ent_auc:.4f}  ({ent_dir})")
    print("\n(2) INCREMENTAL over mean_entropy (5-fold CV logistic)  threshold>=0.02")
    print(f"  mean_entropy alone        AUC {base_auc:.4f}")
    print(f"  mean_entropy + JSD        AUC {full_auc:.4f}")
    print(f"  incremental: {full_auc - base_auc:+.4f}  95% CI [{inc_lo:+.4f}, {inc_hi:+.4f}]  "
          f"P(inc>0)={p_inc_pos:.2f}")
    print("\n(3) AGREEMENT  (prior: Spearman~0.507 w/ mean_entropy at stage-6)")
    print(f"  Spearman(JSD, mean_entropy):  {rho_ent:+.3f}  95% CI [{re_lo:+.3f}, {re_hi:+.3f}]")
    print(f"  Spearman(JSD, semantic_ent):  {rho_sem:+.3f}  "
          f"95% CI [{rs_lo:+.3f}, {rs_hi:+.3f}]  (n={sem_n})")
    print("\n(4) BOOLEAN argmax-disagreement contingency (descriptive)")
    print(f"  disagree on {int(dis.sum())}/{n} ({dis.mean()*100:.0f}%); "
          f"wrong-rate | disagree {wr_dis:.3f} vs | agree {wr_agr:.3f}")
    print("\nReading (precise — see pre-reg): standalone validates E_quant_3 only")
    print("  if JSD AUC>=0.65; incremental earns its complexity only if >=0.02 w/")
    print("  CI excluding 0. High Spearman w/ mean_entropy (prior ~0.5) = redundant.")
    print("  Correctness-prediction, NOT uncertainty-measurement. Does NOT test P4")
    print("  (graph-structural composite, dead on single-trajectory MCQ).")


def cli() -> None:
    base = Path.home() / "work" / "eunosia" / "artifacts"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument("--mlx-model", default=None,
                   help="MLX model id; default = adapter's Qwen2.5-7B-4bit")
    p.add_argument("--gguf-model", default="qwen2.5-7b-instruct-q4km",
                   help="GGUF model label (logging only; server serves the file)")
    p.add_argument("--n-questions", type=int, default=150)
    p.add_argument("--gguf-host", default="http://127.0.0.1:8085")
    p.add_argument("--cache-path",
                   default=str(base / "medqa-cross-quant" / "cache.jsonl"))
    p.add_argument("--semantic-cache",
                   default=str(base / "medqa-semantic-entropy" / "cache.jsonl"))
    raise SystemExit(main(p.parse_args()))


if __name__ == "__main__":
    cli()
