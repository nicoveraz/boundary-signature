#!/usr/bin/env python
"""Semantic-entropy vs single-run predictive-entropy on MedQA correctness.

The proven-method comparator (Anatomy/Kuhn use semantic entropy as the
core UQ metric). Tests the framework's compute-constraint claim head-to-
head: does CHEAP single-run predictive entropy (one forward over the
answer letters) predict correctness as well as EXPENSIVE N-sample
semantic entropy?

For MCQ, semantic equivalence = same answer letter (no NLI needed), so
semantic entropy = Shannon entropy over the answer-letter distribution
across N sampled full-CoT completions (reasoning paths diverge → answers
diverge). N-sample generation is serial (batch=1), so this runs on the
pinned mlx 0.31.2 — the rope batched-decode bug does not apply.

Per question:
  - single_run_entropy: entropy of get_token_probabilities over the answer
    letters at a direct-answer prompt (1 forward). predicted = argmax.
  - semantic_entropy: entropy over answer letters from N temp>0 CoT samples.
  - correct: single-run argmax == gold (the deployed prediction).

CONCEPTUAL SCOPE (precise, do not over-read). This comparator tests two
distinct questions and NOT a third:
  (1) CORRECTNESS-PREDICTION PARITY: sign-aware AUC of each signal vs
      correctness, with bootstrap CIs and a paired bootstrap on the gap.
      "Does cheap single-run predict correctness as well as expensive
      sampling?" — the deployment-relevant question.
  (2) SIGNAL AGREEMENT: Spearman(single_run, semantic) independent of
      correctness. "Do the two signals order questions the same way?"
  NOT tested: whether either signal MEASURES UNCERTAINTY as a latent
  construct. Correctness-correlation is correctness-prediction, not
  uncertainty-measurement; convergent validity / controlled-ambiguity
  manipulation / calibration-regime discrimination would be required and
  are out of scope here. On a 4-option MCQ semantic equivalence collapses
  to "same answer letter," so semantic entropy degenerates to
  letter-agreement entropy on a tiny shared support — a high (1)-AUC
  parity or (2) Spearman is therefore partly MECHANICAL, weak evidence for
  "measuring the same thing." Report all three numbers; claim only
  correctness-prediction parity + deployment cost. See pre-reg
  docs/decisions/prereg_semantic_entropy_comparator.md.

Usage::
    python 11_semantic_entropy_comparator.py --n-questions 200 --n-samples 8
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bsig.medqa.trajectory_sources.medqa import MedQAQuestionLoader

_ANSWER_RE = re.compile(r"(?i)answer\s*[:=]?\s*\(?\s*([A-E])\b")
_FALLBACK_RE = re.compile(r"\b([A-E])\b")


def _entropy_bits(probs) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def _extract_letter(text: str, letters: list[str]) -> str | None:
    m = list(_ANSWER_RE.finditer(text))
    cand = m[-1].group(1).upper() if m else None
    if cand is None:  # fallback: last standalone letter
        fm = list(_FALLBACK_RE.finditer(text))
        cand = fm[-1].group(1).upper() if fm else None
    return cand if cand in letters else None


def _completed(cache: Path) -> set[str]:
    if not cache.exists():
        return set()
    return {json.loads(ln)["question_id"]
            for ln in cache.read_text().splitlines() if ln.strip()}


def main(args: argparse.Namespace) -> int:
    from bsig.reference.llm_mlx import MLXLLMAdapter

    llm = MLXLLMAdapter() if not args.mlx_model else MLXLLMAdapter(model=args.mlx_model)
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
    print(f"[run] {len(done)} cached; {len(todo)} to do "
          f"(N={args.n_questions}, samples={args.n_samples})", flush=True)

    with cache.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            letters = sorted(rec.choices.keys())
            choices = "\n".join(f"{k}. {rec.choices[k]}" for k in letters)
            # single-run predictive entropy
            meas = llm.get_token_probabilities(
                f"{rec.question}\n{choices}\nThe single best answer is:",
                letters,
            )
            dist = meas.distribution
            predicted = max(dist, key=lambda k: dist[k])
            single_run_entropy = _entropy_bits(dist.values())
            # N-sample semantic entropy (full-CoT, temp>0)
            cot_prompt = (
                f"{rec.question}\n{choices}\nReason step by step, then end "
                f"with 'Answer: <letter>'.\n"
            )
            sampled: list[str] = []
            n_forced = 0
            for _s in range(args.n_samples):
                txt = llm.generate(cot_prompt, max_tokens=args.max_tokens,
                                   temperature=args.temperature)
                lt = _extract_letter(txt, letters)
                if lt is None and txt.strip():
                    # truncated/unparseable CoT -> forced extraction: argmax
                    # over the letters on the CoT continuation (a valid answer
                    # read, not a fabricated sample). Avoids the downward bias
                    # that dropping unparseable samples introduces.
                    fm = llm.get_token_probabilities(
                        cot_prompt + txt + "\nAnswer:", letters).distribution
                    lt = max(fm, key=lambda k: fm[k])
                    n_forced += 1
                if lt:
                    sampled.append(lt)
            if sampled:
                counts = pd.Series(sampled).value_counts(normalize=True)
                semantic_entropy = _entropy_bits(counts.to_numpy())
            else:
                semantic_entropy = float("nan")
            fh.write(json.dumps({
                "question_id": rec.question_id,
                "predicted": predicted,
                "gold": rec.answer_letter,
                "correct": int(predicted == rec.answer_letter),
                "single_run_entropy": single_run_entropy,
                "semantic_entropy": semantic_entropy,
                "n_valid_samples": len(sampled),
                "n_forced": n_forced,
            }) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[run] {i}/{len(todo)}", flush=True)

    _evaluate(cache)
    return 0


def _oriented_auc(y: np.ndarray, score: np.ndarray, direction: str) -> float:
    """AUC reoriented to a FIXED direction (sign-aware convention: resamples
    that flip direction land < 0.5, honestly reflecting direction uncertainty)."""
    a = roc_auc_score(y, score)
    return a if direction == "greater" else 1.0 - a


def _boot_ci(vals: list[float]) -> tuple[float, float]:
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _evaluate(cache: Path, n_boot: int = 5000, seed: int = 0) -> None:
    df = pd.DataFrame(
        json.loads(ln) for ln in cache.read_text().splitlines() if ln.strip()
    )
    df = df.dropna(subset=["semantic_entropy"])
    y_wrong = 1 - df["correct"].to_numpy()  # deferral target = prediction wrong
    sr = df["single_run_entropy"].to_numpy()
    se = df["semantic_entropy"].to_numpy()
    n, n_wrong = len(df), int(y_wrong.sum())

    def sa(score: np.ndarray) -> tuple[float, str]:
        a = roc_auc_score(y_wrong, score)
        return max(a, 1 - a), ("greater" if a >= 0.5 else "less")

    sr_auc, sr_dir = sa(sr)
    se_auc, se_dir = sa(se)
    rho = float(pd.Series(sr).corr(pd.Series(se), method="spearman"))

    # bootstrap: AUC CIs (sign-aware, reoriented to point direction),
    # paired lift CI + P(lift>0), and Spearman CI. Paired = same resample.
    rng = np.random.default_rng(seed)
    sr_b, se_b, lift_b, rho_b = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_wrong[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        a_sr = _oriented_auc(yb, sr[idx], sr_dir)
        a_se = _oriented_auc(yb, se[idx], se_dir)
        sr_b.append(a_sr)
        se_b.append(a_se)
        lift_b.append(a_se - a_sr)
        rho_b.append(float(pd.Series(sr[idx]).corr(pd.Series(se[idx]),
                                                    method="spearman")))
    sr_lo, sr_hi = _boot_ci(sr_b)
    se_lo, se_hi = _boot_ci(se_b)
    lift_lo, lift_hi = _boot_ci(lift_b)
    rho_lo, rho_hi = _boot_ci(rho_b)
    p_lift_pos = float(np.mean(np.array(lift_b) > 0))

    print("=" * 72)
    print("  Semantic entropy vs single-run predictive entropy (MedQA)")
    print("=" * 72)
    print(f"N={n}  accuracy {(n - n_wrong) / n * 100:.1f}%  (wrong={n_wrong})  "
          f"[{n_boot} bootstrap]")
    print("\n(1) CORRECTNESS-PREDICTION PARITY  (sign-aware AUC vs wrong-prediction)")
    print(f"  single_run_entropy   {sr_auc:.4f}  ({sr_dir})  "
          f"95% CI [{sr_lo:.4f}, {sr_hi:.4f}]")
    print(f"  semantic_entropy     {se_auc:.4f}  ({se_dir})  "
          f"95% CI [{se_lo:.4f}, {se_hi:.4f}]")
    print(f"  paired lift (semantic - single_run): {se_auc - sr_auc:+.4f}  "
          f"95% CI [{lift_lo:+.4f}, {lift_hi:+.4f}]  P(lift>0)={p_lift_pos:.2f}")
    print("\n(2) SIGNAL AGREEMENT  (independent of correctness)")
    print(f"  Spearman(single_run, semantic): {rho:+.3f}  "
          f"95% CI [{rho_lo:+.3f}, {rho_hi:+.3f}]")
    print("\nReading (precise — see module docstring + pre-reg):")
    print("  (1) is a CORRECTNESS-PREDICTION comparison. If the lift CI includes")
    print("      0, cheap single-run predicts correctness on par with expensive")
    print("      N-sample semantic entropy -> supports the compute-constraint")
    print("      claim (correctness-prediction parity at lower cost). It does NOT")
    print("      establish that either signal MEASURES uncertainty.")
    print("  (2) high Spearman = same ordering; but on 4-option MCQ semantic")
    print("      entropy ~ letter-agreement entropy, so agreement is partly")
    print("      mechanical -> weak convergent-validity evidence, not strong.")


def cli() -> None:
    base = Path.home() / "work" / "eunosia" / "artifacts" / "medqa-semantic-entropy"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="test")
    p.add_argument("--mlx-model", default=None,
                   help="MLX model id; default = adapter's Qwen2.5-7B-4bit")
    p.add_argument("--n-questions", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--cache-path", default=str(base / "cache.jsonl"))
    raise SystemExit(main(p.parse_args()))


if __name__ == "__main__":
    cli()
